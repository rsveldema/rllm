#include <JsonTensorHelpers.hpp>
#include <RandomHelpers.hpp>
#include <TransformerBlock.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>

namespace rllm
{
    static_assert(
        static_cast<int>(EmbeddingDimension::MAX) % static_cast<int>(HeadsIndex::MAX) == 0,
        "EmbeddingDimension::MAX must be divisible by HeadsIndex::MAX"
    );

    // ── randomize ─────────────────────────────────────────────────────────────

    void TransformerBlock::randomize()
    {
        const float sd = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(EmbeddingDimension::MAX)));
        const float sf = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(FFDimension::MAX)));

        W_q.fill_rand(-sd, sd);
        W_k.fill_rand(-sd, sd);
        W_v.fill_rand(-sd, sd);
        W_o.fill_rand(-sd, sd);
        W_gate.fill_rand(-sd, sd);
        W_up.fill_rand(-sd, sd);
        W_down.fill_rand(-sf, sf);

        V_q.fill(RLMM_ZERO);
        V_k.fill(RLMM_ZERO);
        V_v.fill(RLMM_ZERO);
        V_o.fill(RLMM_ZERO);
        V_gate.fill(RLMM_ZERO);
        V_up.fill(RLMM_ZERO);
        V_down.fill(RLMM_ZERO);
    }

    // ── normalisation ──────────────────────────────────────────────────────────

    // y_t = x_t / rms(x_t)  for each row t
    void TransformerBlock::rms_norm(
        // OFFLOAD_PARAMETERS(x, y)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& y
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(t, enum_iterator<PositionIndex>(x.num_rows()), (x, y))
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        float sq = 0.f;
        RLLM_OMP_SIMD_REDUCTION_PLUS(sq)
        for (const auto i : enum_iterator<EmbeddingDimension>())
        {
            const rlmm_float val = x[t, i];
            sq += val * val;
        }
        const float inv = 1.0f / std::sqrt(sq / fd + eps);
        for (const auto i : enum_iterator<EmbeddingDimension>())
            y[t, i] = x[t, i] * inv;
        ENDFOR
    }

    // dx += dL/dx  given dy = dL/dy and the original x (not the normalised y).
    // Per row:  dx_j += (1/rms) * (dy_j  -  y_j * mean(dy · y))
    void TransformerBlock::rms_norm_backward(
        // OFFLOAD_PARAMETERS(dy, x, dx)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dy,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dx
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(t, enum_iterator<PositionIndex>(x.num_rows()), (dy, x, dx))
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        rlmm_float sq = 0.f;
        for (const auto i : enum_iterator<EmbeddingDimension>())
            sq += x[t, i] * x[t, i];
        const rlmm_float inv = 1.0f / std::sqrt(sq / fd + eps);

        // dot = mean(dy · y) = (1/d) * sum_i dy[i] * x[i] * inv
        rlmm_float dot = 0.f;
        for (const auto i : enum_iterator<EmbeddingDimension>())
            dot += dy[t, i] * x[t, i] * inv;
        dot /= fd;

        for (const auto i : enum_iterator<EmbeddingDimension>())
            dx[t, i] += inv * (dy[t, i] - x[t, i] * inv * dot);
        ENDFOR
    }

    // ── attention helpers ──────────────────────────────────────────────────────

    /** In-place causal softmax over a [T × T] score matrix (row i only attends
     * to positions j ≤ i).  stride is the distance between rows in floats.
     *
     * @param x The [T × T] matrix of scores to softmax in-place.  Only the top-left [T × T] block is accessed and
     * modified; the rest of the matrix is left as-is to avoid unnecessary writes.
     * @param T The sequence length (the active size of the [T × T] block).
     */
    void TransformerBlock::causal_softmax(
        // OFFLOAD_PARAMETERS(x, T)
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& x,
        PositionIndex T
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator<PositionIndex>(T), (x, T))
        rlmm_float max_val = x[i, PositionIndex::START];
        for (const auto j : enum_iterator<PositionIndex>(inc(PositionIndex::START), inc(i)))
            max_val = math::max(max_val, x[i, j]);
        rlmm_float sum_exp = 0.f;
        // Compute and sum the exponentials for the active [T × T] block of the row; leave the rest of the row as-is
        // to avoid unnecessary writes.
        for (const auto j : enum_iterator<PositionIndex>(inc(i)))
        {
            x[i, j] = static_cast<rlmm_float>(std::exp(x[i, j] - max_val));
            sum_exp += x[i, j];
        }
        const rlmm_float inv = 1.0f / sum_exp;
        // scale the active [T × T] block of the row; leave the rest of the row as-is to avoid unnecessary writes
        for (const auto j : enum_iterator<PositionIndex>(inc(i)))
            x[i, j] *= static_cast<rlmm_float>(inv);

        // clear from the active [T × T] block to the end of the row
        // to avoid stale values affecting the backward pass
        for (const auto j : enum_iterator<PositionIndex>(inc(i), T))
            x[i, j] = static_cast<rlmm_float>(0.f);
        ENDFOR
    }

    /** dscores[T×T] += ∂L/∂scores  via the softmax Jacobian.
     * dp/dscores use stride T; p uses p_stride (the cached matrix's row stride).
     */
    void TransformerBlock::softmax_backward(
        // OFFLOAD_PARAMETERS(dp, p, dscores, T)
        const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dp,
        const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& p,
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dscores,
        PositionIndex T
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator<PositionIndex>(T), (dp, p, dscores))
        rlmm_float dot = 0.f;
        for (const auto j : enum_iterator<PositionIndex>(inc(i)))
            dot += dp[i, j] * p[i, j];
        for (const auto j : enum_iterator<PositionIndex>(inc(i)))
            dscores[i, j] += p[i, j] * (dp[i, j] - dot);
        ENDFOR
    }

    // ── SwiGLU backward ───────────────────────────────────────────────────────

    void TransformerBlock::swiglu_backward(
        // OFFLOAD_PARAMETERS(seq, gate_pre, up_pre, d_ffn_act, d_gate_pre, d_up_pre)
        PositionIndex seq,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_ffn_act,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_gate_pre,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_up_pre
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto gate_range = enum_iterator2D<PositionIndex, FFDimension>(seq);
        OFFLOAD_PARFOR_2D_PARAM(t, f, gate_range, (gate_pre, up_pre, d_ffn_act, d_gate_pre, d_up_pre))
        const rlmm_float g = gate_pre[t, f];
        const rlmm_float sg = 1.0f / (1.0f + std::exp(-g)); // sigma(g)
        const rlmm_float silu = g * sg;
        // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
        // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
        const rlmm_float dsilu_dg = sg * (1.0f + g * (1.0f - sg));
        d_gate_pre[t, f] = d_ffn_act[t, f] * up_pre[t, f] * dsilu_dg;
        d_up_pre[t, f] = d_ffn_act[t, f] * silu;
        ENDFOR
    }

    // ── SGD + momentum ─────────────────────────────────────────────────────────

    // ── forward workspace ─────────────────────────────────────────────────────
    // All large fixed-size matrices live here so they are heap-allocated via
    // unique_ptr and do not blow the stack (~21 MB of combined fixed-size arrays).
    struct ForwardWorkspace
    {
        PositionIndex seq_len;
        // Activations cached for the backward pass
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_in;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_norm_attn;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> Q, K, V;
        // Per-head softmax weight matrices; only the top-left [T × T] block is live.
        fixed_size_vector<flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>, HeadsIndex> attn_w;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> attn_concat;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_mid;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_norm_ff;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> gate_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> up_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> ffn_act;
        // Temporaries used only within forward() itself
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> attn_proj;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> ffn_out;

        explicit ForwardWorkspace(PositionIndex seq)
            : seq_len(seq)
            , h_in(seq)
            , h_norm_attn(seq)
            , Q(seq)
            , K(seq)
            , V(seq)
            , attn_concat(seq)
            , h_mid(seq)
            , h_norm_ff(seq)
            , gate_pre(seq)
            , up_pre(seq)
            , ffn_act(seq)
            , attn_proj(seq)
            , ffn_out(seq)
        {}
    };

    // ── destructor ────────────────────────────────────────────────────────────
    // Defined here so ForwardWorkspace is complete when unique_ptr deleter fires.
    TransformerBlock::TransformerBlock() = default;
    TransformerBlock::~TransformerBlock() = default;
    TransformerBlock::TransformerBlock(TransformerBlock&&) noexcept = default;
    TransformerBlock& TransformerBlock::operator=(TransformerBlock&&) noexcept = default;

    // ── forward pass ──────────────────────────────────────────────────────────

    void TransformerBlock::forward(
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h,
        PositionIndex seq_len
    )
    {
        constexpr int Dh = static_cast<int>(HeadDimension::MAX);

        if (!m_fwd_ws || m_fwd_ws->seq_len != seq_len)
            m_fwd_ws = std::make_unique<ForwardWorkspace>(seq_len);
        auto& ws = *m_fwd_ws;

        ws.h_in = h;

        // ── 1. Pre-norm (attention) ──────────────────────────────────────────
        rms_norm(h, ws.h_norm_attn);

        // ── 2. Q / K / V projections ─────────────────────────────────────────
        PARSECTIONS_BEGIN
        matmul_ABt(ws.h_norm_attn, W_q, ws.Q);
        PARSECTION
        matmul_ABt(ws.h_norm_attn, W_k, ws.K);
        PARSECTION
        matmul_ABt(ws.h_norm_attn, W_v, ws.V);
        PARSECTIONS_END

        // ── 3. Multi-head causal self-attention ──────────────────────────────
        const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        ws.attn_concat.fill(RLMM_ZERO);

        PARFOR(hi, enum_iterator<HeadsIndex>())
        auto& scores_mat = ws.attn_w[hi];
        scores_mat.set_size(seq_len, seq_len);

        const auto hi_int = static_cast<size_t>(hi);
        const auto hStart = static_cast<EmbeddingDimension>(hi_int * static_cast<size_t>(HeadDimension::MAX));
        const auto hEnd = static_cast<EmbeddingDimension>((hi_int + 1) * static_cast<size_t>(HeadDimension::MAX));

        // scores_mat[i, j] = Q[i, hi*Dh..] · K[j, hi*Dh..] * scale
        PARFOR_2D(i, j, enum_iterator2D<PositionIndex, PositionIndex>(seq_len, seq_len))
        float dot = 0.f;
        RLLM_OMP_SIMD_REDUCTION_PLUS(dot)
        for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
            dot += ws.Q[i, d] * ws.K[j, d];
        scores_mat[i, j] = dot * scale;
        ENDFOR

        causal_softmax(scores_mat, seq_len);

        // attn_concat[i, d] = sum_j scores_mat[i,j] * V[j, d]  (causal: j ≤ i)
        PARFOR_2D_TRIANGULAR(i, j, seq_len)
        const float w = scores_mat[i, j];
        for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
            ws.attn_concat[i, d] += w * ws.V[j, d];
        ENDFOR
        ENDFOR

        // ── 4. Output projection + residual ──────────────────────────────────
        matmul_ABt(ws.attn_concat, W_o, ws.attn_proj);

        PARFOR_2D(t, d, enum_iterator2D<PositionIndex, EmbeddingDimension>(seq_len))
        ws.h_mid[t, d] = h[t, d] + ws.attn_proj[t, d];
        ENDFOR

        // ── 5. Pre-norm (FFN) ─────────────────────────────────────────────────
        rms_norm(ws.h_mid, ws.h_norm_ff);

        // ── 6. SwiGLU FFN ─────────────────────────────────────────────────────
        PARSECTIONS_BEGIN
        matmul_ABt(ws.h_norm_ff, W_gate, ws.gate_pre);
        PARSECTION
        matmul_ABt(ws.h_norm_ff, W_up, ws.up_pre);
        PARSECTIONS_END

        PARFOR_2D(t, f, enum_iterator2D<PositionIndex, FFDimension>(seq_len))
        const float g = ws.gate_pre[t, f];
        const float silu = g / (1.0f + std::exp(-g));
        ws.ffn_act[t, f] = silu * ws.up_pre[t, f];
        ENDFOR

        matmul_ABt(ws.ffn_act, W_down, ws.ffn_out);

        // ── 7. Residual ───────────────────────────────────────────────────────
        PARFOR_2D(t, d, enum_iterator2D<PositionIndex, EmbeddingDimension>(seq_len))
        h[t, d] = ws.h_mid[t, d] + ws.ffn_out[t, d];
        ENDFOR
    }

    // ── backward temporaries ──────────────────────────────────────────────────
    // All large matrices live here so they are heap-allocated via unique_ptr and
    // do not blow the stack (~21 MB combined).
    struct BackwardWorkspace
    {
        // FFN backward
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_mid;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_ffn_act;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension> dW_down;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_gate_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_up_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_norm_ff;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> dW_gate;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> dW_up;
        // Attention backward
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_attn_concat;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_o;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_Q;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_K;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_V;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_q;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_k;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_v;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_norm_attn;
        // Scratch buffer shared by both matmul accumulation loops
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> tmp;
        // Per-head scratch: each head writes its own d_scores/d_raw so the hi loop is parallel.
        fixed_size_vector<flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>, HeadsIndex> d_scores;
        fixed_size_vector<flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>, HeadsIndex> d_raw;

        explicit BackwardWorkspace(PositionIndex seq)
            : d_h_mid(seq)
            , d_ffn_act(seq)
            , d_gate_pre(seq)
            , d_up_pre(seq)
            , d_h_norm_ff(seq)
            , d_attn_concat(seq)
            , d_Q(seq)
            , d_K(seq)
            , d_V(seq)
            , d_h_norm_attn(seq)
            , tmp(seq)
        {}

        // Zero only the fields that accumulate across the backward pass.
        // Fields fully overwritten before use (d_h_mid, d_ffn_act, d_gate_pre,
        // d_up_pre, d_attn_concat, tmp, d_scores, d_raw) are left untouched.
        void reset()
        {
            dW_down.fill(RLMM_ZERO);
            d_h_norm_ff.fill(RLMM_ZERO);
            dW_gate.fill(RLMM_ZERO);
            dW_up.fill(RLMM_ZERO);
            dW_o.fill(RLMM_ZERO);
            d_Q.fill(RLMM_ZERO);
            d_K.fill(RLMM_ZERO);
            d_V.fill(RLMM_ZERO);
            dW_q.fill(RLMM_ZERO);
            dW_k.fill(RLMM_ZERO);
            dW_v.fill(RLMM_ZERO);
            d_h_norm_attn.fill(RLMM_ZERO);
        }
    };

    // ── backward pass ─────────────────────────────────────────────────────────

    void TransformerBlock::backward(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dout,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& din,
        float learning_rate
    )
    {
        const PositionIndex seq = m_fwd_ws->seq_len;
        if (!m_bwd_ws || m_bwd_ws->d_h_mid.num_rows() != seq)
            m_bwd_ws = std::make_unique<BackwardWorkspace>(seq);
        m_bwd_ws->reset();
        auto* ws = m_bwd_ws.get();
        auto& fwd = *m_fwd_ws;

        // ── FFN backward ──────────────────────────────────────────────────────
        // h_out = h_mid + ffn_out  → d_h_mid += dout,  d_ffn_out = dout (same buffer)
        ws->d_h_mid = dout; // copy first to avoid overwriting dout before it's used in dW_down

        // d_ffn_act = d_ffn_out @ W_down   (W_down[D × Dff])
        matmul_AB(dout, W_down, ws->d_ffn_act);

        // dW_down[D, Dff] += dout^T @ ffn_act
        matmul_AtB_acc(dout, fwd.ffn_act, ws->dW_down);

        // SwiGLU backward: d(silu(g)*u) / dg, du
        swiglu_backward(fwd.seq_len, fwd.gate_pre, fwd.up_pre, ws->d_ffn_act, ws->d_gate_pre, ws->d_up_pre);

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        matmul_AB(ws->d_gate_pre, W_gate, ws->tmp);
        ws->d_h_norm_ff.element_wise_add(ws->tmp);
        matmul_AB(ws->d_up_pre, W_up, ws->tmp);
        ws->d_h_norm_ff.element_wise_add(ws->tmp);

        // weight gradients for gate, up
        PARSECTIONS_BEGIN
        matmul_AtB_acc(ws->d_gate_pre, fwd.h_norm_ff, ws->dW_gate);
        PARSECTION
        matmul_AtB_acc(ws->d_up_pre, fwd.h_norm_ff, ws->dW_up);
        PARSECTIONS_END

        // RMSNorm backward for FFN: d_h_mid += rms_bwd(d_h_norm_ff, h_mid)
        rms_norm_backward(ws->d_h_norm_ff, fwd.h_mid, ws->d_h_mid);

        // ── Attention backward ─────────────────────────────────────────────────
        // h_mid = h_in + attn_proj  → d_attn_proj = d_h_mid (passed through residual)
        //                             d_h_in (residual part) = d_h_mid (accumulated below)

        // d_attn_concat = d_attn_proj @ W_o
        matmul_AB(ws->d_h_mid, W_o, ws->d_attn_concat);
        matmul_AtB_acc(ws->d_h_mid, fwd.attn_concat, ws->dW_o);

        // Per-head backward
        const float scale = 1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX)));
        PARFOR(hi, enum_iterator<HeadsIndex>())
        // Each head owns its own d_scores_h / d_raw_h so parallel heads never conflict.
        auto& d_scores_h = ws->d_scores[hi];
        auto& d_raw_h = ws->d_raw[hi];
        d_scores_h.set_size(fwd.seq_len, fwd.seq_len);
        d_raw_h.set_size(fwd.seq_len, fwd.seq_len);
        d_raw_h.fill(RLMM_ZERO);

        const auto& scores_mat = fwd.attn_w[hi];
        const auto hi_int = static_cast<size_t>(hi);
        const auto hStart = static_cast<EmbeddingDimension>(hi_int * static_cast<size_t>(HeadDimension::MAX));
        const auto hEnd = static_cast<EmbeddingDimension>((hi_int + 1) * static_cast<size_t>(HeadDimension::MAX));

        // d_V_h[j, d] += sum_i scores_mat[i,j] * d_attn_concat[i, d]
        // Safe across heads: each head writes its own [hStart, hEnd) columns of d_V.
        PARFOR_2D_UPPER_TRIANGULAR(j, i, fwd.seq_len)
        const float w = scores_mat[i, j];
        for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
            ws->d_V[j, d] += w * ws->d_attn_concat[i, d];
        ENDFOR

        // d_scores_h[i,j] = d_attn_concat[i, d] · V[j, d]
        PARFOR_2D_TRIANGULAR(i, j, fwd.seq_len)
        float dot = 0.f;
        for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
            dot += ws->d_attn_concat[i, d] * fwd.V[j, d];
        d_scores_h[i, j] = dot;
        ENDFOR

        // Backward through causal softmax
        softmax_backward(d_scores_h, scores_mat, d_raw_h, fwd.seq_len);

        // d_Q and d_K — each head writes its own [hStart, hEnd) columns.
        PARFOR_2D_TRIANGULAR(i, j, fwd.seq_len)
        const float ds = d_raw_h[i, j] * scale;
        for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
            ws->d_Q[i, d] += ds * fwd.K[j, d];
        ENDFOR

        PARFOR_2D_UPPER_TRIANGULAR(j, i, fwd.seq_len)
        const float ds = d_raw_h[i, j] * scale;
        for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
            ws->d_K[j, d] += ds * fwd.Q[i, d];
        ENDFOR
        ENDFOR

        // Weight gradients for W_q, W_k, W_v
        PARSECTIONS_BEGIN
        matmul_AtB_acc(ws->d_Q, fwd.h_norm_attn, ws->dW_q);
        PARSECTION
        matmul_AtB_acc(ws->d_K, fwd.h_norm_attn, ws->dW_k);
        PARSECTION
        matmul_AtB_acc(ws->d_V, fwd.h_norm_attn, ws->dW_v);
        PARSECTIONS_END

        // d_h_norm_attn = d_Q @ W_q  +  d_K @ W_k  +  d_V @ W_v
        matmul_AB(ws->d_Q, W_q, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);
        matmul_AB(ws->d_K, W_k, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);
        matmul_AB(ws->d_V, W_v, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);

        // ── d_h_in + weight updates ───────────────────────────────────────────
        // d_h_in (residual + RMSNorm backward) and all seven weight updates are
        // fully independent; run them all concurrently.
        PARSECTIONS_BEGIN
        din = ws->d_h_mid;
        rms_norm_backward(ws->d_h_norm_attn, fwd.h_in, din);
        PARSECTION
        sgd_update(W_q, V_q, ws->dW_q, learning_rate);
        PARSECTION
        sgd_update(W_k, V_k, ws->dW_k, learning_rate);
        PARSECTION
        sgd_update(W_v, V_v, ws->dW_v, learning_rate);
        PARSECTION
        sgd_update(W_o, V_o, ws->dW_o, learning_rate);
        PARSECTION
        sgd_update(W_gate, V_gate, ws->dW_gate, learning_rate);
        PARSECTION
        sgd_update(W_up, V_up, ws->dW_up, learning_rate);
        PARSECTION
        sgd_update(W_down, V_down, ws->dW_down, learning_rate);
        PARSECTIONS_END
    }

    // ── serialisation ──────────────────────────────────────────────────────────

    nlohmann::json TransformerBlock::save() const
    {
        using namespace json_helpers;
        return {
            {"W_q", serialize_matrix(W_q)},
            {"W_k", serialize_matrix(W_k)},
            {"W_v", serialize_matrix(W_v)},
            {"W_o", serialize_matrix(W_o)},
            {"W_gate", serialize_matrix(W_gate)},
            {"W_up", serialize_matrix(W_up)},
            {"W_down", serialize_matrix(W_down)},
        };
    }

    void TransformerBlock::load(const nlohmann::json& j)
    {
        using namespace json_helpers;
        deserialize_matrix(j.at("W_q"), W_q);
        deserialize_matrix(j.at("W_k"), W_k);
        deserialize_matrix(j.at("W_v"), W_v);
        deserialize_matrix(j.at("W_o"), W_o);
        deserialize_matrix(j.at("W_gate"), W_gate);
        deserialize_matrix(j.at("W_up"), W_up);
        deserialize_matrix(j.at("W_down"), W_down);

        // Reset momentum on load — do not persist transient training state
        V_q.fill(RLMM_ZERO);
        V_k.fill(RLMM_ZERO);
        V_v.fill(RLMM_ZERO);
        V_o.fill(RLMM_ZERO);
        V_gate.fill(RLMM_ZERO);
        V_up.fill(RLMM_ZERO);
        V_down.fill(RLMM_ZERO);
    }

} // namespace rllm
