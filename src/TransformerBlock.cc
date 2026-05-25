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
        constexpr float sd = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(EmbeddingDimension::MAX)));
        constexpr float sf = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(FFDimension::MAX)));

        W_q.fill_rand(-sd, sd);
        W_k.fill_rand(-sd, sd);
        W_v.fill_rand(-sd, sd);
        W_o.fill_rand(-sd, sd);
        W_gate.fill_rand(-sd, sd);
        W_up.fill_rand(-sd, sd);
        W_down.fill_rand(-sf, sf);

        V_q.fill(0.f);
        V_k.fill(0.f);
        V_v.fill(0.f);
        V_o.fill(0.f);
        V_gate.fill(0.f);
        V_up.fill(0.f);
        V_down.fill(0.f);
    }

    // ── normalisation ──────────────────────────────────────────────────────────

    // y_t = x_t / rms(x_t)  for each row t
    void TransformerBlock::rms_norm(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& y
    )
    {
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        PARFOR(t, enum_iterator<PositionIndex>(x.num_rows()))
            float sq = 0.f;
            for (const auto i : enum_iterator<EmbeddingDimension>())
            {
                const auto val = x[t, i];
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
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dy,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dx
    )
    {
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        PARFOR(t, enum_iterator<PositionIndex>(x.num_rows()))
            float sq = 0.f;
            for (const auto i : enum_iterator<EmbeddingDimension>())
                sq += x[t, i] * x[t, i];
            const float inv = 1.0f / std::sqrt(sq / fd + eps);

            // dot = mean(dy · y) = (1/d) * sum_i dy[i] * x[i] * inv
            float dot = 0.f;
            for (const auto i : enum_iterator<EmbeddingDimension>())
                dot += dy[t, i] * x[t, i] * inv;
            dot /= fd;

            for (const auto i : enum_iterator<EmbeddingDimension>())
                dx[t, i] += inv * (dy[t, i] - x[t, i] * inv * dot);
        ENDFOR
    }

    // ── attention helpers ──────────────────────────────────────────────────────

    // In-place causal softmax over a [T × T] score matrix (row i only attends
    // to positions j ≤ i).  stride is the distance between rows in floats.
    void
    TransformerBlock::causal_softmax(flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& x, PositionIndex T)
    {
        PARFOR(i, enum_iterator<PositionIndex>(T))
            float max_val = x[i, PositionIndex::START];
            for (const auto j : enum_iterator<PositionIndex>(inc(PositionIndex::START), inc(i)))
                max_val = std::max(max_val, x[i, j]);
            float sum_exp = 0.f;
            // Compute and sum the exponentials for the active [T × T] block of the row; leave the rest of the row as-is
            // to avoid unnecessary writes.
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
            {
                x[i, j] = std::exp(x[i, j] - max_val);
                sum_exp += x[i, j];
            }
            const float inv = 1.0f / sum_exp;
            // scale the active [T × T] block of the row; leave the rest of the row as-is to avoid unnecessary writes
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                x[i, j] *= inv;

            // clear from the active [T × T] block to the end of the row
            // to avoid stale values affecting the backward pass
            for (const auto j : enum_iterator<PositionIndex>(inc(i), T))
                x[i, j] = 0.f;
        ENDFOR
    }

    // dscores[T×T] += ∂L/∂scores  via the softmax Jacobian.
    // dp/dscores use stride T; p uses p_stride (the cached matrix's row stride).
    void TransformerBlock::softmax_backward(
        const flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dp,
        const flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& p,
        flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dscores,
        PositionIndex T
    )
    {
        PARFOR(i, enum_iterator<PositionIndex>(T))
            float dot = 0.f;
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                dot += dp[i, j] * p[i, j];
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                dscores[i, j] += p[i, j] * (dp[i, j] - dot);
        ENDFOR
    }

    // ── SwiGLU backward ───────────────────────────────────────────────────────

    void TransformerBlock::swiglu_backward(
        PositionIndex seq,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& up_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& d_ffn_act,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& d_gate_pre,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& d_up_pre
    )
    {
        PARFOR_2D(t, f, enum_iterator2D<PositionIndex, FFDimension>(seq))
            const float g = gate_pre[t, f];
            const float sg = 1.0f / (1.0f + std::exp(-g)); // sigma(g)
            const float silu = g * sg;
            // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
            // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
            const float dsilu_dg = sg * (1.0f + g * (1.0f - sg));
            d_gate_pre[t, f] = d_ffn_act[t, f] * up_pre[t, f] * dsilu_dg;
            d_up_pre[t, f] = d_ffn_act[t, f] * silu;
        ENDFOR
    }

    // ── SGD + momentum ─────────────────────────────────────────────────────────

    // ── forward pass ──────────────────────────────────────────────────────────

    void
    TransformerBlock::forward(flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len)
    {
        constexpr int Dh = static_cast<int>(HeadDimension::MAX);

        m_seq_len = seq_len;
        m_h_in = h;

        // ── 1. Pre-norm (attention) ──────────────────────────────────────────
        m_h_norm_attn.set_rows(seq_len);
        rms_norm(h, m_h_norm_attn);

        // ── 2. Q / K / V projections ─────────────────────────────────────────
        m_Q.set_rows(seq_len);
        m_K.set_rows(seq_len);
        m_V.set_rows(seq_len);

        PARSECTIONS_BEGIN
            matmul_ABt(m_h_norm_attn, W_q, m_Q);
        PARSECTION
            matmul_ABt(m_h_norm_attn, W_k, m_K);
        PARSECTION
            matmul_ABt(m_h_norm_attn, W_v, m_V);
        PARSECTIONS_END

        // ── 3. Multi-head causal self-attention ──────────────────────────────
        constexpr float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        m_attn_concat.set_rows(seq_len);
        m_attn_concat.fill(0.f);

        PARFOR(hi, enum_iterator<HeadsIndex>())
            auto& scores_mat = m_attn_w[hi];
            scores_mat.set_size(seq_len, seq_len);

            const auto hi_int = static_cast<size_t>(hi);
            const auto hStart = static_cast<EmbeddingDimension>(hi_int * static_cast<size_t>(HeadDimension::MAX));
            const auto hEnd = static_cast<EmbeddingDimension>((hi_int + 1) * static_cast<size_t>(HeadDimension::MAX));

            // scores_mat[i, j] = Q[i, hi*Dh..] · K[j, hi*Dh..] * scale
            for (const auto i : enum_iterator<PositionIndex>(seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(seq_len))
                {
                    float dot = 0.f;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        dot += m_Q[i, d] * m_K[j, d];
                    scores_mat[i, j] = dot * scale;
                }
            }

            causal_softmax(scores_mat, seq_len);

            // attn_concat[i, d] = sum_j scores_mat[i,j] * V[j, d]  (causal: j ≤ i)
            for (const auto i : enum_iterator<PositionIndex>(seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                {
                    const float w = scores_mat[i, j];
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        m_attn_concat[i, d] += w * m_V[j, d];
                }
            }
        ENDFOR

        // ── 4. Output projection + residual ──────────────────────────────────
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> attn_proj;
        attn_proj.set_rows(seq_len);
        matmul_ABt(m_attn_concat, W_o, attn_proj);

        m_h_mid.set_rows(seq_len);

        PARFOR_2D(t, d, enum_iterator2D<PositionIndex, EmbeddingDimension>(seq_len))
            m_h_mid[t, d] = h[t, d] + attn_proj[t, d];
        ENDFOR

        // ── 5. Pre-norm (FFN) ─────────────────────────────────────────────────
        m_h_norm_ff.set_rows(seq_len);
        rms_norm(m_h_mid, m_h_norm_ff);

        // ── 6. SwiGLU FFN ─────────────────────────────────────────────────────
        m_gate_pre.set_rows(seq_len);
        m_up_pre.set_rows(seq_len);
        matmul_ABt(m_h_norm_ff, W_gate, m_gate_pre);
        matmul_ABt(m_h_norm_ff, W_up, m_up_pre);

        m_ffn_act.set_rows(seq_len);

        PARFOR_2D(t, f, enum_iterator2D<PositionIndex, FFDimension>(seq_len))
            const float g = m_gate_pre[t, f];
            const float silu = g / (1.0f + std::exp(-g));
            m_ffn_act[t, f] = silu * m_up_pre[t, f];
        ENDFOR

        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> ffn_out;
        ffn_out.set_rows(seq_len);
        matmul_ABt(m_ffn_act, W_down, ffn_out);

        // ── 7. Residual ───────────────────────────────────────────────────────
        for (const auto [t, d] : enum_iterator2D<PositionIndex, EmbeddingDimension>(seq_len))
            h[t, d] = m_h_mid[t, d] + ffn_out[t, d];
    }

    // ── backward temporaries ──────────────────────────────────────────────────
    // All large matrices live here so they are heap-allocated via unique_ptr and
    // do not blow the stack (~21 MB combined).
    struct BackwardWorkspace
    {
        // FFN backward
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_h_mid;
        flexible_rows_matrix<float, PositionIndex, FFDimension> d_ffn_act;
        fixed_size_matrix<float, EmbeddingDimension, FFDimension> dW_down;
        flexible_rows_matrix<float, PositionIndex, FFDimension> d_gate_pre;
        flexible_rows_matrix<float, PositionIndex, FFDimension> d_up_pre;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_h_norm_ff;
        fixed_size_matrix<float, FFDimension, EmbeddingDimension> dW_gate;
        fixed_size_matrix<float, FFDimension, EmbeddingDimension> dW_up;
        // Attention backward
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_attn_concat;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> dW_o;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_Q;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_K;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_V;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> dW_q;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> dW_k;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> dW_v;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> d_h_norm_attn;
        // Scratch buffer shared by both matmul accumulation loops
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> tmp;
        flexible_rows_cols_matrix<float, PositionIndex, PositionIndex> d_scores;
        flexible_rows_cols_matrix<float, PositionIndex, PositionIndex> d_raw;

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
            , d_scores(seq, seq)
            , d_raw(seq, seq)
        {}
    };

    // ── backward pass ─────────────────────────────────────────────────────────

    void TransformerBlock::backward(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dout,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& din,
        float learning_rate
    )
    {
        auto ws = std::make_unique<BackwardWorkspace>(m_seq_len);

        // ── FFN backward ──────────────────────────────────────────────────────
        // h_out = h_mid + ffn_out  → d_h_mid += dout,  d_ffn_out = dout (same buffer)
        ws->d_h_mid = dout; // copy first to avoid overwriting dout before it's used in dW_down

        // d_ffn_act = d_ffn_out @ W_down   (W_down[D × Dff])
        matmul_AB(dout, W_down, ws->d_ffn_act);

        // dW_down[D, Dff] += dout^T @ ffn_act
        matmul_AtB_acc(dout, m_ffn_act, ws->dW_down);

        // SwiGLU backward: d(silu(g)*u) / dg, du
        swiglu_backward(m_seq_len, m_gate_pre, m_up_pre, ws->d_ffn_act, ws->d_gate_pre, ws->d_up_pre);

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        matmul_AB(ws->d_gate_pre, W_gate, ws->tmp);
        ws->d_h_norm_ff.element_wise_add(ws->tmp);
        matmul_AB(ws->d_up_pre, W_up, ws->tmp);
        ws->d_h_norm_ff.element_wise_add(ws->tmp);

        // weight gradients for gate, up
        matmul_AtB_acc(ws->d_gate_pre, m_h_norm_ff, ws->dW_gate);
        matmul_AtB_acc(ws->d_up_pre, m_h_norm_ff, ws->dW_up);

        // RMSNorm backward for FFN: d_h_mid += rms_bwd(d_h_norm_ff, h_mid)
        rms_norm_backward(ws->d_h_norm_ff, m_h_mid, ws->d_h_mid);

        // ── Attention backward ─────────────────────────────────────────────────
        // h_mid = h_in + attn_proj  → d_attn_proj = d_h_mid (passed through residual)
        //                             d_h_in (residual part) = d_h_mid (accumulated below)

        // d_attn_concat = d_attn_proj @ W_o
        matmul_AB(ws->d_h_mid, W_o, ws->d_attn_concat);
        matmul_AtB_acc(ws->d_h_mid, m_attn_concat, ws->dW_o);

        // Per-head backward
        constexpr float scale = 1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX)));
        for (const auto hi : enum_iterator<HeadsIndex>())
        {
            ws->d_raw.fill(0.f); // d_raw accumulates per head; reset each iteration

            const auto& scores_mat = m_attn_w[hi];
            const auto hi_int = static_cast<size_t>(hi);
            const auto hStart = static_cast<EmbeddingDimension>(hi_int * static_cast<size_t>(HeadDimension::MAX));
            const auto hEnd = static_cast<EmbeddingDimension>((hi_int + 1) * static_cast<size_t>(HeadDimension::MAX));

            // d_V_h[j, d] += sum_i scores_mat[i,j] * d_attn_concat[i, d]
            for (const auto j : enum_iterator<PositionIndex>(m_seq_len))
            {
                for (const auto i : enum_iterator<PositionIndex>(j, m_seq_len)) // only non-masked rows
                {
                    const float w = scores_mat[i, j];
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws->d_V[j, d] += w * ws->d_attn_concat[i, d];
                }
            }

            // d_scores[i,j] = d_attn_concat[i, d] · V[j, d]
            for (const auto i : enum_iterator<PositionIndex>(m_seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                {
                    float dot = 0.f;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        dot += ws->d_attn_concat[i, d] * m_V[j, d];
                    ws->d_scores[i, j] = dot;
                }
            }

            // Backward through causal softmax
            softmax_backward(ws->d_scores, scores_mat, ws->d_raw, m_seq_len);

            // d_Q and d_K
            for (const auto i : enum_iterator<PositionIndex>(m_seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                {
                    const float ds = ws->d_raw[i, j] * scale;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws->d_Q[i, d] += ds * m_K[j, d];
                }
            }
            for (const auto j : enum_iterator<PositionIndex>(m_seq_len))
            {
                for (const auto i : enum_iterator<PositionIndex>(j, m_seq_len))
                {
                    const float ds = ws->d_raw[i, j] * scale;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws->d_K[j, d] += ds * m_Q[i, d];
                }
            }
        }

        // Weight gradients for W_q, W_k, W_v
        matmul_AtB_acc(ws->d_Q, m_h_norm_attn, ws->dW_q);
        matmul_AtB_acc(ws->d_K, m_h_norm_attn, ws->dW_k);
        matmul_AtB_acc(ws->d_V, m_h_norm_attn, ws->dW_v);

        // d_h_norm_attn = d_Q @ W_q  +  d_K @ W_k  +  d_V @ W_v
        matmul_AB(ws->d_Q, W_q, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);
        matmul_AB(ws->d_K, W_k, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);
        matmul_AB(ws->d_V, W_v, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);

        // d_h_in: residual from d_h_mid + RMSNorm backward
        din = ws->d_h_mid;
        rms_norm_backward(ws->d_h_norm_attn, m_h_in, din);

        // ── Weight updates ─────────────────────────────────────────────────────
        sgd_update(W_q, V_q, ws->dW_q, learning_rate);
        sgd_update(W_k, V_k, ws->dW_k, learning_rate);
        sgd_update(W_v, V_v, ws->dW_v, learning_rate);
        sgd_update(W_o, V_o, ws->dW_o, learning_rate);
        sgd_update(W_gate, V_gate, ws->dW_gate, learning_rate);
        sgd_update(W_up, V_up, ws->dW_up, learning_rate);
        sgd_update(W_down, V_down, ws->dW_down, learning_rate);
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
        V_q.fill(0.f);
        V_k.fill(0.f);
        V_v.fill(0.f);
        V_o.fill(0.f);
        V_gate.fill(0.f);
        V_up.fill(0.f);
        V_down.fill(0.f);
    }

} // namespace rllm
