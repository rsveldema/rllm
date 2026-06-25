#include <JsonTensorHelpers.hpp>
#include <RandomHelpers.hpp>
#include <TransformerBlock.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <enum_iterator3D.hpp>
#include <flexible_rows_cols_levels_matrix.hpp>
#include <parallel.hpp>
#include <vecmath.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>

namespace rllm
{
    static_assert(static_cast<int>(EmbeddingDimension::MAX) % static_cast<int>(HeadsIndex::MAX) == 0, "EmbeddingDimension::MAX must be divisible by HeadsIndex::MAX");

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
        copy_weights_to_offload_buffer();

        V_q.zero();
        V_k.zero();
        V_v.zero();
        V_o.zero();
        V_gate.zero();
        V_up.zero();
        V_down.zero();
    }

    void TransformerBlock::copy_weights_to_offload_buffer()
    {
        W_q.copy_to_offload_buffer();
        W_k.copy_to_offload_buffer();
        W_v.copy_to_offload_buffer();
        W_o.copy_to_offload_buffer();
        W_gate.copy_to_offload_buffer();
        W_up.copy_to_offload_buffer();
        W_down.copy_to_offload_buffer();
    }

    // ── normalisation ──────────────────────────────────────────────────────────


    void TransformerBlock::rms_norm(
        // OFFLOAD_PARAMETERS(x, y)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& y
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(t, enum_iterator1D<PositionIndex>(x.num_rows()), (x, y))
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        float sq = 0.f;
        for (const auto i : enum_iterator1D<EmbeddingDimension>())
        {
            const float val = x[t, i];
            sq += (val * val);
        }
        const float inv = (1.0f / std::sqrt(((sq / fd) + eps)));
        for (const auto i : enum_iterator1D<EmbeddingDimension>())
            y[t, i] = (x[t, i] * inv);
        ENDFOR
    }



    // dx += dL/dx  given dy = dL/dy and the original x (not the normalised y).
    // Per row:  dx_j += (1/rms) * (dy_j  -  y_j * mean(dy · y))
    void TransformerBlock::rms_norm_backward(
        // OFFLOAD_PARAMETERS(dy, x, dx)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dy,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dx
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(t, enum_iterator1D<PositionIndex>(x.num_rows()), (dy, x, dx))
        constexpr float eps = 1e-6f;

        float sq = 0.f;
        for (const auto i : enum_iterator1D<EmbeddingDimension>()) {
            const float value = x[t, i];
            sq += (value * value);
        }

        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);            
        const float inv = (1.0f / std::sqrt(((sq / fd) + eps)));

        // dot = mean(dy · y) = (1/d) * sum_i dy[i] * x[i] * inv
        float dot = 0.f;
        for (const auto i : enum_iterator1D<EmbeddingDimension>()) {
            dot += ((dy[t, i] * x[t, i]) * inv);
        }

        dot /= fd;

        for (const auto i : enum_iterator1D<EmbeddingDimension>()) {
            dx[t, i] += (inv * (dy[t, i] - ((x[t, i] * inv) * dot)));
        }
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
        flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& x,
        PositionIndex T
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<PositionIndex>(T), (x, T))
        float max_val = x[i, PositionIndex::START];
        for (const auto j : enum_iterator1D<PositionIndex>(inc(PositionIndex::START), inc(i)))
            max_val = math::max(max_val, x[i, j]);
        float sum_exp = 0.f;
        // Compute and sum the exponentials for the active [T × T] block of the row; leave the rest of the row as-is
        // to avoid unnecessary writes.
        for (const auto j : enum_iterator1D<PositionIndex>(inc(i)))
        {
            x[i, j] = static_cast<float>(std::exp((x[i, j] - max_val)));
            sum_exp += x[i, j];
        }
        const float inv = (1.0f / sum_exp);
        // scale the active [T × T] block of the row; leave the rest of the row as-is to avoid unnecessary writes
        for (const auto j : enum_iterator1D<PositionIndex>(inc(i)))
            x[i, j] *= static_cast<float>(inv);

        // clear from the active [T × T] block to the end of the row
        // to avoid stale values affecting the backward pass
        for (const auto j : enum_iterator1D<PositionIndex>(inc(i), T))
            x[i, j] = static_cast<float>(0.f);
        ENDFOR
    }

    // ── SwiGLU backward ───────────────────────────────────────────────────────

    void TransformerBlock::swiglu_backward(
        // OFFLOAD_PARAMETERS(seq, gate_pre, up_pre, d_ffn_act, d_gate_pre, d_up_pre)
        PositionIndex seq,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& up_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& d_ffn_act,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& d_gate_pre,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& d_up_pre
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto gate_range = enum_iterator2D<PositionIndex, FFDimension>(seq);
        OFFLOAD_PARFOR_2D_PARAM(t, f, gate_range, (gate_pre, up_pre, d_ffn_act, d_gate_pre, d_up_pre))
        const float g = gate_pre[t, f];
        const float sg = (1.0f / (1.0f + std::exp(-g))); // sigma(g)
        const float silu = (g * sg);
        // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
        // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
        const float dsilu_dg = (sg * (1.0f + (g * (1.0f - sg))));
        d_gate_pre[t, f] = ((d_ffn_act[t, f] * up_pre[t, f]) * dsilu_dg);
        d_up_pre[t, f] = (d_ffn_act[t, f] * silu);
        ENDFOR
    }

    // ── SGD + momentum ─────────────────────────────────────────────────────────

    // ── destructor ────────────────────────────────────────────────────────────
    // Defined here so ForwardWorkspace is complete when unique_ptr deleter fires.
    TransformerBlock::TransformerBlock() = default;
    TransformerBlock::~TransformerBlock() = default;
    TransformerBlock::TransformerBlock(TransformerBlock&&) noexcept = default;
    TransformerBlock& TransformerBlock::operator=(TransformerBlock&&) noexcept = default;

    // ── forward pass ──────────────────────────────────────────────────────────

    void TransformerBlock::compute_attention_scores_for_head_hi(ForwardWorkspace& ws, PositionIndex seq_len, HeadsIndex hi)
    {
        // OFFLOAD_PARAMETERS(attn_w_h, Q, K, active_seq_len)
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h = ws.attn_w[hi];
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& Q = ws.Q;
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& K = ws.K;
        const PositionIndex active_seq_len = seq_len;
        // END_OFFLOAD_PARAMETERS

        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, active_seq_len, (attn_w_h, Q, K, active_seq_len))
        {
            float dot = 0.f;
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
            {
                dot += (Q[i, d] * K[j, d]);
            }
            const float score = (dot * (1.0f / sqrt(float(HeadDimension::MAX))));
            attn_w_h[i, j] = score;
        }
        ENDFOR
    }


    void TransformerBlock::compute_attention_scores_for_heads(ForwardWorkspace& ws, PositionIndex seq_len)
    {
        ws.attn_w.set_size(HeadsIndex::MAX);

        PARFOR_1D(hi, enum_iterator1D<HeadsIndex>())
            compute_attention_scores_for_head_hi(ws, seq_len, hi);
        ENDFOR
    }

    void TransformerBlock::apply_causal_softmax_for_heads(ForwardWorkspace& ws, PositionIndex seq_len)
    {
        PARFOR_1D(hi, enum_iterator1D<HeadsIndex>())
            apply_causal_softmax_for_head_hi(ws, seq_len, hi);
        ENDFOR
    }

    void TransformerBlock::apply_causal_softmax_for_head_hi(ForwardWorkspace& ws, PositionIndex seq_len, HeadsIndex hi)
    {
        const auto softmax_grid = enum_iterator1D<PositionIndex>(seq_len);

        // OFFLOAD_PARAMETERS(attn_w_h)
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h = ws.attn_w[hi];
        // END_OFFLOAD_PARAMETERS        

        OFFLOAD_PARFOR_1D_PARAM(i, softmax_grid, (attn_w_h))
        {
            float max_val = attn_w_h[i, 0];
            for (int j = 1; j <= static_cast<int>(i); ++j)
                max_val = math::max(max_val, attn_w_h[i, j]);

            float sum_exp = 0.f;
            for (int j = 0; j <= static_cast<int>(i); ++j)
            {
                attn_w_h[i, j] = static_cast<float>(std::exp((attn_w_h[i, j] - max_val)));
                sum_exp += attn_w_h[i, j];
            }

            const float inv = (1.0f / sum_exp);
            for (int j = 0; j <= static_cast<int>(i); ++j)
                attn_w_h[i, j] *= static_cast<float>(inv);
        }
        ENDFOR
    }

    inline void compute_attention_values_for_head(
        // OFFLOAD_PARAMETERS(attn_w_h, V, attn_concat, active_seq_len, hi)
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& V,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& attn_concat,
        int active_seq_len,
        HeadsIndex hi
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, static_cast<PositionIndex>(active_seq_len), (attn_concat, attn_w_h, V, active_seq_len, hi))
        {
            const float w = attn_w_h[i, j];
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
            {
                attn_concat[i, d] += (w * V[j, d]);
            }
        }
        ENDFOR
    }

    void TransformerBlock::compute_attention_values_for_heads(ForwardWorkspace& ws, PositionIndex seq_len)
    {
        auto& attn_w = ws.attn_w;
        const auto& V = ws.V;
        auto& attn_concat = ws.attn_concat;
        const int active_seq_len = static_cast<int>(seq_len);

        // attn_concat[i, d] = sum_j attn_w[hi,i,j] * V[j, d]  (causal: j <= i)
        PARFOR_1D(hi, enum_iterator1D<HeadsIndex>(HeadsIndex::MAX))
            compute_attention_values_for_head(attn_w[hi], V, attn_concat, active_seq_len, hi);
        ENDFOR
    }

    void TransformerBlock::forward_attention_heads(ForwardWorkspace& ws, PositionIndex seq_len)
    {
        ws.attn_w.set_size(HeadsIndex::MAX);

        compute_attention_scores_for_heads(ws, seq_len);
        apply_causal_softmax_for_heads(ws, seq_len);
        compute_attention_values_for_heads(ws, seq_len);
    }

    void TransformerBlock::forward(flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len, 
        ForwardWorkspace& workspace)
    {
        auto& ws = workspace;

        ws.h_in = h;

        // ── 1. Pre-norm (attention) ──────────────────────────────────────────
        //rms_norm_vulkan_optimized_with_workspace(h, ws.h_norm_attn, ws.rms_norm_row_sums, ws.rms_norm_inv_rms);
        rms_norm(h, ws.h_norm_attn);

        // ── 2. Q / K / V projections ─────────────────────────────────────────
        matmul_ABt_3_matrix_muls(ws.h_norm_attn, W_q, ws.Q, W_k, ws.K, W_v, ws.V);

        // ── 3. Multi-head causal self-attention ──────────────────────────────
        ws.attn_concat.zero();
        forward_attention_heads(ws, seq_len);

        // ── 4. Output projection + residual ──────────────────────────────────
        matmul_ABt(ws.attn_concat, W_o, ws.attn_proj);

        element_wise_sum(h, ws.attn_proj, ws.h_mid);

        // ── 5. Pre-norm (FFN) ─────────────────────────────────────────────────
        //rms_norm_vulkan_optimized_with_workspace(ws.h_mid, ws.h_norm_ff, ws.rms_norm_row_sums, ws.rms_norm_inv_rms);
        rms_norm(ws.h_mid, ws.h_norm_ff);

        // ── 6. SwiGLU FFN ─────────────────────────────────────────────────────
        matmul_ABt_2_matrix_muls(ws.h_norm_ff, W_gate, ws.gate_pre, W_up, ws.up_pre);

        swiglu_forward(ws.gate_pre, ws.up_pre, ws.ffn_act, seq_len);

        matmul_ABt(ws.ffn_act, W_down, ws.ffn_out);

        // ── 7. Residual ───────────────────────────────────────────────────────
        element_wise_sum(ws.h_mid, ws.ffn_out, h);
    }

    // ── backward pass ─────────────────────────────────────────────────────────

    inline void accumulate_attention_dv_for_head(
        // OFFLOAD_PARAMETERS(d_V, attn_w_h, d_attn_concat, seq_len, hStart)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_V,
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_attn_concat,
        PositionIndex seq_len,
        int hStart
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, seq_len, (d_V, attn_w_h, d_attn_concat, seq_len, hStart))
        {
            const float w = attn_w_h[i, j];
            for (const auto d_head : enum_iterator1D<HeadDimension>())
            {
                const int d = (hStart + int(d_head));
                d_V[j, d] += (w * d_attn_concat[i, d]);
            }
        }
        ENDFOR
    }

    void TransformerBlock::backward_accumulate_attention_dv_for_heads(BackwardWorkspace& ws, const ForwardWorkspace& fwd)
    {
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_V = ws.d_V;
        const fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& attn_w = fwd.attn_w;
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_attn_concat = ws.d_attn_concat;
        const PositionIndex seq_len = fwd.seq_len;

        // d_V_h[j, d] += sum_i attn_w[hi,i,j] * d_attn_concat[i, d]  (causal: j <= i)
        for (const auto hi : enum_iterator1D<HeadsIndex>(HeadsIndex::MAX))
        {
            const int hStart = (static_cast<int>(hi) * static_cast<int>(HeadDimension::MAX));
            accumulate_attention_dv_for_head(d_V, attn_w[hi], d_attn_concat, seq_len, hStart);
        }
    }

    inline void compute_attention_dscores_for_head(
        // OFFLOAD_PARAMETERS(d_scores_h, d_attn_concat, V, seq_len, hi)
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& d_scores_h,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_attn_concat,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& V,
        PositionIndex seq_len,
        HeadsIndex hi
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, seq_len, (d_scores_h, d_attn_concat, V, seq_len, hi))
        {
            float dot = 0.f;
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
            {
                dot += (d_attn_concat[i, d] * V[j, d]);
            }
            d_scores_h[i, j] = dot;
        }
        ENDFOR
    }

    void TransformerBlock::backward_compute_attention_dscores_for_heads(BackwardWorkspace& ws, const ForwardWorkspace& fwd)
    {
        fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& d_scores = ws.d_scores;
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_attn_concat = ws.d_attn_concat;
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& V = fwd.V;
        const PositionIndex seq_len = fwd.seq_len;

        // d_scores_h[i,j] = d_attn_concat[i, d] · V[j, d]
        // Safe for offload because each triangular cell writes a unique output element.
        for (const auto hi : enum_iterator1D<HeadsIndex>(HeadsIndex::MAX))
        {
            compute_attention_dscores_for_head(d_scores[hi], d_attn_concat, V, seq_len, hi);
        }
    }

    void TransformerBlock::softmax_attention_for_head(
        // OFFLOAD_PARAMETERS(d_scores_h, d_raw_h, attn_w_h, seq_len)
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& d_scores_h,
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& d_raw_h,
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h,
        PositionIndex seq_len
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, seq_len, (d_scores_h, d_raw_h, attn_w_h, seq_len))
        {
            float row_dot = 0.f;
            for (const auto k : enum_iterator1D<PositionIndex>(inc(i)))
            {
                row_dot += (d_scores_h[i, k] * attn_w_h[i, k]);
            }
            d_raw_h[i, j] = (attn_w_h[i, j] * (d_scores_h[i, j] - row_dot));
        }
        ENDFOR
    }

    void TransformerBlock::backward_softmax_attention_for_heads(BackwardWorkspace& ws, const ForwardWorkspace& fwd)
    {
        fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& d_scores = ws.d_scores;
        fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& d_raw = ws.d_raw;
        const PositionIndex seq_len = fwd.seq_len;
        const fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& attn_w = fwd.attn_w;

        for (const auto hi : enum_iterator1D<HeadsIndex>(HeadsIndex::MAX))
        {
            softmax_attention_for_head(d_scores[hi], d_raw[hi], attn_w[hi], seq_len);
        }
    }

    void TransformerBlock::backward_accumulate_attention_dq_for_heads(BackwardWorkspace& ws, const ForwardWorkspace& fwd)
    {
        PARFOR_1D(hi, enum_iterator1D<HeadsIndex>())
            backward_accumulate_attention_dq_for_head_hi(ws, fwd, hi);        
        ENDFOR
    }

    void TransformerBlock::backward_accumulate_attention_dq_for_head_hi(BackwardWorkspace& ws, 
        const ForwardWorkspace& fwd,
        HeadsIndex hi)
    {
        // OFFLOAD_PARAMETERS(d_Q, d_raw_h, K, seq_len, hStart)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_Q = ws.d_Q;
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& d_raw_h = ws.d_raw[hi];
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& K = fwd.K;
        const PositionIndex seq_len = fwd.seq_len;
        const int hStart = (static_cast<int>(hi) * static_cast<int>(HeadDimension::MAX));
        // END_OFFLOAD_PARAMETERS

        // d_Q and d_K are triangular reductions over j/i respectively.
        // Offload per output cell to avoid cross-thread accumulation races.
        const auto dq_grid = enum_iterator2D<PositionIndex, HeadDimension>(seq_len);

        OFFLOAD_PARFOR_2D_PARAM(i, d_head, dq_grid, (d_Q, d_raw_h, K, seq_len, hStart))
        {
            const int d = (hStart + int(d_head));
            const float head_scale = (1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX))));
            float sum_q = 0.f;
            for (const auto j : enum_iterator1D<PositionIndex>(inc(i)))
                sum_q += ((d_raw_h[i, j] * head_scale) * K[j, d]);
            d_Q[i, d] = sum_q;
        }
        ENDFOR
    }

    void TransformerBlock::backward_accumulate_attention_dk_for_heads_hi(BackwardWorkspace& ws, const ForwardWorkspace& fwd, HeadsIndex hi)
    {
        // OFFLOAD_PARAMETERS(d_K, d_raw_h, Q, seq_len, hStart)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_K = ws.d_K;
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& d_raw_h = ws.d_raw[hi];
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& Q = fwd.Q;
        PositionIndex seq_len = fwd.seq_len;
        const int hStart = (static_cast<int>(hi) * static_cast<int>(HeadDimension::MAX));
        // END_OFFLOAD_PARAMETERS
        

        const auto dk_grid = enum_iterator2D<PositionIndex, HeadDimension>(seq_len);
        OFFLOAD_PARFOR_2D_PARAM(j, d_head, dk_grid, (d_K, d_raw_h, Q, seq_len, hStart))
        {
            const int d = (hStart + int(d_head));
            float sum_k = 0.f;
            for (const auto i : enum_iterator1D<PositionIndex>(j, seq_len))
                sum_k += ((d_raw_h[i, j] * (1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX))))) * Q[i, d]);
            d_K[j, d] = sum_k;
        }
        ENDFOR
    }


    void TransformerBlock::backward_accumulate_attention_dk_for_heads(BackwardWorkspace& ws, const ForwardWorkspace& fwd)
    {
        PARFOR_1D(hi, enum_iterator1D<HeadsIndex>())
        backward_accumulate_attention_dk_for_heads_hi(ws, fwd, hi);
        ENDFOR
    }

    void TransformerBlock::backward_attention_heads(BackwardWorkspace& ws, const ForwardWorkspace& fwd)
    {
        backward_accumulate_attention_dv_for_heads(ws, fwd);
        backward_compute_attention_dscores_for_heads(ws, fwd);
        backward_softmax_attention_for_heads(ws, fwd);
        backward_accumulate_attention_dq_for_heads(ws, fwd);
        backward_accumulate_attention_dk_for_heads(ws, fwd);
    }

    void TransformerBlock::backward(const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dout, flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& din, BackwardWorkspace& workspace, float learning_rate,
        ForwardWorkspace& fwd_workspace)
    {
        const PositionIndex seq = fwd_workspace.seq_len;
        workspace.reset(seq);
        auto* ws = &workspace;
        auto& fwd = fwd_workspace;

        // ── FFN backward ──────────────────────────────────────────────────────
        // h_out = h_mid + ffn_out  → d_h_mid += dout,  d_ffn_out = dout (same buffer)
        ws->d_h_mid = dout; // copy first to avoid overwriting dout before it's used in dW_down

        // d_ffn_act = d_ffn_out @ W_down   (W_down[D × Dff])
        matmul_AB(dout, W_down, ws->d_ffn_act);

        // dW_down[D, Dff] += dout^T @ ffn_act
        matmul_AtB_acc(dout, fwd.ffn_act, ws->dW_down, fwd.seq_len);

        // SwiGLU backward: d(silu(g)*u) / dg, du
        swiglu_backward(fwd.seq_len, fwd.gate_pre, fwd.up_pre, ws->d_ffn_act, ws->d_gate_pre, ws->d_up_pre);

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        matmul_AB_add(ws->d_gate_pre, W_gate, ws->d_h_norm_ff);
        matmul_AB_add(ws->d_up_pre, W_up, ws->d_h_norm_ff);

        // weight gradients for gate, up
        matmul_AtB_acc_2_matrix(ws->d_gate_pre, ws->d_up_pre, fwd.h_norm_ff, ws->dW_gate, ws->dW_up, fwd.seq_len);

        // RMSNorm backward for FFN: d_h_mid += rms_bwd(d_h_norm_ff, h_mid)
        rms_norm_backward(ws->d_h_norm_ff, fwd.h_mid, ws->d_h_mid);

        // ── Attention backward ─────────────────────────────────────────────────
        // h_mid = h_in + attn_proj  → d_attn_proj = d_h_mid (passed through residual)
        //                             d_h_in (residual part) = d_h_mid (accumulated below)

        // d_attn_concat = d_attn_proj @ W_o
        matmul_AB(ws->d_h_mid, W_o, ws->d_attn_concat);
        matmul_AtB_acc(ws->d_h_mid, fwd.attn_concat, ws->dW_o, fwd.seq_len);

        // Per-head backward
        backward_attention_heads(*ws, fwd);

        // Weight gradients for W_q, W_k, W_v
        matmul_AtB_acc_3_matrix(ws->d_Q, ws->d_K, ws->d_V, fwd.h_norm_attn, ws->dW_q, ws->dW_k, ws->dW_v, fwd.seq_len);

        // d_h_norm_attn = d_Q @ W_q  +  d_K @ W_k  +  d_V @ W_v
        matmul_AB_add_3_matrix_muls(ws->d_Q, W_q, ws->d_K, W_k, ws->d_V, W_v, ws->d_h_norm_attn);

        // ── d_h_in + weight updates ───────────────────────────────────────────
        // d_h_in (residual + RMSNorm backward) and all seven weight updates are
        // fully independent; run them all concurrently.
        PARSECTIONS_BEGIN
        din = ws->d_h_mid;
        rms_norm_backward(ws->d_h_norm_attn, fwd.h_in, din);
        PARSECTION
        sgd_update_Wqkvo_x_Vqkvo_dWqkvo__4_matrix(W_q, V_q, ws->dW_q, W_k, V_k, ws->dW_k, W_v, V_v, ws->dW_v, W_o, V_o, ws->dW_o, learning_rate);
        PARSECTION
        sgd_update_Wgateup_x_Vgateup_dWgateup__2_matrix(W_gate, V_gate, ws->dW_gate, W_up, V_up, ws->dW_up, learning_rate);
        PARSECTION
        sgd_update_Wdown_x_Vdown_dWdown(W_down, V_down, ws->dW_down, learning_rate);
        PARSECTIONS_END
    }

    // ── serialisation ──────────────────────────────────────────────────────────

     std::unique_ptr<nlohmann::json> TransformerBlock::save() const
    {
        using namespace json_helpers;
        cpu_fixed_matrix<float16, EmbeddingDimension, EmbeddingDimension> cpu_Wq, cpu_Wk, cpu_Wv, cpu_Wo;
        cpu_fixed_matrix<float16, FFDimension, EmbeddingDimension> cpu_Wgate, cpu_Wup;
        cpu_fixed_matrix<float16, EmbeddingDimension, FFDimension> cpu_Wdown;
        W_q.copy_to_cpu(cpu_Wq);
        W_k.copy_to_cpu(cpu_Wk);
        W_v.copy_to_cpu(cpu_Wv);
        W_o.copy_to_cpu(cpu_Wo);
        W_gate.copy_to_cpu(cpu_Wgate);
        W_up.copy_to_cpu(cpu_Wup);
        W_down.copy_to_cpu(cpu_Wdown);
        return std::make_unique<nlohmann::json>(nlohmann::json{
            {"W_q", *serialize_matrix(cpu_Wq)},
            {"W_k", *serialize_matrix(cpu_Wk)},
            {"W_v", *serialize_matrix(cpu_Wv)},
            {"W_o", *serialize_matrix(cpu_Wo)},
            {"W_gate", *serialize_matrix(cpu_Wgate)},
            {"W_up", *serialize_matrix(cpu_Wup)},
            {"W_down", *serialize_matrix(cpu_Wdown)}
        });
    }

    void TransformerBlock::load(const nlohmann::json& j)
    {
        using namespace json_helpers;
        cpu_fixed_matrix<float16, EmbeddingDimension, EmbeddingDimension> cpu_Wq, cpu_Wk, cpu_Wv, cpu_Wo;
        cpu_fixed_matrix<float16, FFDimension, EmbeddingDimension> cpu_Wgate, cpu_Wup;
        cpu_fixed_matrix<float16, EmbeddingDimension, FFDimension> cpu_Wdown;
        deserialize_matrix(j.at("W_q"), cpu_Wq);
        deserialize_matrix(j.at("W_k"), cpu_Wk);
        deserialize_matrix(j.at("W_v"), cpu_Wv);
        deserialize_matrix(j.at("W_o"), cpu_Wo);
        deserialize_matrix(j.at("W_gate"), cpu_Wgate);
        deserialize_matrix(j.at("W_up"), cpu_Wup);
        deserialize_matrix(j.at("W_down"), cpu_Wdown);
        W_q.copy_from_cpu(cpu_Wq);
        W_k.copy_from_cpu(cpu_Wk);
        W_v.copy_from_cpu(cpu_Wv);
        W_o.copy_from_cpu(cpu_Wo);
        W_gate.copy_from_cpu(cpu_Wgate);
        W_up.copy_from_cpu(cpu_Wup);
        W_down.copy_from_cpu(cpu_Wdown);
        copy_weights_to_offload_buffer();

        // Reset momentum on load — do not persist transient training state
        V_q.zero();
        V_k.zero();
        V_v.zero();
        V_o.zero();
        V_gate.zero();
        V_up.zero();
        V_down.zero();
    }

} // namespace rllm
