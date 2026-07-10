#include <JsonTensorHelpers.hpp>
#include <RandomHelpers.hpp>
#include <RuntimeConfig.hpp>
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
#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>

namespace rllm
{
    static void add_dmodel_dmodel_gradient(
        // OFFLOAD_PARAMETERS(src, dst)
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& src,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (src, dst))
        const float old_value = dst[r, c];
        const float grad = src[r, c];
        const float new_value = (old_value + grad);
        dst[r, c] = new_value;
        ENDFOR
    }

    static void add_ff_dmodel_gradient(
        // OFFLOAD_PARAMETERS(src, dst)
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& src,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (src, dst))
        const float old_value = dst[r, c];
        const float grad = src[r, c];
        const float new_value = (old_value + grad);
        dst[r, c] = new_value;
        ENDFOR
    }

    static void add_dmodel_ff_gradient(
        // OFFLOAD_PARAMETERS(src, dst)
        const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& src,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (src, dst))
        const float old_value = dst[r, c];
        const float grad = src[r, c];
        const float new_value = (old_value + grad);
        dst[r, c] = new_value;
        ENDFOR
    }

    void TransformerGradientAccumulator::reset(VulkanQueue& queue)
    {
        dW_down.zero(queue);
        dW_gate.zero(queue);
        dW_up.zero(queue);
        dW_o.zero(queue);
        dW_q.zero(queue);
        dW_k.zero(queue);
        dW_v.zero(queue);
        touched = false;
    }

    static_assert(static_cast<int>(EmbeddingDimension::MAX) % static_cast<int>(HeadsIndex::MAX) == 0, "EmbeddingDimension::MAX must be divisible by HeadsIndex::MAX");

    static fixed_size_vector<int, PositionIndex> make_nan_scan_flag(VulkanQueue& queue)
    {
        cpu_fixed_vector<int, PositionIndex> cpu_flag;
        cpu_flag.set_size(PositionIndex::MAX);
        cpu_flag.zero();

        fixed_size_vector<int, PositionIndex> flag;
        flag.copy_from_cpu(queue, cpu_flag);
        return flag;
    }

    static bool nan_scan_failed(VulkanQueue& queue, fixed_size_vector<int, PositionIndex>& flag)
    {
        cpu_fixed_vector<int, PositionIndex> cpu_flag;
        flag.copy_to_cpu(queue, cpu_flag);
        return cpu_flag[PositionIndex::START] != 0;
    }

    static void scan_embedding_embedding_weight_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_embedding_embedding_velocity_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_ff_embedding_weight_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_ff_embedding_velocity_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_embedding_ff_weight_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_embedding_ff_velocity_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    void TransformerBlock::check_nan_finding_mode(const char* phase)
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_nan_scan_flag(queue);

        scan_embedding_embedding_weight_matrix(W_q, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_embedding_embedding_weight_matrix(W_k, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_embedding_embedding_weight_matrix(W_v, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_embedding_embedding_weight_matrix(W_o, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_embedding_embedding_velocity_matrix(V_q, flag, -VEL_CLIP, VEL_CLIP);
        scan_embedding_embedding_velocity_matrix(V_k, flag, -VEL_CLIP, VEL_CLIP);
        scan_embedding_embedding_velocity_matrix(V_v, flag, -VEL_CLIP, VEL_CLIP);
        scan_embedding_embedding_velocity_matrix(V_o, flag, -VEL_CLIP, VEL_CLIP);

        scan_ff_embedding_weight_matrix(W_gate, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_ff_embedding_weight_matrix(W_up, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_ff_embedding_velocity_matrix(V_gate, flag, -VEL_CLIP, VEL_CLIP);
        scan_ff_embedding_velocity_matrix(V_up, flag, -VEL_CLIP, VEL_CLIP);

        scan_embedding_ff_weight_matrix(W_down, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_embedding_ff_velocity_matrix(V_down, flag, -VEL_CLIP, VEL_CLIP);

        if (nan_scan_failed(queue, flag))
        {
            std::fprintf(stderr, "NAN_FINDING_MODE: TransformerBlock weights/velocities invalid during %s\n", phase);
            std::abort();
        }
    }

    static void scan_transformer_hidden_matrix(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(matrix, flag, rows, lower_bound, upper_bound)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        PositionIndex rows,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, rows, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_transformer_ff_matrix(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(matrix, flag, rows, lower_bound, upper_bound)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        PositionIndex rows,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, rows, lower_bound, upper_bound))
        const float value = matrix[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void check_transformer_hidden_nan_finding_mode(
        VulkanQueue& queue,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& matrix,
        PositionIndex rows,
        const char* name,
        const char* phase
    )
    {
        if (!nan_finding_mode_enabled())
            return;

        static constexpr float bound = 10000.0f;
        auto flag = make_nan_scan_flag(queue);
        scan_transformer_hidden_matrix(queue, matrix, flag, rows, -bound, bound);
        if (nan_scan_failed(queue, flag))
        {
            cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> cpu_matrix;
            cpu_matrix.set_rows(rows);
            matrix.copy_to_cpu(queue, cpu_matrix);

            float offending_value = std::nanf("");
            int offending_row = -1, offending_col = -1;
            for (const auto r : enum_iterator1D<PositionIndex>(rows))
            {
                for (const auto c : enum_iterator1D<EmbeddingDimension>())
                {
                    const float v = cpu_matrix[r, c];
                    if (v != v || v < -bound || v > bound)
                    {
                        offending_value = v;
                        offending_row = static_cast<int>(static_cast<size_t>(r));
                        offending_col = static_cast<int>(static_cast<size_t>(c));
                        break;
                    }
                }
                if (offending_row >= 0)
                    break;
            }

            std::fprintf(
                stderr,
                "NAN_FINDING_MODE: TransformerBlock %s invalid during %s; expected finite values in [%g, %g], "
                "actual value=%g at row=%d col=%d\n",
                name,
                phase,
                static_cast<double>(-bound),
                static_cast<double>(bound),
                static_cast<double>(offending_value),
                offending_row,
                offending_col
            );
            std::abort();
        }
    }

    static void check_transformer_ff_nan_finding_mode(
        VulkanQueue& queue,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& matrix,
        PositionIndex rows,
        const char* name,
        const char* phase
    )
    {
        if (!nan_finding_mode_enabled())
            return;

        static constexpr float bound = 10000.0f;
        auto flag = make_nan_scan_flag(queue);
        scan_transformer_ff_matrix(queue, matrix, flag, rows, -bound, bound);
        if (nan_scan_failed(queue, flag))
        {
            std::fprintf(
                stderr,
                "NAN_FINDING_MODE: TransformerBlock %s invalid during %s; expected finite values in [%g, %g]\n",
                name,
                phase,
                static_cast<double>(-bound),
                static_cast<double>(bound)
            );
            std::abort();
        }
    }

    // ── randomize ─────────────────────────────────────────────────────────────

    void TransformerBlock::randomize()
    {
        const float sd = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(EmbeddingDimension::MAX)));
        const float sf = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(FFDimension::MAX)));

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        W_q.fill_rand(queue, -sd, sd);
        W_k.fill_rand(queue, -sd, sd);
        W_v.fill_rand(queue, -sd, sd);
        W_o.fill_rand(queue, -sd, sd);
        W_gate.fill_rand(queue, -sd, sd);
        W_up.fill_rand(queue, -sd, sd);
        W_down.fill_rand(queue, -sf, sf);

        V_q.zero(queue);
        V_k.zero(queue);
        V_v.zero(queue);
        V_o.zero(queue);
        V_gate.zero(queue);
        V_up.zero(queue);
        V_down.zero(queue);
    }

    // ── normalisation ──────────────────────────────────────────────────────────


    void TransformerBlock::rms_norm(
        // OFFLOAD_PARAMETERS(x, y)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& y
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, t, enum_iterator1D<PositionIndex>(x.num_rows()), (x, y))
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
    void TransformerBlock::rms_norm_backward(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(dy, x, dx)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dy,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dx
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(queue, t, enum_iterator1D<PositionIndex>(x.num_rows()), (dy, x, dx))
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
            const float raw = (dx[t, i] + (inv * (dy[t, i] - ((x[t, i] * inv) * dot))));
            dx[t, i] = math::clamp(raw, -10000.0f, 10000.0f);
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(T), (x, T))
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_PARAM(queue, t, f, gate_range, (gate_pre, up_pre, d_ffn_act, d_gate_pre, d_up_pre))
        const float g = gate_pre[t, f];
        const float sg = (1.0f / (1.0f + std::exp(-g))); // sigma(g)
        const float silu = (g * sg);
        // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
        // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
        const float dsilu_dg = (sg * (1.0f + (g * (1.0f - sg))));
        d_gate_pre[t, f] = math::clamp(((d_ffn_act[t, f] * up_pre[t, f]) * dsilu_dg), -10000.0f, 10000.0f);
        d_up_pre[t, f] = math::clamp((d_ffn_act[t, f] * silu), -10000.0f, 10000.0f);
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
        // OFFLOAD_PARAMETERS(attn_w_h, Q, K, active_seq_len, hStart)
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h = ws.attn_w[hi];
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& Q = ws.Q;
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& K = ws.K;
        const PositionIndex active_seq_len = seq_len;
        const int hStart = (static_cast<int>(hi) * static_cast<int>(HeadDimension::MAX));
        // END_OFFLOAD_PARAMETERS

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, i, j, active_seq_len, (attn_w_h, Q, K, active_seq_len, hStart))
        {
            float dot = 0.f;
            for (const auto d_head : enum_iterator1D<HeadDimension>())
            {
                const int d = (hStart + int(d_head));
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

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, softmax_grid, (attn_w_h))
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
        // OFFLOAD_PARAMETERS(attn_w_h, V, attn_concat, active_seq_len, hStart)
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& attn_w_h,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& V,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& attn_concat,
        int active_seq_len,
        int hStart
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, i, j, static_cast<PositionIndex>(active_seq_len), (attn_concat, attn_w_h, V, active_seq_len, hStart))
        {
            const float w = attn_w_h[i, j];
            for (const auto d_head : enum_iterator1D<HeadDimension>())
            {
                const int d = (hStart + int(d_head));
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
        {
            const int hStart = (static_cast<int>(hi) * static_cast<int>(HeadDimension::MAX));
            compute_attention_values_for_head(attn_w[hi], V, attn_concat, active_seq_len, hStart);
        }
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

        check_nan_finding_mode("forward:start");
        ws.h_in = h;

        // ── 1. Pre-norm (attention) ──────────────────────────────────────────
        //rms_norm_vulkan_optimized_with_workspace(h, ws.h_norm_attn, ws.rms_norm_row_sums, ws.rms_norm_inv_rms);
        rms_norm(h, ws.h_norm_attn);

        // ── 2. Q / K / V projections ─────────────────────────────────────────
        matmul_ABt_3_matrix_muls(ws.h_norm_attn, W_q, ws.Q, W_k, ws.K, W_v, ws.V);

        // ── 3. Multi-head causal self-attention ──────────────────────────────
        {
            auto& queue = rllm::vulkan_runtime::get_queue(0);
            ws.attn_concat.zero(queue);
        }
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
        check_nan_finding_mode("forward:end");
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto dv_grid = enum_iterator2D<PositionIndex, HeadDimension>(seq_len);
        OFFLOAD_PARFOR_2D_PARAM(queue, j, d_head, dv_grid, (d_V, attn_w_h, d_attn_concat, seq_len, hStart))
        {
            const int d = (hStart + int(d_head));
            float sum_v = 0.f;
            for (const auto i : enum_iterator1D<PositionIndex>(j, seq_len))
                sum_v += (attn_w_h[i, j] * d_attn_concat[i, d]);
            d_V[j, d] = math::clamp(sum_v, -10000.0f, 10000.0f);
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
        // OFFLOAD_PARAMETERS(d_scores_h, d_attn_concat, V, seq_len, hStart)
        fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& d_scores_h,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& d_attn_concat,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& V,
        PositionIndex seq_len,
        int hStart
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, i, j, seq_len, (d_scores_h, d_attn_concat, V, seq_len, hStart))
        {
            float dot = 0.f;
            for (const auto d_head : enum_iterator1D<HeadDimension>())
            {
                const int d = (hStart + int(d_head));
                dot += (d_attn_concat[i, d] * V[j, d]);
            }
            d_scores_h[i, j] = math::clamp(dot, -10000.0f, 10000.0f);
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
            const int hStart = (static_cast<int>(hi) * static_cast<int>(HeadDimension::MAX));
            compute_attention_dscores_for_head(d_scores[hi], d_attn_concat, V, seq_len, hStart);
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, i, j, seq_len, (d_scores_h, d_raw_h, attn_w_h, seq_len))
        {
            float row_dot = 0.f;
            for (const auto k : enum_iterator1D<PositionIndex>(inc(i)))
            {
                row_dot += (d_scores_h[i, k] * attn_w_h[i, k]);
            }
            const float raw = (attn_w_h[i, j] * (d_scores_h[i, j] - row_dot));
            d_raw_h[i, j] = math::clamp(raw, -10000.0f, 10000.0f);
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);

        OFFLOAD_PARFOR_2D_PARAM(queue, i, d_head, dq_grid, (d_Q, d_raw_h, K, seq_len, hStart))
        {
            const int d = (hStart + int(d_head));
            const float head_scale = (1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX))));
            float sum_q = 0.f;
            for (const auto j : enum_iterator1D<PositionIndex>(inc(i)))
                sum_q += ((d_raw_h[i, j] * head_scale) * K[j, d]);
            d_Q[i, d] = math::clamp(sum_q, -10000.0f, 10000.0f);
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_PARAM(queue, j, d_head, dk_grid, (d_K, d_raw_h, Q, seq_len, hStart))
        {
            const int d = (hStart + int(d_head));
            float sum_k = 0.f;
            for (const auto i : enum_iterator1D<PositionIndex>(j, seq_len))
                sum_k += ((d_raw_h[i, j] * (1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX))))) * Q[i, d]);
            d_K[j, d] = math::clamp(sum_k, -10000.0f, 10000.0f);
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

    void TransformerBlock::backward(const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dout, flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& din, BackwardWorkspace& workspace,
        ForwardWorkspace& fwd_workspace, bool wait_for_completion)
    {
        const PositionIndex seq = fwd_workspace.seq_len;
        VulkanQueue& queue = vulkan_runtime::get_queue(0);
        workspace.reset(queue, seq);
        auto* ws = &workspace;
        auto& fwd = fwd_workspace;

        check_nan_finding_mode("backward:start");
        check_transformer_hidden_nan_finding_mode(queue, dout, seq, "dout", "backward:start");

        // ── FFN backward ──────────────────────────────────────────────────────
        // h_out = h_mid + ffn_out  → d_h_mid += dout,  d_ffn_out = dout (same buffer)
        ws->d_h_mid = dout; // copy first to avoid overwriting dout before it's used in dW_down
        check_transformer_hidden_nan_finding_mode(queue, ws->d_h_mid, seq, "d_h_mid", "after residual copy from dout");

        // d_ffn_act = d_ffn_out @ W_down   (W_down[D × Dff])
        matmul_AB(dout, W_down, ws->d_ffn_act);
        check_transformer_ff_nan_finding_mode(queue, ws->d_ffn_act, seq, "d_ffn_act", "after FFN down projection backward");

        // dW_down[D, Dff] += dout^T @ ffn_act
        matmul_AtB_acc(dout, fwd.ffn_act, ws->dW_down, fwd.seq_len);

        // SwiGLU backward: d(silu(g)*u) / dg, du
        swiglu_backward(fwd.seq_len, fwd.gate_pre, fwd.up_pre, ws->d_ffn_act, ws->d_gate_pre, ws->d_up_pre);
        check_transformer_ff_nan_finding_mode(queue, ws->d_gate_pre, seq, "d_gate_pre", "after SwiGLU backward");
        check_transformer_ff_nan_finding_mode(queue, ws->d_up_pre, seq, "d_up_pre", "after SwiGLU backward");

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        matmul_AB_add(ws->d_gate_pre, W_gate, ws->d_h_norm_ff);
        matmul_AB_add(ws->d_up_pre, W_up, ws->d_h_norm_ff);
        check_transformer_hidden_nan_finding_mode(queue, ws->d_h_norm_ff, seq, "d_h_norm_ff", "after FFN gate/up projection backward");

        // weight gradients for gate, up
        matmul_AtB_acc_2_matrix(ws->d_gate_pre, ws->d_up_pre, fwd.h_norm_ff, ws->dW_gate, ws->dW_up, fwd.seq_len);

        // RMSNorm backward for FFN: d_h_mid += rms_bwd(d_h_norm_ff, h_mid)
        rms_norm_backward(queue, ws->d_h_norm_ff, fwd.h_mid, ws->d_h_mid);
        check_transformer_hidden_nan_finding_mode(queue, ws->d_h_mid, seq, "d_h_mid", "after FFN RMSNorm backward");

        // ── Attention backward ─────────────────────────────────────────────────
        // h_mid = h_in + attn_proj  → d_attn_proj = d_h_mid (passed through residual)
        //                             d_h_in (residual part) = d_h_mid (accumulated below)

        // d_attn_concat = d_attn_proj @ W_o
        matmul_AB(ws->d_h_mid, W_o, ws->d_attn_concat);
        check_transformer_hidden_nan_finding_mode(queue, ws->d_attn_concat, seq, "d_attn_concat", "after output projection backward");
        matmul_AtB_acc(ws->d_h_mid, fwd.attn_concat, ws->dW_o, fwd.seq_len);

        // Per-head backward
        backward_attention_heads(*ws, fwd);
        check_transformer_hidden_nan_finding_mode(queue, ws->d_Q, seq, "d_Q", "after attention heads backward");
        check_transformer_hidden_nan_finding_mode(queue, ws->d_K, seq, "d_K", "after attention heads backward");
        check_transformer_hidden_nan_finding_mode(queue, ws->d_V, seq, "d_V", "after attention heads backward");

        // Weight gradients for W_q, W_k, W_v
        matmul_AtB_acc_3_matrix(ws->d_Q, ws->d_K, ws->d_V, fwd.h_norm_attn, ws->dW_q, ws->dW_k, ws->dW_v, fwd.seq_len);

        // d_h_norm_attn = d_Q @ W_q  +  d_K @ W_k  +  d_V @ W_v
        matmul_AB_add_3_matrix_muls(ws->d_Q, W_q, ws->d_K, W_k, ws->d_V, W_v, ws->d_h_norm_attn);
        check_transformer_hidden_nan_finding_mode(queue, ws->d_h_norm_attn, seq, "d_h_norm_attn", "after QKV projection backward");

        // d_h_in = residual gradient + RMSNorm backward. Weight updates are applied later
        // from the accumulated gradient buffers.
        din.copy_from(queue, ws->d_h_mid);
        rms_norm_backward(queue, ws->d_h_norm_attn, fwd.h_in, din);
        if (wait_for_completion)
            queue.wait("TransformerBlock backward din wait idle");
        check_transformer_hidden_nan_finding_mode(queue, din, seq, "din", "after final RMSNorm backward");

        check_nan_finding_mode("backward:end");
    }

    void TransformerBlock::accumulate_gradients(const BackwardWorkspace& workspace, TransformerGradientAccumulator& accumulator)
    {
        add_dmodel_ff_gradient(workspace.dW_down, accumulator.dW_down);
        add_ff_dmodel_gradient(workspace.dW_gate, accumulator.dW_gate);
        add_ff_dmodel_gradient(workspace.dW_up, accumulator.dW_up);
        add_dmodel_dmodel_gradient(workspace.dW_o, accumulator.dW_o);
        add_dmodel_dmodel_gradient(workspace.dW_q, accumulator.dW_q);
        add_dmodel_dmodel_gradient(workspace.dW_k, accumulator.dW_k);
        add_dmodel_dmodel_gradient(workspace.dW_v, accumulator.dW_v);
        accumulator.touched = true;
    }

    void TransformerBlock::apply_accumulated_update(TransformerGradientAccumulator& accumulator, float learning_rate)
    {
        if (!accumulator.touched)
            return;
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        sgd_update_Wqkvo_x_Vqkvo_dWqkvo__4_matrix(queue, W_q, V_q, accumulator.dW_q, W_k, V_k, accumulator.dW_k, W_v, V_v, accumulator.dW_v, W_o, V_o, accumulator.dW_o, learning_rate * EMBEDDING_LEARNING_RATE_SCALE);
        queue.wait("TransformerBlock accumulated qkvo wait idle");
        sgd_update_Wgateup_x_Vgateup_dWgateup__2_matrix(queue, W_gate, V_gate, accumulator.dW_gate, W_up, V_up, accumulator.dW_up, learning_rate * EMBEDDING_LEARNING_RATE_SCALE);
        queue.wait("TransformerBlock accumulated gateup wait idle");
        sgd_update_Wdown_x_Vdown_dWdown(queue, W_down, V_down, accumulator.dW_down, learning_rate * FF_LEARNING_RATE_SCALE);
        queue.wait("TransformerBlock accumulated down wait idle");
    }

    // ── serialisation ──────────────────────────────────────────────────────────

     std::unique_ptr<nlohmann::json> TransformerBlock::save() const
    {
        using namespace json_helpers;
        cpu_fixed_matrix<float16, EmbeddingDimension, EmbeddingDimension> cpu_Wq, cpu_Wk, cpu_Wv, cpu_Wo;
        cpu_fixed_matrix<float16, FFDimension, EmbeddingDimension> cpu_Wgate, cpu_Wup;
        cpu_fixed_matrix<float16, EmbeddingDimension, FFDimension> cpu_Wdown;
        {
            auto& queue = rllm::vulkan_runtime::get_queue(0);
            W_q.copy_to_cpu(queue, cpu_Wq);
            W_k.copy_to_cpu(queue, cpu_Wk);
            W_v.copy_to_cpu(queue, cpu_Wv);
            W_o.copy_to_cpu(queue, cpu_Wo);
            W_gate.copy_to_cpu(queue, cpu_Wgate);
            W_up.copy_to_cpu(queue, cpu_Wup);
            W_down.copy_to_cpu(queue, cpu_Wdown);
        }
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
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        W_q.copy_from_cpu(queue, cpu_Wq);
        W_k.copy_from_cpu(queue, cpu_Wk);
        W_v.copy_from_cpu(queue, cpu_Wv);
        W_o.copy_from_cpu(queue, cpu_Wo);
        W_gate.copy_from_cpu(queue, cpu_Wgate);
        W_up.copy_from_cpu(queue, cpu_Wup);
        W_down.copy_from_cpu(queue, cpu_Wdown);

        // Reset momentum on load — do not persist transient training state
        V_q.zero(queue);
        V_k.zero(queue);
        V_v.zero(queue);
        V_o.zero(queue);
        V_gate.zero(queue);
        V_up.zero(queue);
        V_down.zero(queue);
    }

} // namespace rllm
