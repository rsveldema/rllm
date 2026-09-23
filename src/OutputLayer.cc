#include <OutputLayer.hpp>
#include <OptimizerDiagnostics.hpp>
#include <WeightInitialization.hpp>
#include <RandomHelpers.hpp>
#include <RuntimeConfig.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <parallel.hpp>

#include <enum_iterator2D.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <omp.h>

namespace rllm
{
    using uint = unsigned int;
    static constexpr float NAN_FINDING_H_LAST_ABS_BOUND = 10000.0f;
    static constexpr float NAN_FINDING_LOGIT_ABS_BOUND = 19200.0f;

    // Compute a reasonable upper bound for cross-entropy loss based on the
    // observed logit spread and vocabulary size.
    // loss = max_val - expected_logit + log(sum_exp)
    // When the model is confident but wrong, loss ≈ logit_spread.
    // We allow a margin proportional to the spread plus the uniform entropy.
    static float compute_max_reasonable_loss(float max_val, float expected_logit)
    {
        const float logit_spread = max_val - expected_logit;
        const float uniform_entropy = std::log(static_cast<float>(static_cast<size_t>(TokenID::MAX)));
        // Allow 1.5x the logit spread plus uniform entropy as a safety margin.
        return logit_spread * 1.5f + uniform_entropy;
    }

    static float compute_max_reasonable_index_loss(float max_val, float expected_logit)
    {
        const float logit_spread = max_val - expected_logit;
        const float uniform_entropy = std::log(static_cast<float>(static_cast<size_t>(PositionIndex::MAX)));
        return logit_spread * 1.5f + uniform_entropy;
    }

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

    static void scan_output_layer_weight_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
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

    static void scan_output_layer_weight_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float16, PositionIndex, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>();
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

    static void scan_output_layer_velocity_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float, TokenID, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
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

    static void scan_output_layer_velocity_matrix(
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>();
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

    static void scan_output_embedding_vector(
        // OFFLOAD_PARAMETERS(values, flag, lower_bound, upper_bound)
        const fixed_size_vector<float, EmbeddingDimension>& values,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<EmbeddingDimension>(), (values, flag, lower_bound, upper_bound))
        const float value = values[i];
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

    static void scan_output_token_vector(
        // OFFLOAD_PARAMETERS(values, flag, lower_bound, upper_bound)
        const fixed_size_vector<float, TokenID>& values,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TokenID>(), (values, flag, lower_bound, upper_bound))
        const float value = values[i];
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

    static void scan_output_position_vector(
        // OFFLOAD_PARAMETERS(values, flag, lower_bound, upper_bound)
        const fixed_size_vector<float, PositionIndex>& values,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(), (values, flag, lower_bound, upper_bound))
        const float value = values[i];
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

    void OutputLayer::check_nan_finding_mode(const char* phase)
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_nan_scan_flag(queue);
        scan_output_layer_weight_matrix(W_lm_head, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_output_layer_weight_matrix(W_string_table_index_head, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_output_layer_velocity_matrix(V_lm_head, flag, -GRAD_CLIP, GRAD_CLIP);
        scan_output_layer_velocity_matrix(S_lm_head, flag, 0.0f, GRAD_CLIP * GRAD_CLIP);
        scan_output_layer_velocity_matrix(V_string_table_index_head, flag, -GRAD_CLIP, GRAD_CLIP);
        scan_output_layer_velocity_matrix(S_string_table_index_head, flag, 0.0f, GRAD_CLIP * GRAD_CLIP);
        if (nan_scan_failed(queue, flag))
        {
            std::fprintf(stderr, "NAN_FINDING_MODE: OutputLayer weights/Adam moments invalid during %s\n", phase);
            std::abort();
        }
    }

    static void check_output_vector_scan_failed(
        const char* name,
        const char* phase,
        fixed_size_vector<int, PositionIndex>& flag,
        VulkanQueue& queue,
        float bound
    )
    {
        if (nan_scan_failed(queue, flag))
        {
            std::fprintf(
                stderr,
                "NAN_FINDING_MODE: OutputLayer %s invalid during %s; expected finite values in [%g, %g]\n",
                name,
                phase,
                static_cast<double>(-bound),
                static_cast<double>(bound)
            );
            std::abort();
        }
    }

    static void check_output_h_last_nan_finding_mode(
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        const char* phase
    )
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_nan_scan_flag(queue);
        scan_output_embedding_vector(h_last, flag, -NAN_FINDING_H_LAST_ABS_BOUND, NAN_FINDING_H_LAST_ABS_BOUND);
        check_output_vector_scan_failed("h_last", phase, flag, queue, NAN_FINDING_H_LAST_ABS_BOUND);
    }

    static void check_output_logits_nan_finding_mode(
        const fixed_size_vector<float, TokenID>& logits,
        const char* phase
    )
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_nan_scan_flag(queue);
        scan_output_token_vector(logits, flag, -NAN_FINDING_LOGIT_ABS_BOUND, NAN_FINDING_LOGIT_ABS_BOUND);

        if (nan_scan_failed(queue, flag))
        {
            cpu_fixed_vector<float, TokenID> cpu_logits;
            cpu_logits.set_size(TokenID::MAX);
            logits.copy_to_cpu(queue, cpu_logits);

            float offending_value = std::nanf("");
            int offending_idx = -1;
            for (const auto i : enum_iterator1D<TokenID>())
            {
                const float v = cpu_logits[i];
                if (v != v || v < -NAN_FINDING_LOGIT_ABS_BOUND || v > NAN_FINDING_LOGIT_ABS_BOUND)
                {
                    offending_value = v;
                    offending_idx = static_cast<int>(static_cast<size_t>(i));
                    break;
                }
            }

            std::fprintf(
                stderr,
                "NAN_FINDING_MODE: OutputLayer logits invalid during %s; expected finite values in [%g, %g], "
                "actual value=%g at index=%d\n",
                phase,
                static_cast<double>(-NAN_FINDING_LOGIT_ABS_BOUND),
                static_cast<double>(NAN_FINDING_LOGIT_ABS_BOUND),
                static_cast<double>(offending_value),
                offending_idx
            );
            std::abort();
        }
    }

    static void check_output_index_logits_nan_finding_mode(
        const fixed_size_vector<float, PositionIndex>& logits,
        const char* phase
    )
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_nan_scan_flag(queue);
        scan_output_position_vector(logits, flag, -NAN_FINDING_LOGIT_ABS_BOUND, NAN_FINDING_LOGIT_ABS_BOUND);
        check_output_vector_scan_failed("string_table_index_logits", phase, flag, queue, NAN_FINDING_LOGIT_ABS_BOUND);
    }

    OutputLayer::OutputLayer()
    {
        m_inputs.set_size(TokenID::MAX);
        m_string_table_index_inputs.set_size(PositionIndex::MAX);
        m_string_table_index_inputs_cpu.set_size(PositionIndex::MAX);
    }

    void output_layer_forward_from_hidden_impl(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(h_last, W, inputs)
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        fixed_size_vector<float, TokenID>& inputs
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(queue, v, enum_iterator1D<TokenID>(), (h_last, W, inputs))
        float sum = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
        {
            const float term = (h_last[d] * static_cast<float>(W[v, d]));
            sum += term;
        }
        inputs[v] = math::clamp(sum, -19200.0f, 19200.0f);
        ENDFOR
    }

    static void output_layer_forward_string_table_index_from_hidden_impl(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(h_last, W, inputs)
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float16, PositionIndex, EmbeddingDimension>& W,
        fixed_size_vector<float, PositionIndex>& inputs
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(queue, v, enum_iterator1D<PositionIndex>(), (h_last, W, inputs))
        float sum = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            sum += (h_last[d] * static_cast<float>(W[v, d]));
        inputs[v] = math::clamp(sum, -19200.0f, 19200.0f);
        ENDFOR
    }

    static void output_layer_forward_batched_impl(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(h_last, W, logits, batch_size)
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, v, grid, (h_last, W, logits, batch_size))
        float sum = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            sum += (h_last[batch, d] * static_cast<float>(W[v, d]));
        logits[batch, v] = math::clamp(sum, -19200.0f, 19200.0f);
        ENDFOR
    }

    static void output_layer_forward_batched_string_table_index_impl(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(h_last, W, logits, batch_size, index_count)
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float16, PositionIndex, EmbeddingDimension>& W,
        fixed_size_matrix<float, BatchIndex, PositionIndex>& logits,
        int batch_size,
        PositionIndex index_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<BatchIndex, PositionIndex>(
            static_cast<BatchIndex>(batch_size), index_count);
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, v, grid, (h_last, W, logits, batch_size, index_count))
        float sum = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            sum += (h_last[batch, d] * static_cast<float>(W[v, d]));
        logits[batch, v] = math::clamp(sum, -19200.0f, 19200.0f);
        ENDFOR
    }

    static void accumulate_batched_output_layer_dh_last(
        // OFFLOAD_PARAMETERS(delta, dh_last, W, batch_size)
        const fixed_size_matrix<float, BatchIndex, TokenID>& delta,
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& dh_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, d, grid, (delta, dh_last, W, batch_size))
        float sum = dh_last[batch, d];
        for (const auto v : enum_iterator1D<TokenID>())
            sum += (delta[batch, v] * W[v, d]);
        dh_last[batch, d] = sum;
        ENDFOR
    }

    static void accumulate_batched_output_layer_string_table_index_dh_last(
        // OFFLOAD_PARAMETERS(delta, dh_last, W, batch_size, index_count)
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& delta,
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& dh_last,
        const fixed_size_matrix<float16, PositionIndex, EmbeddingDimension>& W,
        int batch_size,
        PositionIndex index_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, d, grid, (delta, dh_last, W, batch_size, index_count))
        float sum = dh_last[batch, d];
        for (const auto v : enum_iterator1D<PositionIndex>(index_count))
            sum += (delta[batch, v] * W[v, d]);
        dh_last[batch, d] = sum;
        ENDFOR
    }

    static void accumulate_batched_output_layer_dW(
        // OFFLOAD_PARAMETERS(delta, h_last, dW, batch_size)
        const fixed_size_matrix<float, BatchIndex, TokenID>& delta,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& dW,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (delta, h_last, dW, batch_size))
        float grad = dW[v, d];
        for (const auto batch : enum_iterator1D<BatchIndex>(static_cast<BatchIndex>(batch_size)))
            grad += (delta[batch, v] * h_last[batch, d]);
        dW[v, d] = grad;
        ENDFOR
    }

    static void accumulate_batched_output_layer_string_table_index_dW(
        // OFFLOAD_PARAMETERS(delta, h_last, dW, batch_size, index_count)
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& delta,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& dW,
        int batch_size,
        PositionIndex index_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(index_count);
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (delta, h_last, dW, batch_size, index_count))
        float grad = dW[v, d];
        for (const auto batch : enum_iterator1D<BatchIndex>(static_cast<BatchIndex>(batch_size)))
            grad += (delta[batch, v] * h_last[batch, d]);
        dW[v, d] = grad;
        ENDFOR
    }

    static void accumulate_output_layer_dh_last(
        // OFFLOAD_PARAMETERS(delta, dh_last, W)
        const fixed_size_vector<float, TokenID>& delta,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, d, enum_iterator1D<EmbeddingDimension>(), (delta, dh_last, W))
        float sum = dh_last[d];
        for (const auto v : enum_iterator1D<TokenID>())
        {
            const float term = (delta[v] * W[v, d]);
            sum += term;
        }
        dh_last[d] = sum;
        ENDFOR
    }

    static void accumulate_output_layer_string_table_index_dh_last(
        // OFFLOAD_PARAMETERS(delta, dh_last, W)
        const fixed_size_vector<float, PositionIndex>& delta,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        const fixed_size_matrix<float16, PositionIndex, EmbeddingDimension>& W
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, d, enum_iterator1D<EmbeddingDimension>(), (delta, dh_last, W))
        float sum = dh_last[d];
        for (const auto v : enum_iterator1D<PositionIndex>())
            sum += (delta[v] * W[v, d]);
        dh_last[d] = sum;
        ENDFOR
    }

    static void accumulate_output_layer_dW(
        // OFFLOAD_PARAMETERS(delta, h_last, dW)
        const fixed_size_vector<float, TokenID>& delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& dW
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (delta, h_last, dW))
        const float old_value = dW[v, d];
        const float grad = (delta[v] * h_last[d]);
        const float new_value = (old_value + grad);
        dW[v, d] = new_value;
        ENDFOR
    }

    static void accumulate_output_layer_string_table_index_dW(
        // OFFLOAD_PARAMETERS(delta, h_last, dW)
        const fixed_size_vector<float, PositionIndex>& delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& dW
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (delta, h_last, dW))
        dW[v, d] += (delta[v] * h_last[d]);
        ENDFOR
    }

    static void zero_output_layer_string_table_index_dW_rows(
        // OFFLOAD_PARAMETERS(dW, rows)
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& dW,
        PositionIndex rows
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (dW, rows))
        dW[v, d] = 0.0f;
        ENDFOR
    }

    static void update_output_layer_weights_from_gradient(
        // OFFLOAD_PARAMETERS(dW, W, V, S, learning_rate, bias_correction1, bias_correction2, diagnostics, collect_diagnostics)
        const fixed_size_matrix<float, TokenID, EmbeddingDimension>& dW,
        fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& V,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& S,
        float learning_rate,
        float bias_correction1,
        float bias_correction2,
        fixed_size_vector<float, TempStorage>& diagnostics,
        int collect_diagnostics
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (dW, W, V, S, learning_rate, bias_correction1, bias_correction2, diagnostics, collect_diagnostics))
        const float raw_gradient = dW[v, d];
        const float scaled_gradient = (raw_gradient * diagnostics[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE]);
        const float g = math::clamp(scaled_gradient, -OutputLayer::GRAD_CLIP, OutputLayer::GRAD_CLIP);
        V[v, d] = ((OutputLayer::ADAM_BETA1 * V[v, d]) + ((1.0f - OutputLayer::ADAM_BETA1) * g));
        S[v, d] = ((OutputLayer::ADAM_BETA2 * S[v, d]) + ((1.0f - OutputLayer::ADAM_BETA2) * (g * g)));
        const float update = ((V[v, d] / bias_correction1) / (sqrt((S[v, d] / bias_correction2)) + OutputLayer::ADAM_EPSILON));
        const float decayed = (static_cast<float>(W[v, d]) * (1.0f - (learning_rate * OutputLayer::WEIGHT_DECAY)));
        const float old_weight = static_cast<float>(W[v, d]);
        const float new_weight = math::clamp((decayed + (learning_rate * update)), -OutputLayer::WEIGHT_CLAMP, OutputLayer::WEIGHT_CLAMP);
        W[v, d] = new_weight;
        if (collect_diagnostics != 0) {
            const float signed_weight_delta = (new_weight - old_weight);
            const float weight_delta = abs(signed_weight_delta);
            atomicMax(diagnostics[TempStorage::OPTIMIZER_GRADIENT_MAX], abs(raw_gradient));
            if (abs(scaled_gradient) > OutputLayer::GRAD_CLIP) atomicAdd(diagnostics[TempStorage::OPTIMIZER_CLIPPED_COUNT], 1.0f);
            atomicAdd(diagnostics[TempStorage::OPTIMIZER_ADAM_UPDATE_SQUARE_SUM], (update * update));
            atomicMax(diagnostics[TempStorage::OPTIMIZER_ADAM_UPDATE_MAX], abs(update));
            atomicAdd(diagnostics[TempStorage::OPTIMIZER_WEIGHT_UPDATE_SQUARE_SUM], (weight_delta * weight_delta));
            atomicMax(diagnostics[TempStorage::OPTIMIZER_WEIGHT_UPDATE_MAX], weight_delta);
            atomicAdd(diagnostics[TempStorage::OPTIMIZER_PARAMETER_COUNT], 1.0f);
        }
        ENDFOR
    }

    static void update_output_layer_weights_from_gradient(
        // OFFLOAD_PARAMETERS(dW, W, V, S, learning_rate, bias_correction1, bias_correction2, diagnostics, collect_diagnostics, rows)
        const fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& dW,
        fixed_size_matrix<float16, PositionIndex, EmbeddingDimension>& W,
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& V,
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension>& S,
        float learning_rate,
        float bias_correction1,
        float bias_correction2,
        fixed_size_vector<float, TempStorage>& diagnostics,
        int collect_diagnostics,
        PositionIndex rows
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (dW, W, V, S, learning_rate, bias_correction1, bias_correction2, diagnostics, collect_diagnostics, rows))
        const float raw_gradient = dW[v, d];
        const float scaled_gradient = (raw_gradient * diagnostics[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE]);
        const float g = math::clamp(scaled_gradient, -OutputLayer::GRAD_CLIP, OutputLayer::GRAD_CLIP);
        V[v, d] = ((OutputLayer::ADAM_BETA1 * V[v, d]) + ((1.0f - OutputLayer::ADAM_BETA1) * g));
        S[v, d] = ((OutputLayer::ADAM_BETA2 * S[v, d]) + ((1.0f - OutputLayer::ADAM_BETA2) * (g * g)));
        const float update = ((V[v, d] / bias_correction1) / (sqrt((S[v, d] / bias_correction2)) + OutputLayer::ADAM_EPSILON));
        const float decayed = (static_cast<float>(W[v, d]) * (1.0f - (learning_rate * OutputLayer::WEIGHT_DECAY)));
        const float old_weight = static_cast<float>(W[v, d]);
        const float new_weight = math::clamp((decayed + (learning_rate * update)), -OutputLayer::WEIGHT_CLAMP, OutputLayer::WEIGHT_CLAMP);
        W[v, d] = new_weight;
        if (collect_diagnostics != 0) {
            const float signed_weight_delta = (new_weight - old_weight);
            const float weight_delta = abs(signed_weight_delta);
            atomicMax(diagnostics[TempStorage::OPTIMIZER_GRADIENT_MAX], abs(raw_gradient));
            if (abs(scaled_gradient) > OutputLayer::GRAD_CLIP) atomicAdd(diagnostics[TempStorage::OPTIMIZER_CLIPPED_COUNT], 1.0f);
            atomicAdd(diagnostics[TempStorage::OPTIMIZER_ADAM_UPDATE_SQUARE_SUM], (update * update));
            atomicMax(diagnostics[TempStorage::OPTIMIZER_ADAM_UPDATE_MAX], abs(update));
            atomicAdd(diagnostics[TempStorage::OPTIMIZER_WEIGHT_UPDATE_SQUARE_SUM], (weight_delta * weight_delta));
            atomicMax(diagnostics[TempStorage::OPTIMIZER_WEIGHT_UPDATE_MAX], weight_delta);
            atomicAdd(diagnostics[TempStorage::OPTIMIZER_PARAMETER_COUNT], 1.0f);
        }
        ENDFOR
    }

    [[maybe_unused]] static void initialize_softmax_temp_values(
        // OFFLOAD_PARAMETERS(temp_values)
        fixed_size_vector<float, TempStorage>& temp_values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TempStorage>(), (temp_values))
        temp_values[i] = (i == TempStorage::START) ? -3.402823e38f : 0.0f;
        ENDFOR
    }

    [[maybe_unused]] static void reduce_logits_max_to_temp(
        // OFFLOAD_PARAMETERS(inputs, temp_values)
        const fixed_size_vector<float, TokenID>& inputs,
        fixed_size_vector<float, TempStorage>& temp_values
        // END_OFFLOAD_PARAMETERS
    )
    {
        // PARFOR_SHARED_VARIABLES(workgroup_max)
        // ENDPARFOR_SHARED_VARIABLES

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TokenID>(), (inputs, temp_values))
        {
            atomicMax(temp_values[TempStorage::ZERO], inputs[i]);
        }
        ENDFOR
    }

    [[maybe_unused]] static void reduce_index_logits_max_to_temp(
        // OFFLOAD_PARAMETERS(inputs, temp_values)
        const fixed_size_vector<float, PositionIndex>& inputs,
        fixed_size_vector<float, TempStorage>& temp_values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(), (inputs, temp_values))
        atomicMax(temp_values[TempStorage::ZERO], inputs[i]);
        ENDFOR
    }

    [[maybe_unused]] static void compute_exp_and_accumulate_sum(
        // OFFLOAD_PARAMETERS(inputs, values, temp_values)
        const fixed_size_vector<float, TokenID>& inputs,
        fixed_size_vector<float, TokenID>& values,
        fixed_size_vector<float, TempStorage>& temp_values
        // END_OFFLOAD_PARAMETERS
    )
    {
        // PARFOR_SHARED_VARIABLES()
        // ENDPARFOR_SHARED_VARIABLES

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TokenID>(), (inputs, values, temp_values))
        {
            const float max_val = temp_values[TempStorage::ZERO];
            const float exp_value = exp((inputs[i] - max_val));
            values[i] = exp_value;

            atomicAdd(temp_values[TempStorage::ONE], exp_value);
        }
        ENDFOR
    }

    [[maybe_unused]] static void compute_index_exp_and_accumulate_sum(
        // OFFLOAD_PARAMETERS(inputs, values, temp_values)
        const fixed_size_vector<float, PositionIndex>& inputs,
        fixed_size_vector<float, PositionIndex>& values,
        fixed_size_vector<float, TempStorage>& temp_values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(), (inputs, values, temp_values))
        const float max_val = temp_values[TempStorage::ZERO];
        const float exp_value = exp((inputs[i] - max_val));
        values[i] = exp_value;
        atomicAdd(temp_values[TempStorage::ONE], exp_value);
        ENDFOR
    }

    [[maybe_unused]] static void finalize_softmax_delta(
        // OFFLOAD_PARAMETERS(values, temp_values, expected_output_token, weight)
        fixed_size_vector<float, TokenID>& values,
        const fixed_size_vector<float, TempStorage>& temp_values,
        TokenID expected_output_token,
        float weight
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TokenID>(), (values, temp_values, expected_output_token, weight))
        float sum_exp = temp_values[TempStorage::ONE];
        float delta = (OutputLayer::smooth - (values[i] / sum_exp));
        if (i == expected_output_token)
            delta += (1.0f - OutputLayer::LABEL_SMOOTHING);
        delta *= weight;
        values[i] = delta;
        ENDFOR
    }

    [[maybe_unused]] static void finalize_index_softmax_delta(
        // OFFLOAD_PARAMETERS(values, temp_values, expected_string_table_index, weight)
        fixed_size_vector<float, PositionIndex>& values,
        const fixed_size_vector<float, TempStorage>& temp_values,
        PositionIndex expected_string_table_index,
        float weight
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(), (values, temp_values, expected_string_table_index, weight))
        const float sum_exp = temp_values[TempStorage::ONE];
        float delta = -(values[i] / sum_exp);
        if (i == expected_string_table_index)
            delta += 1.0f;
        delta *= weight;
        values[i] = delta;
        ENDFOR
    }

    OutputLayer::OutputLayer(const Corpus& corpus)
        : OutputLayer()
    {
        (void) corpus;
    }

    void OutputLayerGradientAccumulator::reset(VulkanQueue& queue)
    {
        dW_lm_head.zero(queue);
        if (!string_table_index_gradients_initialized || string_table_index_count == PositionIndex::MAX)
        {
            dW_string_table_index_head.zero(queue);
            string_table_index_gradients_initialized = true;
        }
        else if (string_table_index_count != PositionIndex::START)
            zero_output_layer_string_table_index_dW_rows(dW_string_table_index_head, string_table_index_count);
        string_table_index_count = PositionIndex::START;
        touched = false;
    }

    void OutputLayer::set_random_weights(WeightInitializerType type)
    {
        const size_t d_model = static_cast<size_t>(EmbeddingDimension::MAX);
        const size_t vocab_size = static_cast<size_t>(TokenID::MAX);
        // The mixed profile reserves Xavier for transformer input projections;
        // the LM head remains on the legacy scale.
        if (type == WeightInitializerType::XavierInputProjections)
            type = WeightInitializerType::LegacyUniform;
        auto initializer = make_weight_initializer(type, d_model, vocab_size);
        cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> cpu_tmp;
        for (const auto v : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                cpu_tmp.set(v, d, static_cast<float16>(initializer->getNextValue()));
        auto index_initializer = make_weight_initializer(type, d_model, static_cast<size_t>(PositionIndex::MAX));
        cpu_fixed_matrix<float16, PositionIndex, EmbeddingDimension> cpu_index_tmp;
        for (const auto v : enum_iterator1D<PositionIndex>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                cpu_index_tmp.set(v, d, static_cast<float16>(index_initializer->getNextValue()));
        {
            auto& queue = rllm::vulkan_runtime::get_queue(0);
            W_lm_head.copy_from_cpu(queue, cpu_tmp);
            W_string_table_index_head.copy_from_cpu(queue, cpu_index_tmp);
            V_lm_head.zero(queue);
            S_lm_head.zero(queue);
            V_string_table_index_head.zero(queue);
            S_string_table_index_head.zero(queue);
        }
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<float, EmbeddingDimension>& h_last,
              VulkanQueue& queue)
    {
        forward_from_hidden(h_last, m_inputs, m_inputs_cpu, queue);
    }

    void OutputLayer::forward_from_hidden(
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_vector<float, TokenID>& inputs,
        cpu_fixed_vector<float, TokenID>& inputs_cpu,
        VulkanQueue& queue
    )
    {
        check_nan_finding_mode("forward:start");
        check_output_h_last_nan_finding_mode(h_last, "forward:start");
        inputs.set_size(TokenID::MAX);
        output_layer_forward_from_hidden_impl(queue, h_last, W_lm_head, inputs);
        output_layer_forward_string_table_index_from_hidden_impl(
            queue, h_last, W_string_table_index_head, m_string_table_index_inputs);
        check_output_logits_nan_finding_mode(inputs, "forward:end");
        check_output_index_logits_nan_finding_mode(m_string_table_index_inputs, "forward:end");
        inputs.copy_to_cpu(queue, inputs_cpu);
        m_string_table_index_inputs.copy_to_cpu(queue, m_string_table_index_inputs_cpu);
        check_nan_finding_mode("forward:end");
    }

    void OutputLayer::backward_accumulate(
        const fixed_size_vector<float, TokenID>& delta,
        const fixed_size_vector<float, PositionIndex>& string_table_index_delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        OutputLayerGradientAccumulator& accumulator
    )
    {
        check_nan_finding_mode("backward_accumulate:start");
        accumulate_output_layer_dh_last(delta, dh_last, W_lm_head);
        accumulate_output_layer_dW(delta, h_last, accumulator.dW_lm_head);
        accumulate_output_layer_string_table_index_dh_last(
            string_table_index_delta, dh_last, W_string_table_index_head);
        accumulate_output_layer_string_table_index_dW(
            string_table_index_delta, h_last, accumulator.dW_string_table_index_head);
        accumulator.string_table_index_count = PositionIndex::MAX;
        accumulator.touched = true;
        check_nan_finding_mode("backward_accumulate:end");
    }

    void OutputLayer::forward_batched(
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
        BatchIndex batch_size,
        fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        VulkanQueue& queue
    )
    {
        output_layer_forward_batched_impl(queue, h_last, W_lm_head, logits, static_cast<int>(batch_size));
    }

    static void initialize_batched_softmax(
        // OFFLOAD_PARAMETERS(temp, losses, batch_size)
        fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        fixed_size_vector<float, BatchIndex>& losses,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TempStorage>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, slot, grid, (temp, losses, batch_size))
        temp[batch, slot] = (slot == TempStorage::START) ? -3.402823e38f : 0.0f;
        if (slot == TempStorage::START)
            losses[batch] = 3.402823e38f;
        ENDFOR
    }

    static void reduce_batched_logits_max(
        // OFFLOAD_PARAMETERS(logits, temp, active_examples, batch_size)
        const fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, token, grid, (logits, temp, active_examples, batch_size))
        if (active_examples[batch] != 0)
            atomicMax(temp[batch, TempStorage::START], logits[batch, token]);
        ENDFOR
    }

    static void compute_batched_exp_sum(
        // OFFLOAD_PARAMETERS(logits, delta, temp, active_examples, batch_size)
        const fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        fixed_size_matrix<float, BatchIndex, TokenID>& delta,
        fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, token, grid, (logits, delta, temp, active_examples, batch_size))
        if (active_examples[batch] != 0)
        {
            const float value = exp((logits[batch, token] - temp[batch, TempStorage::START]));
            delta[batch, token] = value;
            atomicAdd(temp[batch, static_cast<TempStorage>(1)], value);
        }
        else
            delta[batch, token] = 0.0f;
        ENDFOR
    }

    static void mark_batched_string_table_index_examples(
        // OFFLOAD_PARAMETERS(expected_string_table_indices, active_examples, string_table_index_active_examples, batch_size)
        const fixed_size_vector<int, BatchIndex>& expected_string_table_indices,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        fixed_size_vector<int, BatchIndex>& string_table_index_active_examples,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, batch, enum_iterator1D<BatchIndex>(static_cast<BatchIndex>(batch_size)), (expected_string_table_indices, active_examples, string_table_index_active_examples, batch_size))
        string_table_index_active_examples[batch] = 0;
        if (active_examples[batch] != 0)
            if (expected_string_table_indices[batch] != -1)
                string_table_index_active_examples[batch] = 1;
        ENDFOR
    }

    static void initialize_batched_index_softmax_temp(
        // OFFLOAD_PARAMETERS(temp, batch_size)
        fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        int batch_size
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TempStorage>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, slot, grid, (temp, batch_size))
        temp[batch, slot] = (slot == TempStorage::START) ? -3.402823e38f : 0.0f;
        ENDFOR
    }

    static void reduce_batched_index_logits_max(
        // OFFLOAD_PARAMETERS(logits, temp, active_examples, value_counts, batch_size, index_count)
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& logits,
        fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        const fixed_size_vector<int, BatchIndex>& value_counts,
        int batch_size,
        PositionIndex index_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, PositionIndex>(
            static_cast<BatchIndex>(batch_size), index_count);
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, index, grid, (logits, temp, active_examples, value_counts, batch_size, index_count))
        if (active_examples[batch] != 0)
            if (index < value_counts[batch])
                atomicMax(temp[batch, TempStorage::START], logits[batch, index]);
        ENDFOR
    }

    static void compute_batched_index_exp_sum(
        // OFFLOAD_PARAMETERS(logits, delta, temp, active_examples, value_counts, numeric_distance_costs, batch_size, index_count)
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& logits,
        fixed_size_matrix<float, BatchIndex, PositionIndex>& delta,
        fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        const fixed_size_vector<int, BatchIndex>& value_counts,
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& numeric_distance_costs,
        int batch_size,
        PositionIndex index_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, PositionIndex>(
            static_cast<BatchIndex>(batch_size), index_count);
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, index, grid, (logits, delta, temp, active_examples, value_counts, numeric_distance_costs, batch_size, index_count))
        delta[batch, index] = 0.0f;
        if (active_examples[batch] != 0)
            if (index < value_counts[batch])
            {
                const float value = exp((logits[batch, index] - temp[batch, TempStorage::START]));
                delta[batch, index] = value;
                atomicAdd(temp[batch, static_cast<TempStorage>(1)], value);
                float weighted_cost = numeric_distance_costs[batch, index];
                weighted_cost *= value;
                atomicAdd(temp[batch, static_cast<TempStorage>(2)], weighted_cost);
            }
        ENDFOR
    }

    static void finalize_batched_index_softmax_delta(
        // OFFLOAD_PARAMETERS(logits, delta, temp, expected_string_table_indices, active_examples, value_counts, numeric_distance_costs, losses, batch_size, index_count, loss_gradient_scale, reference_loss_weight, numeric_distance_loss_weight)
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& logits,
        fixed_size_matrix<float, BatchIndex, PositionIndex>& delta,
        const fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        const fixed_size_vector<int, BatchIndex>& expected_string_table_indices,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        const fixed_size_vector<int, BatchIndex>& value_counts,
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& numeric_distance_costs,
        fixed_size_vector<float, BatchIndex>& losses,
        int batch_size,
        PositionIndex index_count,
        float loss_gradient_scale,
        float reference_loss_weight,
        float numeric_distance_loss_weight
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, PositionIndex>(
            static_cast<BatchIndex>(batch_size), index_count);
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, index, grid, (logits, delta, temp, expected_string_table_indices, active_examples, value_counts, numeric_distance_costs, losses, batch_size, index_count, loss_gradient_scale, reference_loss_weight, numeric_distance_loss_weight))
        if (active_examples[batch] != 0)
        if (index < value_counts[batch])
        {
            const float sum_exp = temp[batch, static_cast<TempStorage>(1)];
            float probability = delta[batch, index];
            probability /= sum_exp;
            float expected_numeric_cost = temp[batch, static_cast<TempStorage>(2)];
            expected_numeric_cost /= sum_exp;
            float value = -probability;
            if (static_cast<int>(index) == expected_string_table_indices[batch])
            {
                value += 1.0f;
                float index_loss = temp[batch, TempStorage::START];
                index_loss += log(sum_exp);
                index_loss -= logits[batch, index];
                index_loss *= reference_loss_weight;
                losses[batch] += index_loss;
                float numeric_loss = expected_numeric_cost;
                numeric_loss *= numeric_distance_loss_weight;
                losses[batch] += numeric_loss;
            }
            float weighted_scale = loss_gradient_scale;
            weighted_scale *= reference_loss_weight;
            value *= weighted_scale;
            float candidate_cost = numeric_distance_costs[batch, index];
            float numeric_value = expected_numeric_cost;
            numeric_value -= candidate_cost;
            numeric_value *= probability;
            numeric_value *= numeric_distance_loss_weight;
            numeric_value *= loss_gradient_scale;
            value += numeric_value;
            delta[batch, index] = value;
        }
        ENDFOR
    }

    static void finalize_batched_softmax_delta(
        // OFFLOAD_PARAMETERS(logits, delta, temp, expected_tokens, active_examples, losses, correct_token_probabilities, top1_correct, batch_size, loss_gradient_scale, category_loss_bonus, local_token, param_token, global_token, field_token, loop_token, class_name_token)
        const fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        fixed_size_matrix<float, BatchIndex, TokenID>& delta,
        const fixed_size_matrix<float, BatchIndex, TempStorage>& temp,
        const fixed_size_vector<int, BatchIndex>& expected_tokens,
        const fixed_size_vector<int, BatchIndex>& active_examples,
        fixed_size_vector<float, BatchIndex>& losses,
        fixed_size_vector<float, BatchIndex>& correct_token_probabilities,
        fixed_size_vector<int, BatchIndex>& top1_correct,
        int batch_size,
        float loss_gradient_scale,
        float category_loss_bonus,
        int local_token, int param_token, int global_token,
        int field_token, int loop_token, int class_name_token
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(static_cast<BatchIndex>(batch_size));
        OFFLOAD_PARFOR_2D_PARAM(queue, batch, token, grid, (logits, delta, temp, expected_tokens, active_examples, losses, correct_token_probabilities, top1_correct, batch_size, loss_gradient_scale, category_loss_bonus, local_token, param_token, global_token, field_token, loop_token, class_name_token))
        if (active_examples[batch] != 0)
        {
            const int expected = expected_tokens[batch];
            int category = 0;
            if (expected == local_token) category = 1;
            if (expected == param_token) category = 1;
            if (expected == global_token) category = 1;
            if (expected == field_token) category = 1;
            if (expected == loop_token) category = 1;
            if (expected == class_name_token) category = 1;
            float weight = 1.0f;
            if (category != 0) weight += category_loss_bonus;
            const float sum_exp = temp[batch, static_cast<TempStorage>(1)];
            float value = (OutputLayer::smooth - (delta[batch, token] / sum_exp));
            if (static_cast<int>(token) == expected_tokens[batch])
            {
                float correct_probability = delta[batch, token];
                correct_probability /= sum_exp;
                correct_token_probabilities[batch] = correct_probability;
                top1_correct[batch] = 0;
                if (logits[batch, token] == temp[batch, TempStorage::START])
                {
                    int earlier_tie = 0;
                    for (const auto candidate : enum_iterator1D<TokenID>())
                        if (candidate < token)
                            if (logits[batch, candidate] == logits[batch, token])
                                earlier_tie = 1;
                    if (earlier_tie == 0)
                        top1_correct[batch] = 1;
                }
                value += (1.0f - OutputLayer::LABEL_SMOOTHING);
                float token_loss = temp[batch, TempStorage::START];
                token_loss += log(sum_exp);
                token_loss -= logits[batch, token];
                token_loss *= weight;
                losses[batch] = token_loss;
            }
            float weighted_scale = loss_gradient_scale;
            weighted_scale *= weight;
            value *= weighted_scale;
            delta[batch, token] = value;
        }
        ENDFOR
    }

    void OutputLayer::compute_batched_delta(
        const fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        BatchIndex batch_size,
        BatchedOutputWorkspace& workspace,
        VulkanQueue& queue,
        float loss_gradient_scale)
    {
        (void) queue;
        assert(loss_gradient_scale > 0.0f);
        const int count = static_cast<int>(batch_size);
        initialize_batched_softmax(workspace.softmax_temp, workspace.losses, count);
        workspace.correct_token_probabilities.zero(queue);
        workspace.top1_correct.zero(queue);
        reduce_batched_logits_max(logits, workspace.softmax_temp, workspace.active_examples, count);
        compute_batched_exp_sum(logits, workspace.delta, workspace.softmax_temp, workspace.active_examples, count);
        finalize_batched_softmax_delta(logits, workspace.delta, workspace.softmax_temp,
            workspace.expected_tokens, workspace.active_examples, workspace.losses,
            workspace.correct_token_probabilities, workspace.top1_correct, count,
            loss_gradient_scale, CATEGORY_LOSS_BONUS,
            static_cast<int>(TokenID::LOCAL), static_cast<int>(TokenID::PARAM),
            static_cast<int>(TokenID::GLOBAL), static_cast<int>(TokenID::FIELD),
            static_cast<int>(TokenID::LOOP), static_cast<int>(TokenID::CLASS_NAME));
        workspace.string_table_index_delta.zero(queue);
    }

    void OutputLayer::compute_batched_delta(
        const fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        BatchIndex batch_size,
        BatchedOutputWorkspace& workspace,
        VulkanQueue& queue,
        const fixed_size_vector<int, BatchIndex>& expected_string_table_indices,
        float loss_gradient_scale,
        PositionIndex string_table_index_count)
    {
        (void) queue;
        assert(loss_gradient_scale > 0.0f);
        const int count = static_cast<int>(batch_size);
        initialize_batched_softmax(workspace.softmax_temp, workspace.losses, count);
        workspace.correct_token_probabilities.zero(queue);
        workspace.top1_correct.zero(queue);
        reduce_batched_logits_max(logits, workspace.softmax_temp, workspace.active_examples, count);
        compute_batched_exp_sum(logits, workspace.delta, workspace.softmax_temp, workspace.active_examples, count);
        finalize_batched_softmax_delta(logits, workspace.delta, workspace.softmax_temp,
            workspace.expected_tokens, workspace.active_examples, workspace.losses,
            workspace.correct_token_probabilities, workspace.top1_correct, count,
            loss_gradient_scale, CATEGORY_LOSS_BONUS,
            static_cast<int>(TokenID::LOCAL), static_cast<int>(TokenID::PARAM),
            static_cast<int>(TokenID::GLOBAL), static_cast<int>(TokenID::FIELD),
            static_cast<int>(TokenID::LOOP), static_cast<int>(TokenID::CLASS_NAME));

        if (string_table_index_count == PositionIndex::START)
            return;

        mark_batched_string_table_index_examples(
            expected_string_table_indices, workspace.active_examples,
            workspace.string_table_index_active_examples, count);
        output_layer_forward_batched_string_table_index_impl(
            queue, workspace.h_last, W_string_table_index_head,
            workspace.string_table_index_logits, count, string_table_index_count);
        initialize_batched_index_softmax_temp(workspace.softmax_temp, count);
        reduce_batched_index_logits_max(workspace.string_table_index_logits, workspace.softmax_temp,
            workspace.string_table_index_active_examples, workspace.expected_value_counts,
            count, string_table_index_count);
        compute_batched_index_exp_sum(workspace.string_table_index_logits,
            workspace.string_table_index_delta, workspace.softmax_temp,
            workspace.string_table_index_active_examples, workspace.expected_value_counts,
            workspace.numeric_distance_costs, count, string_table_index_count);
        finalize_batched_index_softmax_delta(workspace.string_table_index_logits,
            workspace.string_table_index_delta, workspace.softmax_temp,
            expected_string_table_indices, workspace.string_table_index_active_examples,
            workspace.expected_value_counts, workspace.numeric_distance_costs,
            workspace.losses, count, string_table_index_count, loss_gradient_scale,
            REFERENCE_LOSS_WEIGHT, NUMERIC_DISTANCE_LOSS_WEIGHT);
    }

    void OutputLayer::backward_batched_accumulate(
        const fixed_size_matrix<float, BatchIndex, TokenID>& delta,
        const fixed_size_matrix<float, BatchIndex, PositionIndex>& string_table_index_delta,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
        BatchIndex batch_size,
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& dh_last,
        OutputLayerGradientAccumulator& accumulator,
        PositionIndex string_table_index_count
    )
    {
        accumulate_batched_output_layer_dh_last(delta, dh_last, W_lm_head, static_cast<int>(batch_size));
        accumulate_batched_output_layer_dW(delta, h_last, accumulator.dW_lm_head, static_cast<int>(batch_size));
        if (string_table_index_count != PositionIndex::START)
        {
            accumulate_batched_output_layer_string_table_index_dh_last(
                string_table_index_delta, dh_last, W_string_table_index_head,
                static_cast<int>(batch_size), string_table_index_count);
            accumulate_batched_output_layer_string_table_index_dW(
                string_table_index_delta, h_last, accumulator.dW_string_table_index_head,
                static_cast<int>(batch_size), string_table_index_count);
            if (static_cast<size_t>(string_table_index_count) > static_cast<size_t>(accumulator.string_table_index_count))
                accumulator.string_table_index_count = string_table_index_count;
        }
        accumulator.touched = true;
    }

    void OutputLayer::apply_accumulated_update(OutputLayerGradientAccumulator& accumulator, float learning_rate, float bias_correction1, float bias_correction2)
    {
        if (!accumulator.touched)
            return;
        check_nan_finding_mode("apply_accumulated_update:start");
        auto& diagnostics = optimizer_diagnostics_buffer();
        prepare_optimizer_gradient_clip(diagnostics);
        accumulate_optimizer_gradient_norm(accumulator.dW_lm_head, diagnostics);
        if (accumulator.string_table_index_count != PositionIndex::START)
            accumulate_optimizer_gradient_norm(accumulator.dW_string_table_index_head, accumulator.string_table_index_count, diagnostics);
        finalize_optimizer_gradient_clip(diagnostics);
        update_output_layer_weights_from_gradient(accumulator.dW_lm_head, W_lm_head, V_lm_head, S_lm_head, learning_rate, bias_correction1, bias_correction2,
            diagnostics, optimizer_diagnostics_enabled() ? 1 : 0);
        if (accumulator.string_table_index_count != PositionIndex::START)
        {
            update_output_layer_weights_from_gradient(accumulator.dW_string_table_index_head, W_string_table_index_head,
                V_string_table_index_head, S_string_table_index_head, learning_rate, bias_correction1, bias_correction2,
                diagnostics, optimizer_diagnostics_enabled() ? 1 : 0, accumulator.string_table_index_count);
        }
        check_nan_finding_mode("apply_accumulated_update:end");
    }

    std::vector<OutputToken> OutputLayer::get_top_k_by_logit(size_t k) const
    {
        assert(k != 0);

        std::vector<OutputToken> top_k;
        for (const auto i : enum_iterator1D<TokenID>())
        {
            const float logit = m_inputs_cpu[i];
            if (top_k.size() < k)
            {
                top_k.push_back({i, logit});
                std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                    return a.activation > b.activation;
                });
            }
            else if (logit >= top_k.back().activation)
            {
                top_k.back() = {i, logit};
                std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                    return a.activation > b.activation;
                });
            }
        }
        return top_k;
    }

    std::vector<OutputStringTableIndex> OutputLayer::get_top_k_string_table_indices_by_logit(size_t k) const
    {
        assert(k != 0);

        std::vector<OutputStringTableIndex> top_k;
        for (const auto i : enum_iterator1D<PositionIndex>())
        {
            const float logit = m_string_table_index_inputs_cpu[i];
            if (top_k.size() < k)
            {
                top_k.push_back({i, logit});
                std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                    return a.activation > b.activation;
                });
            }
            else if (logit >= top_k.back().activation)
            {
                top_k.back() = {i, logit};
                std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                    return a.activation > b.activation;
                });
            }
        }
        return top_k;
    }


    // Compute softmax deltas (with label smoothing) for backprop and return the
    // cross-entropy loss -log(softmax[target]).
    float OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        return compute_score(m_inputs, m_inputs_cpu, score, expected_output_token, NO_STRING_TABLE_INDEX);
    }

    float OutputLayer::compute_score(
        Score& score,
        const TokenID expected_output_token,
        size_t expected_string_table_index)
    {
        return compute_score(m_inputs, m_inputs_cpu, score, expected_output_token, expected_string_table_index);
    }

    float OutputLayer::compute_score(
        const fixed_size_vector<float, TokenID>& inputs,
        const cpu_fixed_vector<float, TokenID>& inputs_cpu,
        Score& score,
        const TokenID expected_output_token
    )
    {
        return compute_score(inputs, inputs_cpu, score, expected_output_token, NO_STRING_TABLE_INDEX);
    }

    float OutputLayer::compute_score(
        const fixed_size_vector<float, TokenID>& inputs,
        const cpu_fixed_vector<float, TokenID>& inputs_cpu,
        Score& score,
        const TokenID expected_output_token,
        size_t expected_string_table_index
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        initialize_softmax_temp_values(score.temp_values);
        check_output_logits_nan_finding_mode(inputs, "compute_score:start");
        reduce_logits_max_to_temp(inputs, score.temp_values);
        compute_exp_and_accumulate_sum(inputs, score.values, score.temp_values);
        const float token_weight = is_identifier_category_token(expected_output_token)
            ? 1.0f + CATEGORY_LOSS_BONUS : 1.0f;
        finalize_softmax_delta(score.values, score.temp_values, expected_output_token, token_weight);
        score.temp_values.copy_to_cpu(queue, score.temp_values_cpu);

        const float max_val = score.temp_values_cpu[TempStorage::START];
        const float sum_exp = score.temp_values_cpu[TempStorage::ONE];
        const float expected_logit = inputs_cpu[expected_output_token];
        const float log_prob = expected_logit - max_val - std::log(sum_exp);
        float loss = token_weight * -log_prob;
        score.string_table_index_prediction_active = false;
        score.string_table_index_values.zero(queue);
        if (token_string_category(expected_output_token) != TokenStringCategory::None &&
            expected_string_table_index != NO_STRING_TABLE_INDEX)
        {
            assert(expected_string_table_index < static_cast<size_t>(PositionIndex::MAX));
            score.string_table_index_prediction_active = true;
            initialize_softmax_temp_values(score.temp_values);
            reduce_index_logits_max_to_temp(m_string_table_index_inputs, score.temp_values);
            compute_index_exp_and_accumulate_sum(
                m_string_table_index_inputs, score.string_table_index_values, score.temp_values);
            finalize_index_softmax_delta(
                score.string_table_index_values, score.temp_values,
                static_cast<PositionIndex>(expected_string_table_index), REFERENCE_LOSS_WEIGHT);
            score.temp_values.copy_to_cpu(queue, score.temp_values_cpu);
            const float index_max_val = score.temp_values_cpu[TempStorage::START];
            const float index_sum_exp = score.temp_values_cpu[TempStorage::ONE];
            const auto expected_index = static_cast<PositionIndex>(expected_string_table_index);
            const float expected_index_logit = m_string_table_index_inputs_cpu[expected_index];
            const float index_log_prob = expected_index_logit - index_max_val - std::log(index_sum_exp);
            const float index_loss = -index_log_prob;
            const float max_reasonable_index_loss =
                compute_max_reasonable_index_loss(index_max_val, expected_index_logit);
            if (!std::isfinite(expected_index_logit) || !std::isfinite(index_max_val) ||
                !std::isfinite(index_sum_exp) || !std::isfinite(index_log_prob) ||
                index_sum_exp <= 0.0f || index_log_prob > 1e-4f ||
                index_loss > max_reasonable_index_loss)
            {
                std::fprintf(
                    stderr,
                    "compute_score invalid string-table-index softmax state: target=%zu expected_logit=%g "
                    "max_val=%g sum_exp=%g log_prob=%g loss=%g max_reasonable_loss=%g\n",
                    expected_string_table_index,
                    static_cast<double>(expected_index_logit),
                    static_cast<double>(index_max_val),
                    static_cast<double>(index_sum_exp),
                    static_cast<double>(index_log_prob),
                    static_cast<double>(index_loss),
                    static_cast<double>(max_reasonable_index_loss)
                );
                std::abort();
            }
            loss += REFERENCE_LOSS_WEIGHT * index_loss;
        }
        const float max_reasonable_loss =
            token_weight * compute_max_reasonable_loss(max_val, expected_logit) +
            (score.string_table_index_prediction_active
                ? REFERENCE_LOSS_WEIGHT * compute_max_reasonable_index_loss(
                    score.temp_values_cpu[TempStorage::START],
                    m_string_table_index_inputs_cpu[static_cast<PositionIndex>(expected_string_table_index)])
                : 0.0f);
        if (!std::isfinite(expected_logit) || !std::isfinite(max_val) || !std::isfinite(sum_exp) ||
            !std::isfinite(log_prob) || sum_exp <= 0.0f || log_prob > 1e-4f ||
            loss > max_reasonable_loss)
        {
            std::fprintf(
                stderr,
                "compute_score invalid softmax state: target=%zu expected_logit=%g max_val=%g sum_exp=%g "
                "log_prob=%g loss=%g max_reasonable_loss=%g\n",
                static_cast<size_t>(expected_output_token),
                static_cast<double>(expected_logit),
                static_cast<double>(max_val),
                static_cast<double>(sum_exp),
                static_cast<double>(log_prob),
                static_cast<double>(loss),
                static_cast<double>(max_reasonable_loss)
            );
            std::abort();
        }
        return loss;
    }
} // namespace rllm
