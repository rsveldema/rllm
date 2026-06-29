#include <OutputLayer.hpp>
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
    static constexpr float NAN_FINDING_LOGIT_ABS_BOUND = 1000.0f;
    static constexpr float MAX_REASONABLE_CROSS_ENTROPY_LOSS = 1000.0f;

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

    void OutputLayer::check_nan_finding_mode(const char* phase)
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_nan_scan_flag(queue);
        scan_output_layer_weight_matrix(W_lm_head, flag, -WEIGHT_CLAMP, WEIGHT_CLAMP);
        scan_output_layer_velocity_matrix(V_lm_head, flag, -VEL_CLIP, VEL_CLIP);
        if (nan_scan_failed(queue, flag))
        {
            std::fprintf(stderr, "NAN_FINDING_MODE: OutputLayer weights/velocities invalid during %s\n", phase);
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
        check_output_vector_scan_failed("logits", phase, flag, queue, NAN_FINDING_LOGIT_ABS_BOUND);
    }

    OutputLayer::OutputLayer()
    {
        m_inputs.set_size(TokenID::MAX);
        W_lm_head_cpu.zero();
        V_lm_head_cpu.zero();
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
        inputs[v] = sum;
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

    static void update_output_layer_weights(
        // OFFLOAD_PARAMETERS(delta, h_last, W, V, learning_rate)
        const fixed_size_vector<float, TokenID>& delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& V,
        float learning_rate
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, v, d, grid, (delta, h_last, W, V, learning_rate))
        const float raw_g = (delta[v] * h_last[d]);
        const float g = math::clamp(raw_g, -OutputLayer::GRAD_CLIP, OutputLayer::GRAD_CLIP);
        const float raw_v = ((OutputLayer::MOMENTUM_BETA * V[v, d]) + (learning_rate * g));
        V[v, d] = math::clamp(raw_v, -OutputLayer::VEL_CLIP, OutputLayer::VEL_CLIP);
        const float raw_w = (W[v, d] + V[v, d]);
        W[v, d] = math::clamp(raw_w, -OutputLayer::WEIGHT_CLAMP, OutputLayer::WEIGHT_CLAMP);
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

    [[maybe_unused]] static void finalize_softmax_delta(
        // OFFLOAD_PARAMETERS(values, temp_values, expected_output_token)
        fixed_size_vector<float, TokenID>& values,
        const fixed_size_vector<float, TempStorage>& temp_values,
        TokenID expected_output_token
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TokenID>(), (values, temp_values, expected_output_token))
        float sum_exp = temp_values[TempStorage::ONE];
        float delta = (OutputLayer::smooth - (values[i] / sum_exp));
        if (i == expected_output_token)
            delta += (1.0f - OutputLayer::LABEL_SMOOTHING);
        values[i] = delta;
        ENDFOR
    }

    OutputLayer::OutputLayer(const Corpus& corpus)
        : OutputLayer()
    {
        (void) corpus;
    }

    void OutputLayer::set_random_weights()
    {
        const int D = static_cast<int>(EmbeddingDimension::MAX);
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> cpu_tmp;
        for (const auto v : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                cpu_tmp.set(v, d, static_cast<float16>(get_random_value(-scale, scale)));
        {
            auto& queue = rllm::vulkan_runtime::get_queue(0);
            W_lm_head_cpu = cpu_tmp;
            V_lm_head_cpu.zero();
            W_lm_head.copy_from_cpu(queue, cpu_tmp);
            V_lm_head.zero(queue);
        }
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<float, EmbeddingDimension>& h_last,
              VulkanQueue& queue)
    {
        check_nan_finding_mode("forward:start");
        check_output_h_last_nan_finding_mode(h_last, "forward:start");
        output_layer_forward_from_hidden_impl(queue, h_last, W_lm_head, m_inputs);
        check_output_logits_nan_finding_mode(m_inputs, "forward:end");
        m_inputs.copy_to_cpu(queue, m_inputs_cpu);
        check_nan_finding_mode("forward:end");
    }

    // Accumulates dL/dh_last[D] and updates W_lm_head.
    void OutputLayer::backward_and_update(
        const fixed_size_vector<float, TokenID>& delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        float learning_rate
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        check_nan_finding_mode("backward:start");
        accumulate_output_layer_dh_last(delta, dh_last, W_lm_head);
        update_output_layer_weights(delta, h_last, W_lm_head, V_lm_head, learning_rate);
        W_lm_head.copy_to_cpu(queue, W_lm_head_cpu);
        V_lm_head.copy_to_cpu(queue, V_lm_head_cpu);
        check_nan_finding_mode("backward:end");
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


    // Compute softmax deltas (with label smoothing) for backprop and return the
    // cross-entropy loss -log(softmax[target]).
    float OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        initialize_softmax_temp_values(score.temp_values);
        check_output_logits_nan_finding_mode(m_inputs, "compute_score:start");
        reduce_logits_max_to_temp(m_inputs, score.temp_values);
        compute_exp_and_accumulate_sum(m_inputs, score.values, score.temp_values);
        finalize_softmax_delta(score.values, score.temp_values, expected_output_token);
        score.temp_values.copy_to_cpu(queue, score.temp_values_cpu);

        const float max_val = score.temp_values_cpu[TempStorage::START];
        const float sum_exp = score.temp_values_cpu[TempStorage::ONE];
        const float expected_logit = m_inputs_cpu[expected_output_token];
        const float log_prob = expected_logit - max_val - std::log(sum_exp);
        const float loss = -log_prob;
        if (!std::isfinite(expected_logit) || !std::isfinite(max_val) || !std::isfinite(sum_exp) ||
            !std::isfinite(log_prob) || sum_exp <= 0.0f || log_prob > 1e-4f ||
            loss > MAX_REASONABLE_CROSS_ENTROPY_LOSS)
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
                static_cast<double>(MAX_REASONABLE_CROSS_ENTROPY_LOSS)
            );
            std::abort();
        }
        return loss;
    }
} // namespace rllm
