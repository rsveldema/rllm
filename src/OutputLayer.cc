#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <parallel.hpp>

#include <enum_iterator2D.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <omp.h>

namespace rllm
{
    using uint = unsigned int;

    static inline float finite_or_zero(float value)
    {
        return std::isfinite(value) ? value : 0.0f;
    }

    static inline float finite_clamp(float value, float lo, float hi)
    {
        if (!std::isfinite(value))
            return 0.0f;
        return math::clamp(value, lo, hi);
    }

    OutputLayer::OutputLayer()
    {
        m_inputs.set_size(TokenID::MAX);
    }

    void output_layer_forward_from_hidden_impl(
        // OFFLOAD_PARAMETERS(h_last, W, inputs)
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        fixed_size_vector<float, TokenID>& inputs
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, v, enum_iterator1D<TokenID>(), (h_last, W, inputs))
        float sum = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            sum += (h_last[d] * static_cast<float>(W[v, d]));
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
            sum += (delta[v] * W[v, d]);
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
        const float g = math::clamp((delta[v] * h_last[d]), -OutputLayer::GRAD_CLIP, OutputLayer::GRAD_CLIP);
        V[v, d] = math::clamp(
            ((OutputLayer::MOMENTUM_BETA * V[v, d]) + (learning_rate * g)), -OutputLayer::VEL_CLIP, OutputLayer::VEL_CLIP
        );
        W[v, d] = math::clamp((W[v, d] + V[v, d]), -OutputLayer::WEIGHT_CLAMP, OutputLayer::WEIGHT_CLAMP);
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
            float exp_value = exp((inputs[i] - max_val));
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
        const float sum_exp = temp_values[TempStorage::ONE];
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
        W_lm_head.copy_from_cpu(cpu_tmp);
        V_lm_head.zero();
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<float, EmbeddingDimension>& h_last)
    {
        output_layer_forward_from_hidden_impl(h_last, W_lm_head, m_inputs);
        m_inputs.copy_to_cpu(m_inputs_cpu);
    }

    // Accumulates dL/dh_last[D] and updates W_lm_head.
    void OutputLayer::backward_and_update(
        const fixed_size_vector<float, TokenID>& delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        float learning_rate
    )
    {
        accumulate_output_layer_dh_last(delta, dh_last, W_lm_head);
        update_output_layer_weights(delta, h_last, W_lm_head, V_lm_head, learning_rate);
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
    // cross-entropy loss -log(softmax[target]) in a single pass over the logits.
    float OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        float max_val = -std::numeric_limits<float>::infinity();
        for (const auto token : enum_iterator1D<TokenID>()) {
            max_val = math::max(max_val, m_inputs_cpu[token]);
        }

        score.temp_values_cpu[TempStorage::START] = max_val;
        score.temp_values_cpu[TempStorage::ONE] = 0.0f;
        score.temp_values.copy_from_cpu(score.temp_values_cpu);
        compute_exp_and_accumulate_sum(m_inputs, score.values, score.temp_values);
        finalize_softmax_delta(score.values, score.temp_values, expected_output_token);
        score.temp_values.copy_to_cpu(score.temp_values_cpu);

        const float sum_exp = score.temp_values_cpu[TempStorage::ONE];
        const float expected_logit = m_inputs_cpu[expected_output_token];
        assert(! std::isnan(expected_logit));
        assert(! std::isnan(max_val));
        assert(! std::isnan(sum_exp));
        const float log_prob = expected_logit - max_val - std::log(sum_exp);
        assert(! std::isnan(log_prob));
        return -log_prob;
    }
} // namespace rllm
