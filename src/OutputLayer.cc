#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>
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

    static void output_layer_forward_from_hidden_float_impl(
        // OFFLOAD_PARAMETERS(h_last, W_float, inputs)
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float, TokenID, EmbeddingDimension>& W_float,
        fixed_size_vector<float, TokenID>& inputs
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(v, enum_iterator1D<TokenID>(), (h_last, W_float, inputs))
        float sum = 0.f;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            sum += (h_last[d] * W_float[v, d]);
        inputs[v] = sum;
        ENDFOR
    }

    void output_layer_forward_from_hidden_impl(
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W,
        fixed_size_vector<float, TokenID>& inputs
    )
    {
        fixed_size_matrix<float, TokenID, EmbeddingDimension> W_float;
        for (const auto v : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                W_float[v, d] = finite_or_zero(static_cast<float>(W[v, d]));
        output_layer_forward_from_hidden_float_impl(h_last, W_float, inputs);
    }

    static void accumulate_output_layer_dh_last(
        // OFFLOAD_PARAMETERS(delta, dh_last, W)
        const fixed_size_vector<float, TokenID>& delta,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& W
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(d, enum_iterator1D<EmbeddingDimension>(), (delta, dh_last, W))
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
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(v, d, grid, (delta, h_last, W, V, learning_rate))
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
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<TempStorage>(), (temp_values))
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

        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<TokenID>(), (inputs, temp_values))
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

        temp_values[TempStorage::ONE] = 0;

        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<TokenID>(), (inputs, values, temp_values))
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
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator1D<TokenID>(), (values, temp_values, expected_output_token))
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
        for (const auto v : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                W_lm_head[v, d] = get_random_value(-scale, scale);
        W_lm_head.copy_to_offload_buffer();
        V_lm_head.zero();
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<float, EmbeddingDimension>& h_last)
    {
        output_layer_forward_from_hidden_impl(h_last, W_lm_head, m_inputs);
    }

    // Accumulates dL/dh_last[D] and updates W_lm_head.
    void OutputLayer::backward_and_update(
        const fixed_size_vector<float, TokenID>& delta,
        const fixed_size_vector<float, EmbeddingDimension>& h_last,
        fixed_size_vector<float, EmbeddingDimension>& dh_last,
        float learning_rate
    )
    {
#if ! USE_VULKAN_OFFLOAD
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
        {
            float sum = finite_or_zero(dh_last[d]);
            for (const auto v : enum_iterator1D<TokenID>())
                sum += (finite_or_zero(delta[v]) * finite_or_zero(static_cast<float>(W_lm_head[v, d])));
            dh_last[d] = finite_or_zero(sum);
        }

        for (const auto v : enum_iterator1D<TokenID>())
        {
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
            {
                const float g = finite_clamp(
                    (finite_or_zero(delta[v]) * finite_or_zero(h_last[d])),
                    -OutputLayer::GRAD_CLIP,
                    OutputLayer::GRAD_CLIP
                );
                const float velocity = finite_clamp(
                    ((OutputLayer::MOMENTUM_BETA * finite_or_zero(V_lm_head[v, d])) + (learning_rate * g)),
                    -OutputLayer::VEL_CLIP,
                    OutputLayer::VEL_CLIP
                );
                V_lm_head[v, d] = velocity;
                const float weight = finite_clamp(
                    (finite_or_zero(static_cast<float>(W_lm_head[v, d])) + velocity),
                    -OutputLayer::WEIGHT_CLAMP,
                    OutputLayer::WEIGHT_CLAMP
                );
                W_lm_head[v, d] = static_cast<float16>(weight);
            }
        }
#else
        accumulate_output_layer_dh_last(delta, dh_last, W_lm_head);
        update_output_layer_weights(delta, h_last, W_lm_head, V_lm_head, learning_rate);
#endif
    }

    std::vector<OutputToken> OutputLayer::get_top_k_by_logit(size_t k) const
    {
        assert(k != 0);

        std::vector<OutputToken> top_k;
        for (const auto i : enum_iterator1D<TokenID>())
        {
            const float logit = m_inputs[i];
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
#if !USE_VULKAN_OFFLOAD
        float max_val = -std::numeric_limits<float>::infinity();
        for (const auto token : enum_iterator1D<TokenID>()) {
            const auto v = finite_or_zero(m_inputs.get_offload_synced(token));
            assert(! std::isnan(v));
            max_val = math::max(max_val, v);
        }
        if (!std::isfinite(max_val))
            max_val = 0.0f;

        score.temp_values[TempStorage::START] = max_val;
        score.temp_values[TempStorage::ONE] = 0.0f;
        for (const auto token : enum_iterator1D<TokenID>())
        {
            const float exp_arg = finite_clamp((finite_or_zero(m_inputs[token]) - max_val), -80.0f, 80.0f);
            const float exp_value = finite_or_zero(std::exp(exp_arg));
            assert(! std::isnan((exp_value)));
            score.values[token] = exp_value;
            score.temp_values[TempStorage::ONE] += exp_value;
        }

        float sum_exp = score.temp_values[TempStorage::ONE];
        if (!std::isfinite(sum_exp) || sum_exp <= 0.0f)
            sum_exp = static_cast<float>(TokenID::MAX);
        for (const auto token : enum_iterator1D<TokenID>())
        {
            float delta = OutputLayer::smooth - score.values[token] / sum_exp;
            if (token == expected_output_token)
                delta += (1.0f - OutputLayer::LABEL_SMOOTHING);
            score.values[token] = delta;
        }

        const float expected_logit = finite_or_zero(m_inputs.get_offload_synced(expected_output_token));
#else
        float max_val = -std::numeric_limits<float>::infinity();
        for (const auto token : enum_iterator1D<TokenID>()) {
            max_val = math::max(max_val, m_inputs[token]);
        }

        score.temp_values[TempStorage::START] = max_val;
        score.temp_values[TempStorage::ONE] = 0.0f;
        compute_exp_and_accumulate_sum(m_inputs, score.values, score.temp_values);
        finalize_softmax_delta(score.values, score.temp_values, expected_output_token);

        const float sum_exp = score.temp_values[TempStorage::ONE];
        const float expected_logit = m_inputs[expected_output_token];
#endif
        assert(! std::isnan(expected_logit));
        assert(! std::isnan(max_val));
        assert(! std::isnan(sum_exp));
        const float log_prob = expected_logit - max_val - std::log(sum_exp);
        assert(! std::isnan(log_prob));
        return -log_prob;
    }
} // namespace rllm
