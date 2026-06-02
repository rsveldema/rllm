#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <enum_iterator2D.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

namespace rllm
{
    OutputLayer::OutputLayer()
    {
        m_inputs.set_size(TokenID::MAX);
    }

    static void output_layer_forward_from_hidden_impl(
        // OFFLOAD_PARAMETERS(h_last, W, inputs)
        const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
        const fixed_size_matrix<rlmm_float_small, TokenID, EmbeddingDimension>& W,
        fixed_size_vector<rlmm_float, TokenID>& inputs
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(v, enum_iterator<TokenID>(), (h_last, W, inputs))
        float sum = 0.f;
        for (const auto d : enum_iterator<EmbeddingDimension>())
            sum += h_last[d] * W[v, d];
        inputs[v] = sum;
        ENDFOR
    }

    static void accumulate_output_layer_dh_last(
        // OFFLOAD_PARAMETERS(delta, dh_last, W)
        const fixed_size_vector<rlmm_float, TokenID>& delta,
        fixed_size_vector<rlmm_float, EmbeddingDimension>& dh_last,
        const fixed_size_matrix<rlmm_float_small, TokenID, EmbeddingDimension>& W
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(d, enum_iterator<EmbeddingDimension>(), (delta, dh_last, W))
        float sum = dh_last[d];
        for (const auto v : enum_iterator<TokenID>())
            sum += delta[v] * W[v, d];
        dh_last[d] = sum;
        ENDFOR
    }

    static void update_output_layer_weights(
        // OFFLOAD_PARAMETERS(delta, h_last, W, V, learning_rate)
        const fixed_size_vector<rlmm_float, TokenID>& delta,
        const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
        fixed_size_matrix<rlmm_float_small, TokenID, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, TokenID, EmbeddingDimension>& V,
        float learning_rate
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(v, d, grid, (delta, h_last, W, V, learning_rate))
        const float g = math::clamp(delta[v] * h_last[d], -OutputLayer::GRAD_CLIP, OutputLayer::GRAD_CLIP);
        V[v, d] = math::clamp(
            OutputLayer::MOMENTUM_BETA * V[v, d] + learning_rate * g,
            -OutputLayer::VEL_CLIP,
            OutputLayer::VEL_CLIP
        );
        W[v, d] = math::clamp(
            W[v, d] + V[v, d],
            -OutputLayer::WEIGHT_CLAMP,
            OutputLayer::WEIGHT_CLAMP
        );
        ENDFOR
    }

    OutputLayer::OutputLayer(const Corpus& corpus)
        : OutputLayer()
    {
        (void)corpus;
    }

    void OutputLayer::set_random_weights()
    {
        const int D = static_cast<int>(EmbeddingDimension::MAX);
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (const auto v : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                W_lm_head[v, d] = get_random_value(-scale, scale);
        W_lm_head.copy_to_offload_buffer();
        V_lm_head.zero();
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last)
    {
        output_layer_forward_from_hidden_impl(h_last, W_lm_head, m_inputs);
    }

    // Accumulates dL/dh_last[D] and updates W_lm_head.
    void OutputLayer::backward_and_update(
        const fixed_size_vector<rlmm_float, TokenID>& delta,
        const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
        fixed_size_vector<rlmm_float, EmbeddingDimension>& dh_last,
        float learning_rate
    )
    {
        accumulate_output_layer_dh_last(delta, dh_last, W_lm_head);
        update_output_layer_weights(delta, h_last, W_lm_head, V_lm_head, learning_rate);
    }

    std::vector<OutputToken> OutputLayer::get_top_k_by_logit(size_t k) const
    {
        assert (k != 0);

        std::vector<OutputToken> top_k;
        for (const auto i : enum_iterator<TokenID>())
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
        // Label smoothing (ε=0.1): instead of a one-hot target, each non-target
        // token gets a small positive gradient of ε/V. This prevents the model
        // from driving non-target logits to -∞ and collapsing to one token.
        static constexpr float LABEL_SMOOTHING = 0.1f;
        const float smooth = LABEL_SMOOTHING / static_cast<float>(static_cast<int>(TokenID::MAX));

        float max_val = m_inputs[TokenID::START];
        for (const auto i : enum_iterator<TokenID>())
            max_val = math::max(max_val, m_inputs[i]);

        float sum_exp = 0.0f;
        for (const auto i : enum_iterator<TokenID>())
        {
            score.values[i] = std::exp(m_inputs[i] - max_val);
            sum_exp += score.values[i];
        }

        const float log_prob = m_inputs[expected_output_token] - max_val - std::log(sum_exp);

        // delta[i] = smooth - softmax[i]  (small positive floor for all non-targets)
        for (const auto i : enum_iterator<TokenID>())
            score.values[i] = smooth - score.values[i] / sum_exp;

        // delta[expected] += (1 - LABEL_SMOOTHING)
        score.values[expected_output_token] += (RLMM_ONE - LABEL_SMOOTHING);

        return -log_prob;
    }
} // namespace rllm
