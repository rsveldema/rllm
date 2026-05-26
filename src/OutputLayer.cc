#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

namespace rllm
{
    static constexpr float MOMENTUM_BETA = 0.9f;
    static constexpr float GRAD_CLIP = 1.0f;
    static constexpr float VEL_CLIP = 0.1f;
    static constexpr float WEIGHT_CLAMP = 2.0f;

    OutputLayer::OutputLayer(const Corpus& corpus)
        : m_corpus(corpus)
    {}

    void OutputLayer::set_random_weights()
    {
        const int D = static_cast<int>(EmbeddingDimension::MAX);
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (const auto v : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                W_lm_head[v, d] = get_random_value(-scale, scale);
        V_lm_head.fill(RLMM_ZERO);
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last)
    {
        m_inputs.fill(RLMM_ZERO);
        for (const auto v : enum_iterator<TokenID>())
        {
            float sum = 0.f;
            for (const auto d : enum_iterator<EmbeddingDimension>())
                sum += h_last[d] * W_lm_head[v, d];
            m_inputs[v] = sum;
        }
    }

    // Returns dL/dh_last[D] and updates W_lm_head.
    fixed_size_vector<rlmm_float, EmbeddingDimension> OutputLayer::backward_and_update(
        const fixed_size_vector<rlmm_float, TokenID>& delta,
        const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
        float learning_rate
    )
    {
        fixed_size_vector<rlmm_float, EmbeddingDimension> dh;

        for (const auto v : enum_iterator<TokenID>())
        {
            const float dv = delta[v];
            for (const auto d : enum_iterator<EmbeddingDimension>())
            {
                dh[d] += dv * W_lm_head[v, d];
                const float g = math::clamp(dv * h_last[d], -GRAD_CLIP, GRAD_CLIP);
                V_lm_head[v, d] = math::clamp(
                    MOMENTUM_BETA * V_lm_head[v, d] + learning_rate * g,
                    -VEL_CLIP,
                    VEL_CLIP
                );
                W_lm_head[v, d] = math::clamp(
                    W_lm_head[v, d] + V_lm_head[v, d],
                    -WEIGHT_CLAMP,
                    WEIGHT_CLAMP
                );
            }
        }
        return dh;
    }

    // ── scoring (unchanged from previous architecture) ─────────────────────────

    void OutputLayer::rms_normalize_inputs()
    {
        constexpr float eps = 1e-6f;
        const float n = static_cast<float>(TokenID::MAX);
        float sum_sq = 0.0f;
        for (const auto i : enum_iterator<TokenID>())
            sum_sq += m_inputs[i] * m_inputs[i];
        const float rms = std::sqrt(sum_sq / n + eps);
        for (const auto i : enum_iterator<TokenID>())
            m_inputs[i] /= rms;
    }

    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        float max_val = m_inputs[TokenID::START];
        for (const auto i : enum_iterator<TokenID>())
            max_val = math::max(max_val, m_inputs[i]);

        float sum_exp = 0.0f;
        for (const auto i : enum_iterator<TokenID>())
        {
            score.values[i] = std::exp(m_inputs[i] - max_val);
            sum_exp += score.values[i];
        }

        for (const auto i : enum_iterator<TokenID>())
            score.values[i] = -score.values[i] / sum_exp;

        score.values[expected_output_token] += RLMM_ONE;
    }

    void OutputLayer::compute_deltas(const Score& score, fixed_size_vector<rlmm_float, TokenID>& deltas) const
    {
        deltas = score.values;
    }

} // namespace rllm
