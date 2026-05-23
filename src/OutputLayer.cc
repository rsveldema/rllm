#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cmath>

namespace rllm
{
    //root-mean-square normalize the accumulated logits so the softmax sees reasonable magnitudes
    //regardless of the intermediate-layer fan-in.
    void OutputLayer::rms_normalize_inputs()
    {
        constexpr float eps = 1e-6f;
        const float n = static_cast<float>(m_corpus.number_of_token_types());
        const auto active_end = static_cast<TokenID>(m_corpus.number_of_token_types());
        float sum_sq = 0.0f;
        for (const auto i : enum_iterator<TokenID>(active_end))
            sum_sq += m_inputs[i] * m_inputs[i];
        const float rms = std::sqrt(sum_sq / n + eps);
        for (const auto i : enum_iterator<TokenID>(active_end))
            m_inputs[i] /= rms;
    }

    // Compute numerically-stable softmax of m_inputs and store the backprop delta
    // in score.values using the convention delta = one_hot - softmax (positive means
    // "this token should fire more").  The backprop weight update is weight += lr*delta*act,
    // so the delta must have the sign of (target - actual), NOT (actual - target).
    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        const auto active_end = static_cast<TokenID>(m_corpus.number_of_token_types());
        // Numerically stable softmax: subtract max before exponentiation.
        float max_val = m_inputs[TokenID::START];
        for (const auto i : enum_iterator<TokenID>(active_end))
            max_val = std::max(max_val, m_inputs[i]);

        float sum_exp = 0.0f;
        for (const auto i : enum_iterator<TokenID>(active_end))
        {
            score.values[i] = std::exp(m_inputs[i] - max_val);
            sum_exp += score.values[i];
        }

        // delta = one_hot - softmax: target slot gets (1 - p) > 0, others get -p < 0.
        for (const auto i : enum_iterator<TokenID>(active_end))
            score.values[i] = -score.values[i] / sum_exp;

        score.values[expected_output_token] += 1.0f;
    }

    void OutputLayer::compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const
    {
        deltas = score.values;
    }

} // namespace rllm
