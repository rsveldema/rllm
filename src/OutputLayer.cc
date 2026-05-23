#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cmath>

namespace rllm
{
    // Compute numerically-stable softmax of m_inputs and store the cross-entropy
    // gradient (softmax - one_hot) into score.values, ready for backprop.
    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        // Numerically stable softmax: subtract max before exponentiation.
        float max_val = m_inputs[TokenID::START];
        for (const auto i : enum_iterator<TokenID>())
            max_val = std::max(max_val, m_inputs[i]);

        float sum_exp = 0.0f;
        for (const auto i : enum_iterator<TokenID>())
        {
            score.values[i] = std::exp(m_inputs[i] - max_val);
            sum_exp += score.values[i];
        }

        // Normalize to get probabilities, then subtract one-hot target to get gradient.
        for (const auto i : enum_iterator<TokenID>())
            score.values[i] /= sum_exp;

        score.values[expected_output_token] -= 1.0f;
    }

    void OutputLayer::compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const
    {
        deltas = score.values;
    }

} // namespace rllm
