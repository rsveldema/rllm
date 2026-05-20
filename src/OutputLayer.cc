#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>

namespace rllm
{
    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            // One-hot target: expected token should fire, all others should not.
            score.values[i] = (i == expected_output_token) ? 1.0f : 0.0f;
        }
    }

    void OutputLayer::compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            deltas[i] = score.values[i] - m_inputs[i];
        }
    }

} // namespace rllm
