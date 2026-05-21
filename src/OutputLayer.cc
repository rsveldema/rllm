#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>

namespace rllm
{
    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        score.values.fill(0.0f);
        score.values[expected_output_token] = 1.0f;
    }

    void OutputLayer::compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            deltas[i] = score.values[i] - m_inputs[i];
        }
    }

} // namespace rllm
