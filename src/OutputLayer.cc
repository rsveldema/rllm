#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>

namespace rllm
{
    void OutputLayer::set_random_weights_and_connections_for_output_layer(Corpus& corpus)
    {
        // setup the output layer itself. It has no connections to other neurons.
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            m_trigger_values[i] = get_random_value();
        }
    }

    void OutputLayer::update_output_weights(const template_token_vector<float, TokenID>& delta, float learning_rate)
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            // Adjust trigger: lower when delta > 0 (fire more), raise when delta < 0.
            m_trigger_values[i] = std::clamp(m_trigger_values[i] - learning_rate * delta[i], 0.0f, 1.0f);
        }
    }

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
