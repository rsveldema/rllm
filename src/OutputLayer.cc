#include <OutputLayer.hpp>

#include <algorithm>

namespace rllm
{

    namespace
    {
        static float get_random_value()
        {
            return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    } // namespace

    void OutputLayer::set_random_weights_and_connections_for_output_layer(Corpus& corpus)
    {
        // setup the output layer itself. It has no connections to other neurons.
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            m_trigger_values[i] = get_random_value();
            m_weights[i] = get_random_value();
        }
    }

    void OutputLayer::update_output_weights(const template_token_vector<float, TokenID>& delta, float learning_rate)
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            m_weights[i] = std::clamp(m_weights[i] + learning_rate * delta[i] * m_inputs[i], 0.0f, 1.0f);
            // Adjust trigger: lower when delta > 0 (fire more), raise when delta < 0.
            m_trigger_values[i] = std::clamp(m_trigger_values[i] - learning_rate * delta[i], 0.0f, 1.0f);
        }
    }

    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            score.values[i] = m_inputs[i];
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
