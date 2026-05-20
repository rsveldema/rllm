#include <RLLM.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <print>

namespace rllm
{
    // Helper functions

    template <typename X, typename Y>
    static std::pair<X, Y> get_random_value_centered_around(X x, Y y, int range = 10)
    {
        int k1 = rand() % (2 * range + 1) - range;
        int k2 = rand() % (2 * range + 1) - range;

        if ((static_cast<int>(x) + k1) < 0)
        {
            k1 = 0;
        }

        if ((static_cast<int>(y) + k2) < 0)
        {
            k2 = 0;
        }

        if ((static_cast<int>(x) + k1) >= static_cast<int>(X::MAX))
        {
            k1 = 0;
        }

        if (static_cast<int>(y) + k2 >= static_cast<int>(Y::MAX))
        {
            k2 = 0;
        }

        return std::make_pair(static_cast<X>(static_cast<int>(x) + k1), static_cast<Y>(static_cast<int>(y) + k2));
    }

    static size_t clip_max(size_t value, size_t max)
    {
        if (value > max)
            return max;
        return value;
    }

    static float get_random_value()
    {
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Layers

    void InputLayer::set_input_layer(const InputLine& input)
    {
        m_inputs.fill(0.0f);
        for (size_t i = 0; i < input.size(); ++i)
        {
            if (i >= static_cast<size_t>(PositionIndex::MAX))
            {
                std::println(
                    "Warning: input line has more tokens than the max position index. Ignoring tokens beyond position "
                    "{}.",
                    static_cast<size_t>(PositionIndex::MAX)
                );
                break; // ignore tokens beyond the max position index
            }

            const auto token_id = input[i];
            m_inputs.set(token_id, static_cast<PositionIndex>(i), 1.0f);
        }
    }

    void InputLayer::set_random_weights_and_connections()
    {
        for (auto i = TokenID::START; i < TokenID::MAX; inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value());
                m_weights.set(i, pos, get_random_value());
                auto [target, pos_index] = get_random_value_centered_around(i, pos);
                m_connections.set(i, pos, std::make_pair(static_cast<IntermediateLayerIndex>(target), pos_index));
            }
        }
    }

    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value());
                m_weights.set(i, pos, get_random_value());
                auto target = get_random_value_centered_around(i, pos);
                m_connections.set(i, pos, target);
            }
        }
    }

    void IntermediateLayer::set_random_weights_and_connections_to_output_layer(Corpus& corpus)
    {
        // setup the layer JUST before the output layer.
        // It needs to have connections to the output layer that are distributed
        // across the tokens in the corpus.
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value());
                m_weights.set(i, pos, get_random_value());
                m_connections.set(
                    i,
                    pos,
                    std::make_pair(
                        static_cast<IntermediateLayerIndex>(static_cast<int>(i) % corpus.size()), PositionIndex::START
                    )
                );
            }
        }
    }

    void OutputLayer::set_random_weights_and_connections_for_output_layer(Corpus& corpus)
    {
        // setup the output layer itself. It has no connections to other neurons.
        for (TokenID i = TokenID::START; i < TokenID::MAX; inc(i))
        {
            m_trigger_values[i] = get_random_value();
            m_weights[i] = get_random_value();
        }
    }

    void IntermediateLayer::propagate_forward(IntermediateLayer& next_layer)
    {
        next_layer.m_inputs.fill(0.0f);

        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; inc(pos))
            {
                if (m_inputs.get(i, pos) >= m_trigger_values.get(i, pos))
                {
                    const auto next_neuron_index = m_connections.get(i, pos);

                    const auto weight = m_weights.get(i, pos);
                    next_layer.m_inputs.set(
                        next_neuron_index,
                        std::clamp(
                            next_layer.m_inputs.get(next_neuron_index) + weight * m_inputs.get(i, pos), 0.0f, 1.0f
                        )
                    );
                }
            }
        }
    }

    void NeuralNetwork::propagate_forward()
    {
        for (size_t i = 0; i < (m_intermediate_layers.size() - 1); ++i)
        {
            m_intermediate_layers[i].propagate_forward(m_intermediate_layers[i + 1]);
        }
    }


    void OutputLayer::update_output_weights(const template_token_vector<float, TokenID>& delta, float learning_rate)
    {
        for (const auto i = TokenID::START; i < TokenID::MAX; inc(i))
        {
            m_weights[i] = std::clamp(m_weights[i] + learning_rate * delta[i] * m_inputs[i], 0.0f, 1.0f);
            // Adjust trigger: lower when delta > 0 (fire more), raise when delta < 0.
            m_trigger_values[i] = std::clamp(m_trigger_values[i] - learning_rate * delta[i], 0.0f, 1.0f);
        }
    }

    void IntermediateLayer::propagate_backward(
        const template_token_vector<float, TokenID>& delta,
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
        float learning_rate
    )
    {
        for (const auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; inc(i))
        {
            for (const auto pos = PositionIndex::START; pos < PositionIndex::MAX; inc(pos))
            {
                if (m_inputs.get(i, pos) < m_trigger_values.get(i, pos))
                    continue; // neuron did not fire, no gradient to propagate

                const auto next_neuron_index = m_connections.get(i, pos);
                assert(static_cast<TokenID>(i) < TokenID::MAX);
                const float d = delta[static_cast<TokenID>(i)];

                // Increase weight when downstream error is positive (need more signal).
                m_weights.set(
                    i, pos, std::clamp(m_weights.get(i, pos) + learning_rate * d * m_inputs.get(i, pos), 0.0f, 1.0f)
                );

                // Lower trigger makes this neuron fire more easily — helpful when
                // downstream error is positive.
                m_trigger_values.set(i, pos, std::clamp(m_trigger_values.get(i, pos) - learning_rate * d, 0.0f, 1.0f));

                // Accumulate gradient for the layer below.
                prev_delta.set(i, pos, prev_delta.get(i, pos) + d * m_weights.get(i, pos));
            }
        }
    }


    void IntermediateLayer::propagate_backward(
        const template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& delta,
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
        float learning_rate
    )
    {
        for (const auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; inc(i))
        {
            for (const auto pos = PositionIndex::START; pos < PositionIndex::MAX; inc(pos))
            {
                if (m_inputs.get(i, pos) < m_trigger_values.get(i, pos))
                    continue; // neuron did not fire, no gradient to propagate

                const auto next_neuron_index = m_connections.get(i, pos);
                const float d = delta.get(i, pos);

                // Increase weight when downstream error is positive (need more signal).
                m_weights.set(
                    i, pos, std::clamp(m_weights.get(i, pos) + learning_rate * d * m_inputs.get(i, pos), 0.0f, 1.0f)
                );

                // Lower trigger makes this neuron fire more easily — helpful when
                // downstream error is positive.
                m_trigger_values.set(i, pos, std::clamp(m_trigger_values.get(i, pos) - learning_rate * d, 0.0f, 1.0f));

                // Accumulate gradient for the layer below.
                prev_delta.set(i, pos, prev_delta.get(i, pos) + d * m_weights.get(i, pos));
            }
        }
    }

    // NeuralNetwork

    static void try_add_to_top_k(std::vector<OutputToken>& top_k, TokenID id, float value, float weight, size_t k)
    {
        if (top_k.size() < k)
        {
            top_k.push_back({id, value, weight});
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
        else if (!top_k.empty() && value > top_k.back().activation)
        {
            top_k.back() = {id, value, weight};
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
    }

    void NeuralNetwork::set_input_layer(const InputLine& input)
    {
        assert(!m_intermediate_layers.empty());
        m_input_layer.set_input_layer(input);
    }

    // returns the top-K with the biggest activation in the output layer,
    // sorted descending by activation
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k, Corpus& corpus) const
    {
        assert(!m_intermediate_layers.empty());

        std::vector<OutputToken> top_k_pairs;
        for (const auto i = TokenID::START; i < TokenID::MAX; inc(i))
        {
            if (m_output_layer.m_inputs[i] < m_output_layer.m_trigger_values[i])
                continue; // neuron did not fire
            try_add_to_top_k(
                top_k_pairs, static_cast<TokenID>(i), m_output_layer.m_inputs[i], m_output_layer.m_weights[i], top_k
            );
        }
        return top_k_pairs;
    }

    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        for (const auto i = TokenID::START; i < TokenID::MAX; inc(i))
        {
            score.values[i] = m_inputs[i];
        }
    }

    void NeuralNetwork::compute_score(Score& score, const TokenID expected_output_token)
    {
        m_output_layer.compute_score(score, expected_output_token);
    }

    void OutputLayer::compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const
    {
        for (const auto i = TokenID::START; i < TokenID::MAX; inc(i))
        {
            deltas[i] = score.values[i] - m_inputs[i];
        }
    }


    void NeuralNetwork::propagate_backward(const Score& score)
    {
        static constexpr float LEARNING_RATE = 0.01f;

        // delta[i] = signed error: target - actual.
        // Positive  → neuron fires too little  → increase weight.
        // Negative  → neuron fires too much    → decrease weight.

        template_token_vector<float, TokenID> output_layer_delta;
        m_output_layer.compute_deltas(score, output_layer_delta);
        // Update the output layer's weights directly from the score delta.
        m_output_layer.update_output_weights(output_layer_delta, LEARNING_RATE);

        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> delta;
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> prev_delta;
        m_intermediate_layers.back().propagate_backward(output_layer_delta, delta, LEARNING_RATE);

        // Walk backwards through the remaining layers.
        assert(m_intermediate_layers.size() >= 2);
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            prev_delta.fill(0.0f);
            m_intermediate_layers[l].propagate_backward(delta, prev_delta, LEARNING_RATE);
            delta = prev_delta;
        }
    }

    void NeuralNetwork::set_random_weights_and_connections(Corpus& corpus)
    {
        m_input_layer.set_random_weights_and_connections();

        for (size_t i = 0; i < m_intermediate_layers.size() - 1; ++i)
        {
            m_intermediate_layers[i].set_random_weights_and_connections();
        }
        m_intermediate_layers[m_intermediate_layers.size() - 1].set_random_weights_and_connections_to_output_layer(
            corpus
        );

        m_output_layer.set_random_weights_and_connections_for_output_layer(corpus);
    }

    void NeuralNetwork::train(Corpus& corpus)
    {
        std::println("Training the neural network...");

        set_random_weights_and_connections(corpus);

        // Get a training example from the corpus. The example needs at least 2
        // tokens. Note that the list can be padded to 2 tokens using
        // UNKNOWN_TOKEN_ID if necessary.
        const auto train_output = corpus.get_training_input_line(2);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string = corpus.get_line(train_output);

        size_t num_iterations = 1000000;
        for (size_t i = 0; i < num_iterations; ++i)
        {
            Score score;
            set_input_layer(train_input);
            propagate_forward();
            compute_score(score, expected_output_token);
            propagate_backward(score);

            if (i % 100 == 0)
            {
                const auto expected_token = corpus.get_token_from_id(expected_output_token);
                std::println(
                    "Training iteration[{}], wanted: '{}' ({}), full string: '{}'",
                    i,
                    expected_token,
                    expected_output_token,
                    full_string
                );

                int prediction_index = 0;
                const auto predicted_token_id_lists = get_best_output_token_ids(5, corpus);
                for (const auto& entry : predicted_token_id_lists)
                {
                    const auto predicted_token = corpus.get_token_from_id(entry.token_id);
                    std::println(
                        "\t predicted: {} / pred:'{}' (id: '{}'), {} (weight: {})",
                        prediction_index,
                        predicted_token,
                        entry.token_id,
                        entry.activation,
                        entry.weight
                    );
                    prediction_index++;
                }
            }
        }
    }

} // namespace rllm
