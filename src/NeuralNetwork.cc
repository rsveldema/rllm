#include <RLLM.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <print>

namespace rllm
{
    static size_t clip_max(size_t value, size_t max)
    {
        if (value > max)
            return max;
        return value;
    }

    // Layers

    void NeuralNetwork::propagate_forward()
    {
        assert(!m_intermediate_layers.empty());

        // Propagate from input layer into the first intermediate layer.
        auto& first_layer = m_intermediate_layers.front();
        first_layer.m_inputs.fill(0.0f);
        for (auto token = TokenID::START; token < TokenID::MAX; token = inc(token))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                if (m_input_layer.m_inputs.get(token, pos) < m_input_layer.m_trigger_values.get(token, pos))
                    continue;

                const auto next_neuron_index = m_input_layer.m_connections.get(token, pos);
                const auto weight = m_input_layer.m_weights.get(token, pos);
                first_layer.m_inputs.set(
                    next_neuron_index,
                    std::clamp(
                        first_layer.m_inputs.get(next_neuron_index) + weight * m_input_layer.m_inputs.get(token, pos),
                        0.0f,
                        1.0f
                    )
                );
            }
        }

        // Propagate across intermediate layers.
        for (size_t i = 0; i < (m_intermediate_layers.size() - 1); ++i)
        {
            m_intermediate_layers[i].propagate_forward(m_intermediate_layers[i + 1]);
        }

        // Propagate from the last intermediate layer into the output layer.
        m_output_layer.m_inputs.fill(0.0f);
        const auto& last_layer = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                if (last_layer.m_inputs.get(i, pos) < last_layer.m_trigger_values.get(i, pos))
                    continue;

                const auto [target, _target_pos] = last_layer.m_connections.get(i, pos);
                if (static_cast<size_t>(target) >= static_cast<size_t>(TokenID::MAX))
                    continue;

                const auto token_id = static_cast<TokenID>(target);
                const auto weight = last_layer.m_weights.get(i, pos);
                m_output_layer.m_inputs[token_id] = std::clamp(
                    m_output_layer.m_inputs[token_id] + weight * last_layer.m_inputs.get(i, pos), 0.0f, 1.0f
                );
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
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            if (m_output_layer.m_inputs[i] < m_output_layer.m_trigger_values[i])
                continue; // neuron did not fire
            try_add_to_top_k(
                top_k_pairs, static_cast<TokenID>(i), m_output_layer.m_inputs[i], m_output_layer.m_weights[i], top_k
            );
        }
        return top_k_pairs;
    }

    void NeuralNetwork::compute_score(Score& score, const TokenID expected_output_token)
    {
        m_output_layer.compute_score(score, expected_output_token);
    }


    void NeuralNetwork::propagate_backward(const Score& score)
    {
        static constexpr float LEARNING_RATE = 0.01f;

        // delta[i] = signed error: target - actual.
        // Positive  → neuron fires too little  → increase weight.
        // Negative  → neuron fires too much    → decrease weight.

        auto output_layer_delta = std::make_unique<template_token_vector<float, TokenID>>();
        static_assert(sizeof(*output_layer_delta) < 65536, "output_layer_delta is too large for the stack" );
        m_output_layer.compute_deltas(score, *output_layer_delta);
        // Update the output layer's weights directly from the score delta.
        m_output_layer.update_output_weights(*output_layer_delta, LEARNING_RATE);

        auto delta = std::make_unique<template_token_matrix<float, IntermediateLayerIndex, PositionIndex>>();
        auto prev_delta = std::make_unique<template_token_matrix<float, IntermediateLayerIndex, PositionIndex>>();
        m_intermediate_layers.back().propagate_backward(*output_layer_delta, *delta, LEARNING_RATE);

        // Walk backwards through the remaining layers.
        assert(m_intermediate_layers.size() >= 2);
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            prev_delta->fill(0.0f);
            m_intermediate_layers[l].propagate_backward(*delta, *prev_delta, LEARNING_RATE);
            *delta = *prev_delta;
        }
    }

    void NeuralNetwork::set_random_weights_and_connections(Corpus& corpus)
    {
        m_input_layer.set_random_weights_and_connections();

        for (size_t i = 0; i < m_intermediate_layers.size() - 1; ++i)
        {
            m_intermediate_layers[i].set_random_weights_and_connections();
        }
        m_intermediate_layers.back().set_random_weights_and_connections_to_output_layer(
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
