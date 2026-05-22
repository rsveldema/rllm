#include <CircularBuffer.hpp>
#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <print>

namespace rllm
{
    constexpr size_t MAX_TRAINING_ITERATIONS_PER_LINE = 1000;
    constexpr float CONVERGENCE_THRESHOLD = 0.001f;


    // Layers

    void NeuralNetwork::propagate_forward(const InputLine& input)
    {
        assert(!m_intermediate_layers.empty());

        // Propagate from input layer into the first intermediate layer.
        m_input_layer.propagate_forward(input, m_intermediate_layers.front());

        // Propagate across intermediate layers.
        for (size_t i = 0; i < (m_intermediate_layers.size() - 1); ++i)
        {
            m_intermediate_layers[i].propagate_forward(m_intermediate_layers[i + 1]);
        }

        // Propagate from the last intermediate layer into the output layer.
        m_intermediate_layers.back().propagate_forward_to_output(m_output_layer);
    }


    // NeuralNetwork

    static void try_add_to_top_k(std::vector<OutputToken>& top_k, TokenID id, float value, size_t k)
    {
        if (top_k.size() < k)
        {
            top_k.emplace_back(id, value);
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
        else if (!top_k.empty() && value > top_k.back().activation)
        {
            top_k.back() = {id, value};
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
    }

    // returns the top-K with the biggest activation in the output layer,
    // sorted descending by activation.
    // if no prediction can be made, returns an empty list.
    // Do not try to return any if the activation is 0 or negative, as that means the token did not activate at all.
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k) const
    {
        assert(!m_intermediate_layers.empty());

        std::vector<OutputToken> top_k_pairs;
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            const auto activation = m_output_layer.m_inputs[i];
            if (activation <= 0.0f)
            {
                continue; // skip tokens that did not activate at all
            }
            try_add_to_top_k(top_k_pairs, i, activation, top_k);
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
        static_assert(sizeof(*output_layer_delta) < 65536, "output_layer_delta is too large for the stack");
        m_output_layer.compute_deltas(score, *output_layer_delta);

        auto delta = std::make_unique<template_token_vector<float, IntermediateLayerIndex>>();
        auto prev_delta = std::make_unique<template_token_vector<float, IntermediateLayerIndex>>();
        m_intermediate_layers.back().propagate_backward_from_output_layer(*output_layer_delta, *delta, LEARNING_RATE);

        // Walk backwards through the remaining layers.
        assert(m_intermediate_layers.size() >= 2);
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            prev_delta->fill(0.0f);
            m_intermediate_layers[l].propagate_backward(*delta, *prev_delta, LEARNING_RATE);
            *delta = *prev_delta;
        }
    }


    void NeuralNetwork::set_random_weights_and_connections()
    {
        for (size_t i = 0; i < m_intermediate_layers.size() - 1; ++i)
        {
            m_intermediate_layers[i].set_random_weights_and_connections();
        }
        m_intermediate_layers.back().set_random_weights_and_connections_to_output_layer();
    }


    void NeuralNetwork::dump_top_predictions()
    {
        int prediction_index = 0;
        const auto predicted_token_id_lists = get_best_output_token_ids(5);
        for (const auto& entry : predicted_token_id_lists)
        {
            const auto predicted_token = m_corpus.get_token_from_id(entry.token_id);
            std::println(
                "\t prediction[{} of {}] / pred:'{}' (id: '{}'), {}",
                prediction_index,
                predicted_token_id_lists.size(),
                predicted_token,
                entry.token_id,
                entry.activation
            );
            prediction_index++;
        }
    }

    // Compute mean squared error loss between output activations and expected output
    float NeuralNetwork::compute_loss(TokenID expected_output_token) const
    {
        float loss = 0.0f;
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            float target = (i == expected_output_token) ? 1.0f : 0.0f;
            float pred = m_output_layer.m_inputs[i];
            float diff = pred - target;
            loss += diff * diff;
        }
        return loss / static_cast<float>(static_cast<int>(TokenID::MAX));
    }

    void NeuralNetwork::train(bool verbose)
    {
        std::println("Training the neural network...");

        Statistics::TotalLearnRecorderScope total_learn_recorder_scope(m_stats);

        set_random_weights_and_connections();

        int total_lines = m_corpus.count_num_lines();
        int lines_visited = 0;

        std::vector<InputLine> training_lines;
        m_corpus.visit_lines([&](const InputLine& line) {
            training_lines.push_back(line);
        });

        for (const auto& line_of_file : training_lines)
        {
            lines_visited++;
            const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

            for (const auto& line_substring_length : enum_iterator<PositionIndex>(line_of_file.size()))
            {
                const auto line = line_of_file.substr(line_substring_length);
                if (line.empty())
                    continue; // skip empty lines that can't be used for training

                const auto full_string_opt = m_corpus.get_line(line);
                assert(full_string_opt.has_value());
                const auto& full_string = *full_string_opt;

                if (static_cast<int>(line.size()) < 2)
                {
                    std::println(
                        "Skipping line with size {} (too short for training): '{}'",
                        static_cast<int>(line.size()),
                        full_string
                    );
                    continue; // skip too-short lines that can't be used for training
                }
                std::println(
                    "Training on line[{}]: '{}', {:0.2f}% done", lines_visited, full_string, progress * 100.0f
                );

                do_training(line, verbose);
            }
        }
    }


    void NeuralNetwork::do_training(const InputLine& train_output, bool verbose)
    {
        assert(static_cast<int>(train_output.size()) >= 2);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        CircularBuffer<float, 100> recent_losses;
        auto loss_function_converged = [&](float loss) {
            recent_losses.push_back(loss);
            if (recent_losses.size() < recent_losses.capacity())
                return false; // not enough data yet
            float average_loss = std::accumulate(recent_losses.begin(), recent_losses.end(), 0.0f) /
                static_cast<float>(recent_losses.size());
            return average_loss < CONVERGENCE_THRESHOLD; // convergence threshold
        };

        size_t i = 0;
        while (true)
        {
            i++;

            if (i > MAX_TRAINING_ITERATIONS_PER_LINE)
            {
                std::println("Reached maximum training iterations for this line. Stopping training on this line.");

                m_stats.record_learning_failure();
                break;
            }

            Score score;
            propagate_forward(train_input);
            compute_score(score, expected_output_token);
            propagate_backward(score);

            float loss = compute_loss(expected_output_token);

            if (loss_function_converged(loss))
            {
                std::println(
                    "Convergence reached at iteration {} for token '{}'",
                    i,
                    m_corpus.get_token_from_id(expected_output_token)
                );
                dump_top_predictions();
                m_stats.record_learning_success();
                break;
            }


            if (verbose && i % 100 == 0)
            {
                const auto expected_token = m_corpus.get_token_from_id(expected_output_token);
                std::println(
                    "Training iteration[{}], wanted: '{}' ({}), full string: '{}'",
                    i,
                    expected_token,
                    expected_output_token,
                    full_string
                );
                std::println("  Loss: {:.6f}", loss);
                dump_top_predictions();
            }
        }
    }

} // namespace rllm
