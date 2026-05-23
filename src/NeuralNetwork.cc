#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <format>
#include <fstream>
#include <map>
#include <random>
#include <set>

static std::ofstream s_nn_log;
#define LOG_INFO(...) (s_nn_log << std::format(__VA_ARGS__) << '\n')

namespace rllm
{
    void set_nn_log_file(const std::string& filename)
    {
        if (s_nn_log.is_open())
            s_nn_log.close();
        s_nn_log.open(filename);
    }
} // namespace rllm

namespace rllm
{
    constexpr size_t MAX_TRAINING_ITERATIONS_PER_LINE = 3000;
    constexpr size_t MAX_TRAINING_EPOCHS = 1000;
    // Per-example gradient steps per epoch: keep small to limit catastrophic forgetting.
    // Other examples undo large per-example bursts; interleaving more often (smaller bursts,
    // more epochs) with the same total updates converges much more reliably.
    constexpr size_t STEPS_PER_EXAMPLE_PER_EPOCH = 20;

    // with TokenID::MAX = 2048, the MSE loss of an all-zero prediction is 1/2048 ≈ 0.000488.
    //  the '1' here is becomes of one-hot encoding of the expected output, where the target token has a value of 1 and
    //  all others have a value of 0.
    constexpr float FIRES_NOTHING_MSE_LOSS = 1.0f / static_cast<float>(static_cast<int>(TokenID::MAX));

    constexpr float CONVERGENCE_THRESHOLD =
        FIRES_NOTHING_MSE_LOSS / 4.0f; // Must be below the all-zero prediction loss for training to work.

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


    void NeuralNetwork::propagate_backward(const Score& score)
    {
        static constexpr float LEARNING_RATE = 0.01f;

        // delta[i] = signed error: target - actual.
        // Positive  → neuron fires too little  → increase weight.
        // Negative  → neuron fires too much    → decrease weight.

        auto output_layer_delta = std::make_unique<template_token_vector<float, TokenID>>();
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
            LOG_INFO(
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

    void NeuralNetwork::train_with_increasingly_longer_sequences(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations
    )
    {
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
                continue; // skip too-short lines that can't be used for training
            }

            LOG_INFO("Training on line: '{}'", full_string);

            do_training(line, verbose, max_iterations);
        }
    }

    void NeuralNetwork::train_with_two_tok(const InputLine& line_of_file, bool verbose, size_t max_iterations)
    {
        // take the 1st token of a sentence and predict the 2nd token.
        // we are not using the rest of the sentence, which makes this a
        // fast way to debug the algorithms.

        assert(static_cast<int>(line_of_file.size()) >= 2);

        const auto train_input = line_of_file.substr(static_cast<PositionIndex>(2));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line: '{}'", full_string);

        do_training(train_input, verbose, max_iterations);
    }

    void NeuralNetwork::train_with_three_tok(const InputLine& line_of_file, bool verbose, size_t max_iterations)
    {
        // Use the first 2 tokens as input and predict the 3rd token.
        // Falls back to two-tok for lines with fewer than 3 tokens.
        if (static_cast<int>(line_of_file.size()) < 3)
        {
            train_with_two_tok(line_of_file, verbose, max_iterations);
            return;
        }

        const auto train_input = line_of_file.substr(static_cast<PositionIndex>(3));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line: '{}'", full_string);

        do_training(train_input, verbose, max_iterations);
    }

    void NeuralNetwork::train(bool verbose)
    {
        LOG_INFO(
            "Training the neural network...\n"
            "\t $num_layers: {}\n"
            "\t $corpus_size: {}\n"
            "\t $intermediate_layers width: {}\n"
            "\t convergence threshold: {:.10f}\n"
            "\t fires nothing MSE loss: {:.10f}\n"
            "\t max iterations per line: {}",
            m_intermediate_layers.size(),
            m_corpus.number_of_token_types(),
            static_cast<size_t>(IntermediateLayerIndex::MAX),
            CONVERGENCE_THRESHOLD,
            FIRES_NOTHING_MSE_LOSS,
            MAX_TRAINING_ITERATIONS_PER_LINE
        );

        Statistics::TotalLearnRecorderScope total_learn_recorder_scope(m_stats);

        set_random_weights_and_connections();

        std::vector<InputLine> training_lines = m_corpus.get_suitable_training_lines();

        // Multi-epoch training with shuffling prevents catastrophic forgetting:
        // each example only gets STEPS_PER_EXAMPLE_PER_EPOCH gradient updates per
        // pass, so no single example can overwrite all the others.
        std::mt19937 rng{42};
        const auto total_lines = training_lines.size();

        for (size_t epoch = 0; epoch < MAX_TRAINING_EPOCHS; ++epoch)
        {
            std::shuffle(training_lines.begin(), training_lines.end(), rng);

            size_t lines_visited = 0;
            for (const auto& line_of_file : training_lines)
            {
                lines_visited++;
                const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

                LOG_INFO(
                    "Epoch[{}%] line[{}]: {:0.2f}% done",
                    epoch / static_cast<float>(MAX_TRAINING_EPOCHS) * 100.0f,
                    lines_visited,
                    progress * 100.0f
                );

                switch (m_training_method)
                {
                case TrainingMethod::TWO_TOK:
                    train_with_two_tok(line_of_file, verbose, STEPS_PER_EXAMPLE_PER_EPOCH);
                    break;

                case TrainingMethod::THREE_TOK:
                    train_with_three_tok(line_of_file, verbose, STEPS_PER_EXAMPLE_PER_EPOCH);
                    break;

                case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                    train_with_increasingly_longer_sequences(line_of_file, verbose, STEPS_PER_EXAMPLE_PER_EPOCH);
                    break;
                }
            }
        }
    }


    void NeuralNetwork::do_training(const InputLine& train_output, bool verbose, size_t max_iterations)
    {
        assert(static_cast<int>(train_output.size()) >= 2);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        // In multi-epoch training each call gets a small fixed budget (max_iterations).
        // We run all steps unconditionally; convergence emerges across epochs.
        float loss = 0.0f;
        for (size_t i = 0; i < max_iterations; ++i)
        {
            Score score;
            propagate_forward(train_input);
            m_output_layer.compute_score(score, expected_output_token);
            propagate_backward(score);
            loss = compute_loss(expected_output_token);

            if (loss < CONVERGENCE_THRESHOLD)
            {
                LOG_INFO(
                    "Convergence reached after {} steps for token '{}'",
                    max_iterations,
                    m_corpus.get_token_from_id(expected_output_token)
                );
                m_stats.record_learning_success();
                return;
            }


            if (verbose && i % 25 == 0)
            {
                const auto expected_token = m_corpus.get_token_from_id(expected_output_token);
                LOG_INFO(
                    "Training iteration[{}], wanted: '{}' ({}), full string: '{}'",
                    i,
                    expected_token,
                    expected_output_token,
                    full_string
                );
                LOG_INFO("  Loss: {:.6f}", loss);
                dump_top_predictions();
            }
        }

        LOG_INFO(
            "Steps exhausted ({}) for this line. loss = {:.6f}, threshold = {:.6f}",
            max_iterations,
            loss,
            CONVERGENCE_THRESHOLD
        );
        m_stats.record_learning_failure();
    }

} // namespace rllm
