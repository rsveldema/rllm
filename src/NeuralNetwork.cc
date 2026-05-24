#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>

static std::ofstream s_nn_log;
#define LOG_INFO(...) (s_nn_log << std::format(__VA_ARGS__) << '\n' << std::flush)


#define LOG_INFO_EVERY_N(...) \
    do \
    { \
        static int counter = 0; \
        if (counter++ % 100 == 0) \
        { \
            LOG_INFO(__VA_ARGS__); \
        } \
    } while (0)

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
    // how many times to iterate over the entire training dataset
    constexpr size_t MAX_TRAINING_EPOCHS = 1000;
    // Per-example gradient steps per epoch: keep small to limit catastrophic forgetting.
    // Other examples undo large per-example bursts; interleaving more often (smaller bursts,
    // more epochs) with the same total updates converges much more reliably.
    constexpr size_t STEPS_PER_EXAMPLE_PER_EPOCH = 30;

    // Layers

    void NeuralNetwork::propagate_forward(const InputLine& input)
    {
        assert(!m_intermediate_layers.empty());

        m_last_input = input; // retained for embedding backprop

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

        // After the loop *delta = ∂L/∂(first_intermediate_layer.m_inputs).
        // Pass it to the input layer so the embeddings can be updated.
        m_input_layer.propagate_backward(m_last_input, *delta, LEARNING_RATE);
    }


    void NeuralNetwork::set_random_weights_and_connections()
    {
        m_input_layer.set_random_embeddings();
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

    // Compute cross-entropy loss: -log(softmax(logits)[target])
    // Only considers the active corpus tokens, not the full TokenID::MAX space.
    float NeuralNetwork::compute_loss(TokenID expected_output_token) const
    {
        float max_val = m_output_layer.m_inputs[TokenID::START];
        for (const auto i : enum_iterator<TokenID>(TokenID::MAX))
            max_val = std::max(max_val, m_output_layer.m_inputs[i]);

        float sum_exp = 0.0f;
        for (const auto i : enum_iterator<TokenID>(TokenID::MAX))
            sum_exp += std::exp(m_output_layer.m_inputs[i] - max_val);

        const float log_prob = m_output_layer.m_inputs[expected_output_token] - max_val - std::log(sum_exp);
        return -log_prob;
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

    void NeuralNetwork::train_with_up_to_N(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations,
        int num_tokens
    )
    {
        assert(num_tokens >= 2);

        // If the line is too short for num_tokens, fall back to fewer tokens
        // (minimum 2 so there is always at least one input token and one target).
        if (static_cast<int>(line_of_file.size()) < num_tokens)
        {
            if (num_tokens > 2)
                train_with_up_to_N(line_of_file, verbose, max_iterations, num_tokens - 1);
            return;
        }

        const auto train_input = line_of_file.substr(static_cast<PositionIndex>(num_tokens));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line: '{}'", full_string);

        do_training(train_input, verbose, max_iterations);
    }


    void NeuralNetwork::do_line_based_training(bool verbose)
    {
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
                    train_with_up_to_N(line_of_file, verbose, STEPS_PER_EXAMPLE_PER_EPOCH, 2);
                    break;

                case TrainingMethod::THREE_TOK:
                    train_with_up_to_N(line_of_file, verbose, STEPS_PER_EXAMPLE_PER_EPOCH, 3);
                    break;

                case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                    train_with_increasingly_longer_sequences(line_of_file, verbose, STEPS_PER_EXAMPLE_PER_EPOCH);
                    break;

                case TrainingMethod::WINDOW2:
                case TrainingMethod::WINDOW3:
                    // window methods don't use the line-based loop; handled separately below
                    assert(false);
                    break;
                }
            }
        }
    }

    void NeuralNetwork::train_with_window(int window_size, bool verbose)
    {
        assert(window_size >= 2);

        // Collect the full flat token sequence from every corpus file.
        std::vector<TokenID> tokens;
        m_corpus.visit_flat_tokens([&](TokenID tok) {
            tokens.push_back(tok);
        });

        if (tokens.size() < static_cast<size_t>(window_size))
            return;

        // Each valid start index yields one training example.
        const size_t num_windows = tokens.size() - static_cast<size_t>(window_size) + 1;
        std::vector<size_t> indices(num_windows);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng{42};
        size_t total_windows_trained = 0;
        for (size_t epoch = 0; epoch < MAX_TRAINING_EPOCHS; ++epoch)
        {
            LOG_INFO("Epoch[{}%]: {:0.2f}% done", epoch / static_cast<float>(MAX_TRAINING_EPOCHS) * 100.0f, 0.0f);

            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t j = 0; j < num_windows; ++j)
            {
                const float progress = static_cast<float>(j) / static_cast<float>(num_windows);


                InputLine window;
                for (int k = 0; k < window_size; ++k)
                    window.push_back(tokens[indices[j] + static_cast<size_t>(k)]);

                total_windows_trained++;
                if (total_windows_trained % 100 == 0)
                {
                    const auto line_opt = m_corpus.get_line(window);
                    LOG_INFO(
                        "Epoch[{}%] window[{}]: {:0.2f}% done for '{}'",
                        epoch / static_cast<float>(MAX_TRAINING_EPOCHS) * 100.0f,
                        j,
                        progress * 100.0f,
                        line_opt.has_value() ? line_opt->c_str() : "unknown"
                    );
                }

                do_training(window, verbose, STEPS_PER_EXAMPLE_PER_EPOCH);
            }
        }
    }

    void NeuralNetwork::do_whole_corpus_window_based_training(bool verbose)
    {
        // Window methods operate on the flat token stream rather than per-line.
        switch (m_training_method)
        {
        case TrainingMethod::WINDOW2:
            train_with_window(2, verbose);
            break;

        case TrainingMethod::WINDOW3:
            train_with_window(3, verbose);
            break;

        default:
            assert(false);
        }
    }


    void NeuralNetwork::train(bool verbose)
    {
        Statistics::TotalLearnRecorderScope total_learn_recorder_scope(m_stats);

        set_random_weights_and_connections();

        LOG_INFO(
            "Training the neural network...\n"
            "\t $num_layers: {}\n"
            "\t $corpus_size: {}\n"
            "\t $intermediate_layers width: {}\n"
            "\t convergence threshold: {:.6f}\n"
            "\t fires nothing CE loss:  {:.6f}\n"
            "\t steps per example per epoch: {}\n"
            "\t training method: {}\n",
            m_intermediate_layers.size(),
            static_cast<size_t>(TokenID::MAX),
            static_cast<size_t>(IntermediateLayerIndex::MAX),
            m_convergence_threshold,
            m_fires_nothing_ce_loss,
            STEPS_PER_EXAMPLE_PER_EPOCH,
            [this]() -> std::string_view {
                switch (m_training_method)
                {
                case TrainingMethod::TWO_TOK:
                    return "TWO_TOK";
                case TrainingMethod::THREE_TOK:
                    return "THREE_TOK";
                case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                    return "INCREASINGLY_LONGER_SEQUENCES";
                case TrainingMethod::WINDOW2:
                    return "WINDOW2";
                case TrainingMethod::WINDOW3:
                    return "WINDOW3";
                }
                return "UNKNOWN";
            }()
        );

        if (training_method_is_line_based())
        {
            do_line_based_training(verbose);
        }
        else
        {
            do_whole_corpus_window_based_training(verbose);
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

            if (loss < m_convergence_threshold)
            {
                LOG_INFO_EVERY_N(
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
            m_convergence_threshold
        );
        m_stats.record_learning_failure();
    }


} // namespace rllm
