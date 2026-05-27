#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>

namespace rllm
{
    const char* training_method_to_string(TrainingMethod method)
    {
        switch (method)
        {
        case TrainingMethod::TWO_TOK:
            return "two_tok";
        case TrainingMethod::THREE_TOK:
            return "three_tok";
        case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
            return "increasingly_longer";
        case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
            return "random_line_random_len";
        case TrainingMethod::WINDOW:
            return "window";
        }
        return "UNKNOWN";
    }
} // namespace rllm

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
    // Number of gradient-update passes over all layers per training example per epoch.
    constexpr size_t NUMBER_OF_LAYER_VISITS_PER_EXAMPLE = 8;

    // Layers

    struct ForwardWorkspace
    {
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h;
        fixed_size_vector<rlmm_float, EmbeddingDimension> h_last;
        explicit ForwardWorkspace(PositionIndex seq_len) : h(seq_len) {}
    };

    void NeuralNetwork::propagate_forward(const InputLine& input)
    {
        assert(!m_transformer_blocks.empty());

        m_last_input = input;
        m_seq_len = input.size();

        auto ws = std::make_unique<ForwardWorkspace>(m_seq_len);

        // Embed tokens + sinusoidal positional encoding → h[T × D_MODEL]
        m_input_layer.propagate_forward(input, ws->h);

        // Pass through each transformer block in order
        for (auto& block : m_transformer_blocks)
            block.forward(ws->h, m_seq_len);

        // Keep full hidden state for backward pass
        m_last_hidden = ws->h;

        // Project the last-position hidden state to vocabulary logits
        const auto last_pos = dec(m_seq_len);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->h_last[d] = ws->h[last_pos, d];
        m_output_layer.forward_from_hidden(ws->h_last);
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

    static bool timed_checkpoint_due(
        const std::optional<std::chrono::seconds>& checkpointing_interval,
        std::chrono::steady_clock::time_point& last_checkpoint_at
    )
    {
        if (!checkpointing_interval.has_value())
            return false;

        const auto now = std::chrono::steady_clock::now();
        if ((now - last_checkpoint_at) < checkpointing_interval.value())
            return false;

        last_checkpoint_at = now;
        return true;
    }

    // Returns top-K tokens selected by logit, with probabilities normalized
    // over the returned top-K set. This keeps prompt-time probabilities stable
    // and informative even when the full-vocabulary softmax is extremely sharp.
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k) const
    {
        assert(!m_transformer_blocks.empty());
        if (top_k == 0)
            return {};

        // First, keep only top-K logits.
        std::vector<OutputToken> top_k_pairs;
        for (const auto i : enum_iterator<TokenID>())
        {
            const auto logit = m_output_layer.m_inputs[i];
            try_add_to_top_k(top_k_pairs, i, logit, top_k);
        }

        if (top_k_pairs.empty())
            return top_k_pairs;

        // Then normalize only over top-K using a numerically stable softmax.
        const double max_logit = static_cast<double>(top_k_pairs.front().activation);
        double sum_exp = 0.0;
        for (const auto& entry : top_k_pairs)
            sum_exp += std::exp(static_cast<double>(entry.activation) - max_logit);

        for (auto& entry : top_k_pairs)
        {
            const double p = std::exp(static_cast<double>(entry.activation) - max_logit) / sum_exp;
            entry.activation = static_cast<float>(p);
        }

        return top_k_pairs;
    }


    struct BackwardPropWorkspace
    {
        fixed_size_vector<rlmm_float, TokenID>                                   output_layer_delta;
        fixed_size_vector<rlmm_float, EmbeddingDimension>                        h_last_vec;
        fixed_size_vector<rlmm_float, EmbeddingDimension>                        dh_last;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>    dh;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>    din;
        explicit BackwardPropWorkspace(PositionIndex seq_len) : dh(seq_len), din(seq_len) {}
    };

    void NeuralNetwork::propagate_backward(const Score& score)
    {
        static constexpr float LEARNING_RATE = 0.01f;

        auto ws = std::make_unique<BackwardPropWorkspace>(m_seq_len);

        m_output_layer.compute_deltas(score, ws->output_layer_delta);

        // Backpropagate through LM head → get dL/dh_last
        const auto last_pos = dec(m_seq_len);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->h_last_vec[d] = m_last_hidden[last_pos, d];
        ws->dh_last = m_output_layer.backward_and_update(ws->output_layer_delta, ws->h_last_vec, LEARNING_RATE);

        // Initialise full-sequence gradient: zero everywhere except the last position
        ws->dh.fill(RLMM_ZERO);
        ws->din.fill(RLMM_ZERO);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->dh[last_pos, d] = ws->dh_last[d];

        // Backward through transformer blocks in reverse order
        for (int i = static_cast<int>(m_transformer_blocks.size()) - 1; i >= 0; --i)
        {
            m_transformer_blocks[i].backward(ws->dh, ws->din, LEARNING_RATE);
            ws->dh = ws->din;
        }

        // Update token embeddings
        m_input_layer.propagate_backward(m_last_input, ws->dh, LEARNING_RATE);
    }


    void NeuralNetwork::set_random_weights_and_connections()
    {
        m_input_layer.set_random_embeddings();
        for (auto& block : m_transformer_blocks)
            block.randomize();
        m_output_layer.set_random_weights();
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
        const int n_tok = static_cast<int>(TokenID::MAX);
        float max_val = m_output_layer.m_inputs[TokenID::START];
#pragma omp simd reduction(max : max_val)
        for (int i = 0; i < n_tok; ++i)
            max_val = math::max(max_val, m_output_layer.m_inputs[static_cast<TokenID>(i)]);

        float sum_exp = 0.0f;
#pragma omp simd reduction(+ : sum_exp)
        for (int i = 0; i < n_tok; ++i)
        {
            const float term = std::exp(m_output_layer.m_inputs[static_cast<TokenID>(i)] - max_val);
            OVERFLOW_CHECK_ADD(sum_exp, term);
            sum_exp += term;
        }

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
            const auto line = line_of_file.sub_array(line_substring_length);
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

        const auto train_input = line_of_file.sub_array(static_cast<PositionIndex>(num_tokens));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line: '{}'", full_string);

        do_training(train_input, verbose, max_iterations);
    }

    void NeuralNetwork::train_with_random_len_from_start(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations,
        std::mt19937& rng
    )
    {
        const int line_len = static_cast<int>(line_of_file.size());
        if (line_len < 2)
            return;

        std::uniform_int_distribution<int> len_dist(2, line_len);
        const int random_len = len_dist(rng);

        const auto train_input = line_of_file.sub_array(static_cast<PositionIndex>(random_len));
        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());

        LOG_INFO("Training on random line prefix ({} toks): '{}'", random_len, *full_string_opt);

        do_training(train_input, verbose, max_iterations);
    }

    void NeuralNetwork::train_random_line_random_len_epoch(
        size_t epoch,
        const std::vector<InputLine>& training_lines,
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval,
        std::chrono::steady_clock::time_point& last_checkpoint_at,
        std::mt19937& rng
    )
    {
        const auto total_lines = training_lines.size();
        if (total_lines == 0)
            return;

        std::uniform_int_distribution<size_t> line_dist(0, total_lines - 1);

        for (size_t lines_visited = 1; lines_visited <= total_lines; ++lines_visited)
        {
            const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

            if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
            {
                std::println(
                    "creating timed checkpoint at epoch {}, line {}, total lines visited {}",
                    epoch,
                    lines_visited,
                    total_lines
                );
                save(std::format("models/checkpoint-{}.json", epoch * total_lines + lines_visited));
            }

            LOG_INFO(
                "Epoch[{}%] random-line[{}]: {:0.2f}% done",
                epoch / static_cast<float>(num_epochs) * 100.0f,
                lines_visited,
                progress * 100.0f
            );

            const auto& random_line = training_lines[line_dist(rng)];
            train_with_random_len_from_start(
                random_line,
                verbose,
                NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size(),
                rng
            );
        }
    }


    void NeuralNetwork::do_line_based_training(
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        std::vector<InputLine> training_lines = m_corpus.get_suitable_training_lines();

        // Multi-epoch training with shuffling prevents catastrophic forgetting:
        // each example only gets 8*num_layers gradient updates per
        // pass, so no single example can overwrite all the others.
        std::mt19937 rng{42};
        const auto total_lines = training_lines.size();
        auto last_checkpoint_at = std::chrono::steady_clock::now();

        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            if (m_training_method == TrainingMethod::RANDOM_LINE_RANDOM_LEN)
            {
                train_random_line_random_len_epoch(
                    epoch,
                    training_lines,
                    verbose,
                    num_epochs,
                    checkpointing_interval,
                    last_checkpoint_at,
                    rng
                );

                std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                save(std::format("models/checkpoint-epoch-{}.json", epoch));
                continue;
            }

            std::shuffle(training_lines.begin(), training_lines.end(), rng);

            size_t lines_visited = 0;
            for (const auto& line_of_file : training_lines)
            {
                lines_visited++;
                const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

                if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                {
                    std::println(
                        "creating timed checkpoint at epoch {}, line {}, total lines visited {}",
                        epoch,
                        lines_visited,
                        total_lines
                    );
                    save(std::format("models/checkpoint-{}.json", epoch * total_lines + lines_visited));
                }

                LOG_INFO(
                    "Epoch[{}%] line[{}]: {:0.2f}% done",
                    epoch / static_cast<float>(num_epochs) * 100.0f,
                    lines_visited,
                    progress * 100.0f
                );

                switch (m_training_method)
                {
                case TrainingMethod::TWO_TOK:
                    train_with_up_to_N(
                        line_of_file, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size(), 2
                    );
                    break;

                case TrainingMethod::THREE_TOK:
                    train_with_up_to_N(
                        line_of_file, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size(), 3
                    );
                    break;

                case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                    train_with_increasingly_longer_sequences(
                        line_of_file, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size()
                    );
                    break;

                case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
                    // Handled in the random-line loop above.
                    assert(false);
                    break;

                case TrainingMethod::WINDOW:
                    // window methods don't use the line-based loop; handled separately below
                    assert(false);
                    break;
                }
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            save(std::format("models/checkpoint-epoch-{}.json", epoch));
        }
    }

    void NeuralNetwork::train_with_window(
        int window_size,
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
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
        auto last_checkpoint_at = std::chrono::steady_clock::now();
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            LOG_INFO("Epoch[{}%]: {:0.2f}% done", epoch / static_cast<float>(num_epochs) * 100.0f, 0.0f);

            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t j = 0; j < num_windows; ++j)
            {
                const float progress = static_cast<float>(j) / static_cast<float>(num_windows);

                InputLine window;
                int current_try_len = 2 + random_int(0, window_size - 2); // random length between 2 and window_size
                for (int k = 0; k < current_try_len; ++k)
                {
                    window.push_back(tokens[indices[j] + static_cast<size_t>(k)]);
                }
                assert(window.size() == static_cast<PositionIndex>(current_try_len));

                total_windows_trained++;

                if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                {
                    std::println(
                        "creating timed checkpoint at epoch {}, window {}, total windows trained {}",
                        epoch,
                        j,
                        total_windows_trained
                    );
                    save(std::format("models/checkpoint-{}.json", total_windows_trained));
                }


                if (total_windows_trained % 100 == 0)
                {
                    const auto line_opt = m_corpus.get_line(window);
                    LOG_INFO(
                        "Epoch[{}%] window[{}]: {:0.2f}% done for '{}', successes: {}, failures: {}",
                        epoch / static_cast<float>(num_epochs) * 100.0f,
                        j,
                        progress * 100.0f,
                        line_opt.has_value() ? line_opt->c_str() : "unknown",
                        m_stats.num_learning_successes(),
                        m_stats.num_learning_failures()
                    );
                }

                do_training(window, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size());
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            save(std::format("models/checkpoint-epoch-{}.json", epoch));
        }
    }

    void NeuralNetwork::do_whole_corpus_window_based_training(
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        // Window methods operate on the flat token stream rather than per-line.
        assert(m_training_method == TrainingMethod::WINDOW);
        train_with_window(m_window_size, verbose, num_epochs, checkpointing_interval);
    }


    void NeuralNetwork::train(
        bool verbose,
        size_t num_epochs,
        const std::optional<std::string>& input_filename,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        Statistics::TotalLearnRecorderScope total_learn_recorder_scope(m_stats);

        if (input_filename)
        {
            if (!load(*input_filename))
            {
                std::println("Failed to load model from '{}'", *input_filename);
                std::exit(1);
            }
            LOG_INFO("Loaded model from '{}'", *input_filename);
        }
        else
        {
            std::println("No input model specified, starting with random weights.");
            set_random_weights_and_connections();
        }

        const size_t vocab       = static_cast<size_t>(TokenID::MAX);
        const size_t d_model     = static_cast<size_t>(EmbeddingDimension::MAX);
        const size_t d_ff        = static_cast<size_t>(FFDimension::MAX);
        const size_t n_layers    = m_transformer_blocks.size();
        const size_t params_embed    = vocab * d_model;           // token embeddings
        const size_t params_lm_head  = vocab * d_model;           // LM head
        const size_t params_attn     = 4 * d_model * d_model;     // W_q, W_k, W_v, W_o per block
        const size_t params_ffn      = 3 * d_model * d_ff;        // W_gate, W_up, W_down per block (2*d_ff*d_model + d_model*d_ff)
        const size_t params_per_block = params_attn + params_ffn;
        const size_t total_params    = params_embed + params_lm_head + n_layers * params_per_block;

        LOG_INFO(
            "Training the neural network...\n"
            "\t $num_layers: {}\n"
            "\t $corpus_size: {}\n"
            "\t $total_params: {} ({:.2f}M)  [embed:{} lm_head:{} blocks:{}x{}]\n"
            "\t convergence threshold: {:.6f}\n"
            "\t fires nothing CE loss:  {:.6f}\n"
            "\t steps per example per epoch: {}\n"
            "\t num epochs: {}\n"
            "\t training method: {}\n",
            n_layers,
            vocab,
            total_params, static_cast<float>(total_params) / 1e6f,
            params_embed, params_lm_head, n_layers, params_per_block,
            m_convergence_threshold,
            m_fires_nothing_ce_loss,
            NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * n_layers,
            num_epochs,
            training_method_to_string(m_training_method)
        );

        if (training_method_is_line_based())
        {
            do_line_based_training(verbose, num_epochs, checkpointing_interval);
        }
        else
        {
            do_whole_corpus_window_based_training(verbose, num_epochs, checkpointing_interval);
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
                    "Convergence reached after {} steps for expected '{}', full string: '{}', input size: {}",
                    i + 1,
                    m_corpus.get_token_from_id(expected_output_token),
                    full_string,
                    static_cast<size_t>(train_input.size())
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
            "Steps exhausted ({}) for this line. loss = {:.6f}, threshold = {:.6f}, expected token: '{}' ({}), full "
            "string: '{}', input size: {}.",
            max_iterations,
            loss,
            m_convergence_threshold,
            m_corpus.get_token_from_id(expected_output_token),
            expected_output_token,
            full_string,
            static_cast<size_t>(train_input.size())
        );
        m_stats.record_learning_failure();
    }


} // namespace rllm
