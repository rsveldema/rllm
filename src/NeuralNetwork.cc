#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

    static size_t training_steps_per_example(size_t num_layers)
    {
        (void)num_layers;
        // Keep a stable update budget across model depth. Scaling linearly with
        // layer count made deeper models (e.g. 4 layers) overfit tiny corpora
        // and collapse to near one-hot predictions in just a few epochs.
        return NUMBER_OF_LAYER_VISITS_PER_EXAMPLE;
    }

    // Layers

    struct NeuralNetworkForwardWorkspace
    {
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h;
        fixed_size_vector<rlmm_float, EmbeddingDimension> h_last;
        explicit NeuralNetworkForwardWorkspace(PositionIndex seq_len) : h(seq_len) {}
    };

    void NeuralNetwork::propagate_forward(const InputLine& input)
    {
        assert(!m_transformer_blocks.empty());

        m_last_input = input;
        m_seq_len = input.size();

        auto ws = std::make_unique<NeuralNetworkForwardWorkspace>(m_seq_len);

        // Embed tokens + sinusoidal positional encoding → h[T × D_MODEL]
        m_input_layer.propagate_forward(input, ws->h);

        // Pass through each transformer block in order
        for (auto& block : m_transformer_blocks)
            block.forward(ws->h, m_seq_len);

        // Keep full hidden state for backward pass
        m_last_hidden = ws->h;

        // Project the last-position hidden state to vocabulary logits.
        // Given a string of N tokens, the model learns to predict the N+1'th token,
        // so the final output is based on the hidden state at the last input position.
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
        else if (!top_k.empty() && value >= top_k.back().activation)
        {
            // Use >= so ties don't always leave the initial token IDs (0,1,2,...) in place.
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
    // over the returned top-K set.
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k) const
    {
        assert(!m_transformer_blocks.empty());
        if (top_k == 0)
            return {};

        // First, keep only top-K by raw logit (preserves correct rank order).
        std::vector<OutputToken> top_k_pairs;
        for (const auto i : enum_iterator<TokenID>())
        {
            const auto logit = m_output_layer.m_inputs[i];
            try_add_to_top_k(top_k_pairs, i, logit, top_k);
        }

        if (top_k_pairs.empty())
            return top_k_pairs;

        const double best_logit = static_cast<double>(top_k_pairs.front().activation);

        // Cap the logit gap so that all top-K tokens have a non-negligible
        // probability in the softmax (avoids numerical underflow for tokens
        // that are slightly below the best logit).
        static constexpr double MAX_LOGIT_GAP = 5.0;
        for (auto& entry : top_k_pairs)
        {
            const double capped = std::max(static_cast<double>(entry.activation), best_logit - MAX_LOGIT_GAP);
            entry.activation = static_cast<float>(capped);
        }

        // Stable softmax over the gap-capped logits.
        double sum_exp = 0.0;
        for (const auto& entry : top_k_pairs)
            sum_exp += std::exp(static_cast<double>(entry.activation) - best_logit);

        for (auto& entry : top_k_pairs)
        {
            const double p = std::exp(static_cast<double>(entry.activation) - best_logit) / sum_exp;
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
        static constexpr float BASE_LEARNING_RATE = 0.003f;
        const float learning_rate = BASE_LEARNING_RATE / static_cast<float>(std::max<size_t>(1, m_transformer_blocks.size()));

        auto ws = std::make_unique<BackwardPropWorkspace>(m_seq_len);

        m_output_layer.compute_deltas(score, ws->output_layer_delta);

        // Backpropagate through LM head → get dL/dh_last
        const auto last_pos = dec(m_seq_len);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->h_last_vec[d] = m_last_hidden[last_pos, d];
        ws->dh_last = m_output_layer.backward_and_update(ws->output_layer_delta, ws->h_last_vec, learning_rate);

        // Initialise full-sequence gradient: zero everywhere except the last position
        ws->dh.fill(RLMM_ZERO);
        ws->din.fill(RLMM_ZERO);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->dh[last_pos, d] = ws->dh_last[d];

        // Backward through transformer blocks in reverse order
        for (int i = static_cast<int>(m_transformer_blocks.size()) - 1; i >= 0; --i)
        {
            m_transformer_blocks[i].backward(ws->dh, ws->din, learning_rate);
            ws->dh = ws->din;
        }

        // Update token embeddings
        m_input_layer.propagate_backward(m_last_input, ws->dh, learning_rate);
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

    float NeuralNetwork::evaluate_average_loss(const std::vector<InputLine>& evaluation_lines)
    {
        if (evaluation_lines.empty())
            return std::numeric_limits<float>::quiet_NaN();

        double total_loss = 0.0;
        for (const auto& example : evaluation_lines)
        {
            assert(static_cast<int>(example.size()) >= 2);
            auto train_input = example;
            const auto expected_output_token = train_input.back();
            train_input.pop_back();

            propagate_forward(train_input);
            total_loss += static_cast<double>(compute_loss(expected_output_token));
        }

        return static_cast<float>(total_loss / static_cast<double>(evaluation_lines.size()));
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

        // Include short prefixes (len=2) so one-token contexts can learn their next token
        // (e.g. "# -> define/include"). Longer prefixes are still emphasized by the
        // effective_max_iterations scaling below.
        const int min_len = 2;
        std::uniform_int_distribution<int> len_dist(min_len, line_len);
        const int random_len = len_dist(rng);

        const auto train_input = line_of_file.sub_array(static_cast<PositionIndex>(random_len));
        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());

        LOG_INFO("Training on random line prefix ({} toks): '{}'", random_len, *full_string_opt);

        // Reward longer correct sequences with a larger optimization budget.
        // len=2 keeps the base budget; longer prefixes scale up proportionally
        // (capped to avoid runaway per-example compute on large corpora).
        const size_t length_factor = std::max<size_t>(1, static_cast<size_t>(random_len) - 1);
        const size_t capped_factor = std::min<size_t>(length_factor, 4);
        const size_t effective_max_iterations = max_iterations * capped_factor;

        do_training(train_input, verbose, effective_max_iterations);
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
        assert(total_lines > 0);

        // Visit each line exactly once per epoch (in random order).
        // Sampling with replacement over-emphasizes some lines while skipping others,
        // which causes unstable learning and apparent forgetting.
        std::vector<size_t> line_indices(total_lines);
        std::iota(line_indices.begin(), line_indices.end(), 0);
        std::shuffle(line_indices.begin(), line_indices.end(), rng);

        for (size_t lines_visited = 1; lines_visited <= total_lines; ++lines_visited)
        {
            const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

            if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
            {
                LOG_INFO(
                    "Creating timed checkpoint at epoch {}, line {}, total lines visited {}",
                    epoch,
                    lines_visited,
                    total_lines
                );
                checkpoint();
            }

            LOG_INFO(
                "Epoch[{}%] random-line[{}]: {:0.2f}% done",
                epoch / static_cast<float>(num_epochs) * 100.0f,
                lines_visited,
                progress * 100.0f
            );

            const auto& random_line = training_lines[line_indices[lines_visited - 1]];
            train_with_random_len_from_start(
                random_line,
                verbose,
                training_steps_per_example(m_transformer_blocks.size()),
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
        auto split = m_corpus.get_deterministic_training_split(VALIDATION_PERCENT);
        std::vector<InputLine> training_lines = std::move(split.training_lines);
        std::vector<InputLine> validation_lines = std::move(split.validation_lines);

        // With tiny corpora, a 20% split can leave only one held-out line.
        // That makes early stopping and best-checkpoint restoration extremely noisy
        // and can collapse the final model to whichever single line is validated.
        if (validation_lines.size() < 2)
        {
            training_lines.insert(
                training_lines.end(),
                std::make_move_iterator(validation_lines.begin()),
                std::make_move_iterator(validation_lines.end())
            );
            validation_lines.clear();
            LOG_INFO("Validation disabled for tiny corpus (fewer than 2 held-out lines)");
        }

        LOG_INFO(
            "Using {} training lines and {} validation lines ({}% target validation split)",
            training_lines.size(),
            validation_lines.size(),
            VALIDATION_PERCENT
        );

        assert(!training_lines.empty());

        // Multi-epoch training with shuffling prevents catastrophic forgetting:
        // each example only gets 8*num_layers gradient updates per
        // pass, so no single example can overwrite all the others.
        std::mt19937 rng{42};
        const auto total_lines = training_lines.size();
        auto last_checkpoint_at = std::chrono::steady_clock::now();
        float best_validation_loss = std::numeric_limits<float>::infinity();
        size_t epochs_without_improvement = 0;
        bool has_best_checkpoint = false;
        static constexpr const char* BEST_CHECKPOINT_FILENAME = "models/checkpoint-best.json";

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
                checkpoint();
            }
            else
            {
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
                        checkpoint();
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
                            line_of_file, verbose, training_steps_per_example(m_transformer_blocks.size()), 2
                        );
                        break;

                    case TrainingMethod::THREE_TOK:
                        train_with_up_to_N(
                            line_of_file, verbose, training_steps_per_example(m_transformer_blocks.size()), 3
                        );
                        break;

                    case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                        train_with_increasingly_longer_sequences(
                            line_of_file, verbose, training_steps_per_example(m_transformer_blocks.size())
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
                checkpoint();
            }

            if (!validation_lines.empty())
            {
                const float validation_loss = evaluate_average_loss(validation_lines);
                LOG_INFO(
                    "epoch {} validation loss: {:.6f} across {} held-out lines",
                    epoch,
                    validation_loss,
                    validation_lines.size()
                );

                if ((validation_loss + VALIDATION_IMPROVEMENT_EPSILON) < best_validation_loss)
                {
                    best_validation_loss = validation_loss;
                    epochs_without_improvement = 0;
                    save(BEST_CHECKPOINT_FILENAME);
                    has_best_checkpoint = true;
                    LOG_INFO(
                        "Saved new best checkpoint '{}' with validation loss {:.6f}",
                        BEST_CHECKPOINT_FILENAME,
                        validation_loss
                    );
                }
                else
                {
                    epochs_without_improvement++;
                    LOG_INFO(
                        "Validation loss did not improve for {} epoch(s); best remains {:.6f}",
                        epochs_without_improvement,
                        best_validation_loss
                    );
                    if (epochs_without_improvement >= EARLY_STOPPING_PATIENCE)
                    {
                        LOG_INFO(
                            "early stopping at epoch {} after {} epoch(s) without validation improvement",
                            epoch,
                            epochs_without_improvement
                        );
                        break;
                    }
                }
            }
        }

        if (has_best_checkpoint)
        {
            if (load(BEST_CHECKPOINT_FILENAME))
            {
                LOG_INFO(
                    "restored best checkpoint '{}' with validation loss {:.6f}",
                    BEST_CHECKPOINT_FILENAME,
                    best_validation_loss
                );
            }
            else
            {
                LOG_INFO("failed to restore best checkpoint '{}'", BEST_CHECKPOINT_FILENAME);
            }
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
                    LOG_INFO(
                        "creating timed checkpoint at epoch {}, window {}, total windows trained {}",
                        epoch,
                        j,
                        total_windows_trained
                    );
                    checkpoint();
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

                do_training(window, verbose, training_steps_per_example(m_transformer_blocks.size()));
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            checkpoint();
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
            LOG_INFO("No input model specified, starting with random weights.");
            set_random_weights_and_connections();
        }

        const size_t vocab       = static_cast<size_t>(TokenID::MAX);
        const size_t corpus_size = m_corpus.count_num_lines();
        const size_t d_model     = static_cast<size_t>(EmbeddingDimension::MAX);
        const size_t d_ff        = static_cast<size_t>(FFDimension::MAX);
        const size_t n_layers    = m_transformer_blocks.size();
        const size_t params_embed    = vocab * d_model;           // token embeddings
        const size_t params_lm_head  = vocab * d_model;           // LM head
        const size_t params_attn     = 4 * d_model * d_model;     // W_q, W_k, W_v, W_o per block
        const size_t params_ffn      = 3 * d_model * d_ff;        // W_gate, W_up, W_down per block (2*d_ff*d_model + d_model*d_ff)
        const size_t params_per_block = params_attn + params_ffn;
        const size_t total_params    = params_embed + params_lm_head + n_layers * params_per_block;
        const float  total_params_mib = static_cast<float>(total_params * sizeof(rlmm_float)) / (1024.0f * 1024.0f);

        static constexpr size_t CORPUS_SIZE_WARNING_THRESHOLD = 100000;
        if (corpus_size > CORPUS_SIZE_WARNING_THRESHOLD)
        {
            LOG_INFO(
                "WARNING: corpus is large ({} lines > {} threshold). Training may take significantly longer; "
                "consider narrower --filter values, fewer --epochs, or window training.",
                corpus_size,
                CORPUS_SIZE_WARNING_THRESHOLD
            );
        }

        LOG_INFO(
            "Training the neural network...\n"
            "\t $num_layers: {}\n"
            "\t $corpus_size: {}\n"
            "\t $total_params: {} ({:.2f}M, {:.2f} MiB)  [embed:{} lm_head:{} blocks:{}x{}]\n"
            "\t convergence threshold: {:.6f}\n"
            "\t fires nothing CE loss:  {:.6f}\n"
            "\t steps per example per epoch: {}\n"
            "\t num epochs: {}\n"
            "\t training method: {}\n",
            n_layers,
            corpus_size,
            total_params, static_cast<float>(total_params) / 1e6f, total_params_mib,
            params_embed, params_lm_head, n_layers, params_per_block,
            m_convergence_threshold,
            m_fires_nothing_ce_loss,
            training_steps_per_example(n_layers),
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
        // given a training example line [5,34,3,4,1], we use the first N-1 tokens
        // as input ([5,34,3,4]) and the last token as the expected output (1).
        assert(static_cast<int>(train_output.size()) >= 2);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        const bool trace_enabled = []() {
            const char* v = std::getenv("RLLM_TRACE_INCLUDE");
            return v != nullptr && std::string(v) != "0";
        }();
        const bool trace_this_example = trace_enabled && (full_string.find("#include") != std::string::npos);

        auto trace_probes = [&](const char* phase, size_t iter, float loss_value) {
            if (!trace_this_example)
                return;

            static const std::array<std::string, 2> probes = {"#", "#include"};
            for (const auto& probe : probes)
            {
                const auto probe_ids = m_corpus.get_token_ids(probe);
                if (probe_ids.empty())
                    continue;

                propagate_forward(probe_ids);
                const auto top5 = get_best_output_token_ids(5);

                if (top5.empty())
                {
                    LOG_INFO(
                        "[TRACE] phase='{}' iter={} train='{}' probe='{}' top1='<none>' loss={:.6f}",
                        phase,
                        iter,
                        full_string,
                        probe,
                        loss_value
                    );
                    continue;
                }

                const auto top1_tok = m_corpus.get_token_from_id(top5.front().token_id);
                LOG_INFO(
                    "[TRACE] phase='{}' iter={} train='{}' probe='{}' top1='{}' p={:.6f} loss={:.6f}",
                    phase,
                    iter,
                    full_string,
                    probe,
                    top1_tok,
                    top5.front().activation,
                    loss_value
                );

                for (size_t k = 0; k < top5.size(); ++k)
                {
                    LOG_INFO(
                        "[TRACE]   top{} token='{}' id={} p={:.6f}",
                        k,
                        m_corpus.get_token_from_id(top5[k].token_id),
                        top5[k].token_id,
                        top5[k].activation
                    );
                }
            }
        };

        // In multi-epoch training each call gets a small fixed budget (max_iterations).
        // We allow an early return once this example reaches a reasonable confidence target.
        float loss = 0.0f;
        for (size_t i = 0; i < max_iterations; ++i)
        {
            Score score;
            propagate_forward(train_input);
            m_output_layer.compute_score(score, expected_output_token);
            propagate_backward(score);
            loss = compute_loss(expected_output_token);

            if (trace_this_example && i < 4)
                trace_probes("post_step", i + 1, loss);

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
