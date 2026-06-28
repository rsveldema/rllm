#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <parallel.hpp>
#include <vecmath.hpp>
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
        case TrainingMethod::RANDOM_LINE_FULL:
            return "random_line_full";
        case TrainingMethod::WINDOW:
            return "window";
        }
        return "UNKNOWN";
    }
} // namespace rllm

static std::ofstream s_nn_log;

#ifdef LOG_INFO
#undef LOG_INFO
#endif

#define LOG_INFO(...) (s_nn_log << std::format(__VA_ARGS__) << '\n' << std::flush)
#define LOG_ERROR(...) (s_nn_log << std::format(__VA_ARGS__) << '\n' << std::flush)


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
    static void scatter_dh_last_to_row(
        // OFFLOAD_PARAMETERS(dh_last, dh, last_pos)
        const fixed_size_vector<float, EmbeddingDimension>& dh_last,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        PositionIndex last_pos
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(queue, d, enum_iterator1D<EmbeddingDimension>(), (dh_last, dh, last_pos))
        dh[last_pos, d] = dh_last[d];
        ENDFOR
    }

    // Number of gradient-update passes over all layers per training example per epoch.
    constexpr size_t NUMBER_OF_LAYER_VISITS_PER_EXAMPLE = 8;

    static size_t training_steps_per_example(size_t num_layers)
    {
        (void) num_layers;
        // Keep a stable update budget across model depth. Scaling linearly with
        // layer count made deeper models (e.g. 4 layers) overfit tiny corpora
        // and collapse to near one-hot predictions in just a few epochs.
        return NUMBER_OF_LAYER_VISITS_PER_EXAMPLE;
    }

    static void print_parallel_statistics_for_epoch(size_t epoch)
    {
        std::println("Parallel statistics after epoch {}:", epoch);
        parallel::statistics.print_statistics();
    }

    // Layers

    struct NeuralNetworkForwardWorkspace
    {
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> h;
        fixed_size_vector<float, EmbeddingDimension> h_last;

        ForwardWorkspace workspace;

        explicit NeuralNetworkForwardWorkspace(PositionIndex seq_len)
            : h(seq_len),
              workspace(seq_len)
        {
            h_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(PositionIndex seq_len)
        {
            h.set_rows(seq_len);
            workspace.reset(seq_len);
        }
    };

    void NeuralNetwork::propagate_forward()
    {
        assert(!m_transformer_blocks.empty());

        m_seq_len = m_last_input.size();
        m_forward_workspace->reset(m_seq_len);

        auto& ws = *m_forward_workspace;
        ws.h.set_rows(m_seq_len);

        // Embed tokens + sinusoidal positional encoding → h[T × D_MODEL]
        m_input_layer.propagate_forward(m_last_input, ws.h);

        // Pass through each transformer block in order
        for (auto& block : m_transformer_blocks)
            block.forward(ws.h, m_seq_len, ws.workspace);

        // Project the last-position hidden state to vocabulary logits.
        // Given a string of N tokens, the model learns to predict the N+1'th token,
        // so the final output is based on the hidden state at the last input position.
        const auto last_pos = dec(m_seq_len);
        copy_hidden_row_to_vector(ws.h, last_pos, ws.h_last);

        // TODO: inline forward_from_hidden to turn this into a OFFLOAD_PARFOR_2D_PARAM
        PARFOR_1D(output_index, enum_iterator1D<MultiTokenPredictionIndex>())
            m_output_layers[output_index].forward_from_hidden(ws.h_last);
        ENDFOR
    }


    // NeuralNetwork

    static constexpr std::chrono::seconds MIN_TIME_BETWEEN_CHECKPOINTS{30};

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

    // Save a checkpoint, but skip it if less than MIN_TIME_BETWEEN_CHECKPOINTS
    // has elapsed since the last one (avoids rapid-fire saves at epoch boundaries).
    static void epoch_checkpoint(
        std::chrono::steady_clock::time_point& last_checkpoint_at,
        size_t epoch,
        const std::function<void()>& do_checkpoint
    )
    {
        const auto now = std::chrono::steady_clock::now();
        if ((now - last_checkpoint_at) < MIN_TIME_BETWEEN_CHECKPOINTS)
        {
            LOG_INFO("Skipping end-of-epoch checkpoint at epoch {} (too soon since last checkpoint)", epoch);
            return;
        }
        last_checkpoint_at = now;
        do_checkpoint();
    }

    // Returns top-K tokens selected by logit, with probabilities normalized
    // over the returned top-K set.
    std::vector<OutputToken>
    NeuralNetwork::get_best_output_token_ids(size_t top_k, MultiTokenPredictionIndex head) const
    {
        assert(!m_transformer_blocks.empty());
        assert(top_k > 0);

        // First, keep only top-K by raw logit (preserves correct rank order).
        std::vector<OutputToken> top_k_pairs = m_output_layers[head].get_top_k_by_logit(top_k);

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
        fixed_size_vector<float, TokenID> output_layer_delta;
        fixed_size_vector<float, EmbeddingDimension> h_last_vec;
        fixed_size_vector<float, EmbeddingDimension> dh_last;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> dh;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> din;
        BackwardWorkspace transformer_block;

        explicit BackwardPropWorkspace(PositionIndex seq_len)
            : dh(seq_len)
            , din(seq_len)
            , transformer_block(seq_len)
        {
            output_layer_delta.set_size(TokenID::MAX);
            h_last_vec.set_size(EmbeddingDimension::MAX);
            dh_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(PositionIndex seq_len)
        {
            dh.set_rows(seq_len);
            din.set_rows(seq_len);
            transformer_block.reset(seq_len);
            output_layer_delta.zero();
            h_last_vec.zero();
            dh_last.zero();
        }
    };

    NeuralNetwork::NeuralNetwork(size_t num_layers, Corpus& corpus, Statistics& stats)
        : m_corpus(corpus)
        , m_stats(stats)
        , m_input_layer()
        , m_fires_nothing_ce_loss(std::log(static_cast<float>(TokenID::MAX)))
        , m_convergence_threshold(m_fires_nothing_ce_loss / k_convergence_divisor)
    {
        assert(static_cast<size_t>(TokenID::MAX) > 1);
        for (size_t i = 0; i < num_layers; ++i)
            m_transformer_blocks.emplace_back();

        m_forward_workspace = std::make_unique<NeuralNetworkForwardWorkspace>(PositionIndex::START);
        m_backward_workspace = std::make_unique<BackwardPropWorkspace>(PositionIndex::START);
    }

    NeuralNetwork::~NeuralNetwork() = default;

    void NeuralNetwork::propagate_backward_mtp(
        const fixed_size_obj_vector<Score, MultiTokenPredictionIndex>& scores,
        MultiTokenPredictionIndex num_valid
    )
    {
        assert(m_seq_len > PositionIndex::START);

        m_backward_workspace->reset(m_seq_len);

        static constexpr float BASE_LEARNING_RATE = 0.003f;
        const float learning_rate =
            BASE_LEARNING_RATE / static_cast<float>(std::max<size_t>(1, m_transformer_blocks.size()));

        auto& ws = *m_backward_workspace;
        ws.dh.set_rows(m_seq_len);
        ws.din.set_rows(m_seq_len);

        const auto last_pos = dec(m_seq_len);
        copy_hidden_row_to_vector(m_forward_workspace->h, last_pos, ws.h_last_vec);

        // Accumulate dh_last contributions from every valid output head.
        ws.dh_last.zero();
        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>(num_valid))
        {
            ws.output_layer_delta = scores[oi].values;
            m_output_layers[oi].backward_and_update(ws.output_layer_delta, ws.h_last_vec, ws.dh_last, learning_rate);
        }

        // Propagate the summed gradient through the transformer blocks.
        ws.dh.zero();
        ws.din.zero();
        scatter_dh_last_to_row(ws.dh_last, ws.dh, last_pos);

        auto* p_dh  = &ws.dh;
        auto* p_din = &ws.din;
        for (int i = static_cast<int>(m_transformer_blocks.size()) - 1; i >= 0; --i)
        {
            m_transformer_blocks[i].backward(*p_dh, *p_din, ws.transformer_block, learning_rate, m_forward_workspace->workspace);
            std::swap(p_dh, p_din);
        }

        m_input_layer.propagate_backward(m_last_input, *p_dh, learning_rate);
    }


    void NeuralNetwork::set_random_weights_and_connections()
    {
        m_input_layer.set_random_embeddings();

        for (auto& block : m_transformer_blocks)
            block.randomize();

        for (const auto output_index : enum_iterator1D<MultiTokenPredictionIndex>())
            m_output_layers[output_index].set_random_weights();
    }


    void NeuralNetwork::dump_top_predictions()
    {
        int prediction_index = 0;
        const auto predicted_token_id_lists = get_best_output_token_ids(5, MultiTokenPredictionIndex::START);
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


    float NeuralNetwork::evaluate_average_loss(const std::vector<InputLine>& evaluation_lines)
    {
        if (evaluation_lines.empty())
            return std::numeric_limits<float>::quiet_NaN();

        Score score;

        double total_loss = 0.0;
        for (const auto& example : evaluation_lines)
        {
            const int seq_len = static_cast<int>(example.size());
            assert(seq_len >= 2);
            const int num_valid = std::min(seq_len - 1, static_cast<int>(MultiTokenPredictionIndex::MAX));
            const int input_len = seq_len - num_valid;

            example.sub_array(get_last_input(),
                        static_cast<PositionIndex>(input_len));

            propagate_forward();

            double example_loss = 0.0;
            for (int k = 0; k < num_valid; ++k)
            {
                const auto head = static_cast<MultiTokenPredictionIndex>(k);
                const auto target = example[static_cast<PositionIndex>(input_len + k)];
                score.reset();
                example_loss += static_cast<double>(m_output_layers[head].compute_score(score, target));
            }
            total_loss += example_loss / static_cast<double>(num_valid);
        }

        return static_cast<float>(total_loss / static_cast<double>(evaluation_lines.size()));
    }

    void NeuralNetwork::train_with_increasingly_longer_sequences(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations
    )
    {
        InputLine line;
        for (const auto& line_substring_length : enum_iterator1D<PositionIndex>(line_of_file.size()))
        {
            line_of_file.sub_array(line, line_substring_length);
            if (line.empty())
                continue; // skip empty lines that can't be used for training

            const auto full_string_opt = m_corpus.get_line(line);
            assert(full_string_opt.has_value());
            const auto& full_string = *full_string_opt;

            if (static_cast<int>(line.size()) < 2)
            {
                continue; // skip too-short lines that can't be used for training
            }

            LOG_INFO("Training on line[{}]: '{}'", (int)line_substring_length, full_string);

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

        InputLine train_input;
        line_of_file.sub_array(train_input, static_cast<PositionIndex>(num_tokens));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line[{}]: '{}'", num_tokens, full_string);

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

        InputLine train_input;
        line_of_file.sub_array(train_input, static_cast<PositionIndex>(random_len));
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
                    "Creating timed checkpoint at epoch: {}, line: {}, total lines todo: {}",
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
            if (m_training_method == TrainingMethod::RANDOM_LINE_FULL)
            {
                // Train on the full line — all tokens are visible, last token is target.
                if (static_cast<int>(random_line.size()) >= 2)
                    do_training(random_line, verbose, training_steps_per_example(m_transformer_blocks.size()));
            }
            else
            {
                train_with_random_len_from_start(
                    random_line, verbose, training_steps_per_example(m_transformer_blocks.size()), rng
                );
            }
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
        static constexpr const char* BEST_CHECKPOINT_FILENAME = "models/checkpoint-best.st";

        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            bool should_stop = false;

            if (m_training_method == TrainingMethod::RANDOM_LINE_RANDOM_LEN ||
                m_training_method == TrainingMethod::RANDOM_LINE_FULL)
            {
                train_random_line_random_len_epoch(
                    epoch, training_lines, verbose, num_epochs, checkpointing_interval, last_checkpoint_at, rng
                );

                epoch_checkpoint(last_checkpoint_at, epoch, [&] {
                    std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                    checkpoint();
                });
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

                    case TrainingMethod::RANDOM_LINE_FULL:
                        // Handled in the random-line loop above.
                        assert(false);
                        break;

                    case TrainingMethod::WINDOW:
                        // window methods don't use the line-based loop; handled separately below
                        assert(false);
                        break;
                    }
                }

                epoch_checkpoint(last_checkpoint_at, epoch, [&] {
                    std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                    checkpoint();
                });
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
                        should_stop = true;
                    }
                }
            }

            print_parallel_statistics_for_epoch(epoch);

            if (should_stop)
                break;
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
            print_parallel_statistics_for_epoch(epoch);
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

        const size_t vocab = static_cast<size_t>(TokenID::MAX);
        const size_t corpus_size = m_corpus.count_num_lines();
        const size_t d_model = static_cast<size_t>(EmbeddingDimension::MAX);
        const size_t d_ff = static_cast<size_t>(FFDimension::MAX);
        const size_t n_layers = m_transformer_blocks.size();
        const size_t params_embed = vocab * d_model; // token embeddings
        const size_t params_lm_head = static_cast<size_t>(MultiTokenPredictionIndex::MAX) * vocab * d_model;
        const size_t params_attn = 4 * d_model * d_model; // W_q, W_k, W_v, W_o per block
        const size_t params_ffn = 3 * d_model * d_ff; // W_gate, W_up, W_down per block (2*d_ff*d_model + d_model*d_ff)
        const size_t params_per_block = params_attn + params_ffn;
        const size_t total_params = params_embed + params_lm_head + n_layers * params_per_block;
        const float total_params_mib = static_cast<float>(total_params * sizeof(float)) / (1024.0f * 1024.0f);

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
            total_params,
            static_cast<float>(total_params) / 1e6f,
            total_params_mib,
            params_embed,
            params_lm_head,
            n_layers,
            params_per_block,
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
        // Multi-token prediction (MTP): given a sequence [t0,t1,...,tN-1] we
        // use the first (N - num_valid_heads) tokens as context and train each
        // head k to predict the token at position (context_len + k).
        assert(static_cast<int>(train_output.size()) >= 2);
        const int _seq_len = static_cast<int>(train_output.size());
        const int _max_heads = static_cast<int>(MultiTokenPredictionIndex::MAX);
        const int _num_valid = std::min(_seq_len - 1, _max_heads);
        const int _input_len = _seq_len - _num_valid;
        InputLine train_input;
        train_output.sub_array(train_input, static_cast<PositionIndex>(_input_len));
        const auto num_valid_heads = static_cast<MultiTokenPredictionIndex>(_num_valid);
        // Head-0 target is used for logging and convergence checks.
        const auto expected_output_token = train_output[static_cast<PositionIndex>(_input_len)];

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        get_last_input() = train_input; // set the input to the train input for tracing


        // In multi-epoch training each call gets a small fixed budget (max_iterations).
        // We allow an early return once this example reaches a reasonable confidence target.
        // Heap-allocate the MTP score array: MAX_HEADS × Score ≈ 4 × 2.4 KB = ~9.6 KB.
        const auto scores_storage = std::make_unique<fixed_size_obj_vector<Score, MultiTokenPredictionIndex>>();
        float loss = 0.0f;
        for (size_t i = 0; i < max_iterations; ++i)
        {
            propagate_forward();

            scores_storage->set_size(num_valid_heads);
            for (const auto _k : enum_iterator1D<MultiTokenPredictionIndex>(num_valid_heads))
            {
                Score& s = (*scores_storage)[_k];
                const auto _target = train_output[static_cast<PositionIndex>(_input_len + static_cast<int>(_k))];
                const float _k_loss = m_output_layers[_k].compute_score(s, _target);
                if (std::isnan(_k_loss)) {
                    LOG_ERROR("loss became NaN!");
                }
                if (!std::isfinite(_k_loss))
                {
                    LOG_INFO(
                        "Non-finite loss detected for head {}, target '{}' ({}), full string: '{}', input size: {}.",
                        static_cast<size_t>(_k),
                        m_corpus.get_token_from_id(_target),
                        _target,
                        full_string,
                        static_cast<size_t>(train_input.size())
                    );
                    m_stats.record_learning_failure();
                    return;
                }
                if (_k == MultiTokenPredictionIndex::START)
                    loss = _k_loss;
            }

            propagate_backward_mtp(*scores_storage, num_valid_heads);

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
