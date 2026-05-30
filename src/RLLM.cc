#include <RLLM.hpp>

#include <algorithm>
#include <iostream>
#include <print>
#include <string>

namespace rllm
{
    // maximum number of tokens to generate in response to a prompt before stopping
    static constexpr size_t MAX_NUM_ANSWER_TOKENS = 10;

    // Threshold for considering a predicted token as valid (not just noise).
    // This is a tunable hyperparameter.
    static constexpr float VALID_PREDICTION_THRESHOLD = 0.5f / 100.0f;

    void process_command(const std::string& _command, RLLM::PromptOptions& options, NeuralNetwork& nn)
    {
        const auto command = _command.empty() ? "/help" : _command;

        struct command_entry
        {
            std::vector<std::string_view> name;
            std::string_view description;
            std::function<void()> action;
        };

        std::vector<command_entry> commands = {
            {{"/help", "/h", "/?"},
             "Show this help message",
             [&]() {
                 std::println("Available commands:");
                 for (const auto& cmd : commands)
                 {
                     std::println("  {}: {}", cmd.name[0], cmd.description);
                 }
             }},
            {{"/exit"},
             "Exit the prompt mode",
             [&]() {
                 std::println("Exiting prompt mode.");
                 std::exit(0);
             }},
            {{"/info"},
             "Show information about the loaded model",
             [&]() {
                 std::println("Model information:");
                 std::println("  Vocabulary size:      {}", static_cast<size_t>(TokenID::MAX));
                 std::println("  Max context window:   {} tokens", static_cast<size_t>(PositionIndex::MAX));
                 std::println("  Embedding dimensions: {}", static_cast<size_t>(EmbeddingDimension::MAX));
                 std::println("  Attention heads:      {}", static_cast<size_t>(HeadsIndex::MAX));
                 std::println("  Transformer layers:   {}", nn.get_transformer_block_count());
             }},
            {{"/toggle_prio"}, "Toggle highest priority only mode", [&]() {
                 options.highest_prio_only = !options.highest_prio_only;
                 std::println(
                     "Toggled highest priority only mode. Now highest_prio_only is {}.", options.highest_prio_only
                 );
             }},
            {{"/hidden"}, "Mean-pool last hidden state over sequence (contextual embedding): /hidden <string>", [&]() {
                 const auto space = command.find(' ');
                 if (space == std::string::npos)
                 {
                     std::println("Usage: /hidden <string>");
                     return;
                 }
                 const std::string arg = command.substr(space + 1);
                 const auto& corpus = nn.get_corpus();
                 const auto token_ids = corpus.get_token_ids(arg);
                 nn.propagate_forward(token_ids);
                 const auto mean_vec = nn.get_last_hidden_mean();
                 constexpr size_t D = static_cast<size_t>(EmbeddingDimension::MAX);
                 constexpr size_t COLS = 8;
                 std::println("Hidden state embedding for '{}' ({} tokens, mean-pooled):", arg, static_cast<size_t>(token_ids.size()));
                 for (size_t i = 0; i < D; i += COLS)
                 {
                     std::print("  [{:3d}]", i);
                     for (size_t j = i; j < std::min(i + COLS, D); ++j)
                         std::print("  {:+.4f}", static_cast<float>(mean_vec[static_cast<EmbeddingDimension>(j)]));
                     std::println("");
                 }
             }},
            {{"/embed"}, "Print the raw input embedding vector(s) for a given string: /embed <string>", [&]() {
                 const auto space = command.find(' ');
                 if (space == std::string::npos)
                 {
                     std::println("Usage: /embed <string>");
                     return;
                 }
                 const std::string arg = command.substr(space + 1);
                 const auto& corpus = nn.get_corpus();
                 const auto& input_layer = nn.get_input_layer();
                 const auto token_ids = corpus.get_token_ids(arg);
                 constexpr size_t D = static_cast<size_t>(EmbeddingDimension::MAX);
                 constexpr size_t COLS = 8;
                 for (const auto pos : enum_iterator<PositionIndex>(token_ids.size()))
                 {
                     const TokenID tok = token_ids[pos];
                     const auto token_str = corpus.get_token_from_id(tok);
                     std::println("Embedding for token '{}' (id {}):", token_str, static_cast<size_t>(tok));
                     const auto& emb = input_layer.get_embedding(tok);
                     for (size_t i = 0; i < D; i += COLS)
                     {
                         std::print("  [{:3d}]", i);
                         for (size_t j = i; j < std::min(i + COLS, D); ++j)
                             std::print("  {:+.4f}", static_cast<float>(emb[static_cast<EmbeddingDimension>(j)]));
                         std::println("");
                     }
                 }
             }}
        };

        for (const auto& cmd : commands)
        {
            // Match either the exact command or the command followed by a space (for argument-taking commands).
            const bool exact = std::find(cmd.name.begin(), cmd.name.end(), command) != cmd.name.end();
            const bool prefix = std::any_of(cmd.name.begin(), cmd.name.end(), [&](std::string_view name) {
                return command.starts_with(std::string(name) + " ");
            });
            if (exact || prefix)
            {
                cmd.action();
                return;
            }
        }

        std::println("Unknown command: '{}'", command);
    }


    RLLM::RLLM(const std::vector<std::string>& filters)
        : m_filters(filters)
    {
        // Constructor implementation
    }

    void RLLM::prompt_mode(const std::string& filename, const std::optional<std::string>& one_shot_prompt)
    {
        set_nn_log_file("prompt.log");
        Corpus corpus{m_filters};
        size_t _num_layers = 2; // overriden when loaded from file
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(_num_layers, corpus, stats);


        std::println("Loading '{}'...", filename);
        if (! nn->load(filename))
        {
            std::println("Failed to load model from file: '{}'", filename);
            return;
        }
        std::println("Loaded.");

        PromptOptions options;

        if (one_shot_prompt.has_value())
        {
            process_line(*one_shot_prompt, corpus, *nn, options);
        }
        else
        {
            std::string line;
            while (true)
            {
                std::println("Enter input (or '/exit' to quit): ");
                std::print("> ");
                if (!std::getline(std::cin, line))
                {
                    std::println("Exiting prompt mode.");
                    break;
                }
                process_line(line, corpus, *nn, options);
            }
        }
    }

    void RLLM::process_line(const std::string& line, Corpus& corpus, NeuralNetwork& nn, PromptOptions& options)
    {
        if (line.starts_with("/") || line.empty())
        {
            process_command(line, options, nn);
            return;
        }

        auto token_id_list = corpus.get_token_ids(line);
        const auto full_string_opt = corpus.get_line(token_id_list);
        if (!full_string_opt.has_value())
        {
            std::println("Input contains unknown tokens. Please try again.");
            return;
        }
        std::println("Input tokens: {}", *full_string_opt);

        for (size_t iter = 0; iter < MAX_NUM_ANSWER_TOKENS; ++iter)
        {
            nn.propagate_forward(token_id_list);

            const auto output_token_id_lists = nn.get_best_output_token_ids(5, rllm::MultiTokenPredictionIndex::START);
            if (output_token_id_lists.empty())
            {
                std::println("No output tokens predicted.");
                break;
            }

            int ix = 0;
            int num_valid_tokens = 0;
            float top_k_probability_sum = 0.0f;
            for (const auto& entry : output_token_id_lists)
                top_k_probability_sum += entry.activation;

            for (const auto& entry : output_token_id_lists)
            {
                auto tok = nn.get_corpus().get_token_from_id(entry.token_id);
                if (tok == "\n")
                    tok = "\\n";
                const float global_percent = entry.activation * 100.0f;
                const float top_k_percent = (top_k_probability_sum > 0.0f)
                    ? (entry.activation / top_k_probability_sum) * 100.0f
                    : 0.0f;

                if (global_percent >= 0.01f)
                {
                    std::println(
                        "\t     prediction[{}]: '{}' (id: '{}'), {:.6f}% (top-5 {:.2f}%)",
                        ix,
                        tok,
                        static_cast<int>(entry.token_id),
                        global_percent,
                        top_k_percent
                    );
                }
                else
                {
                    std::println(
                        "\t     prediction[{}]: '{}' (id: '{}'), {:.3e}% (top-5 {:.2f}%)",
                        ix,
                        tok,
                        static_cast<int>(entry.token_id),
                        global_percent,
                        top_k_percent
                    );
                }
                ix++;
                if (entry.activation > VALID_PREDICTION_THRESHOLD)
                    num_valid_tokens++;
            }

            if (num_valid_tokens == 0)
            {
                std::println("No output tokens predicted.");
                break;
            }

            size_t random_index = 0;
            if (!options.highest_prio_only)
                random_index = static_cast<size_t>(rand()) % num_valid_tokens;

            const auto& entry = output_token_id_lists[random_index];
            auto output_token = nn.get_corpus().get_token_from_id(entry.token_id);
            if (output_token == "<UNK>")
            {
                std::println("Predicted next token is unknown. Stopping generation.");
                break;
            }
            if (output_token == "\n")  output_token = "\\n";
            if (output_token == "\t")  output_token = "\\t";
            std::println("Predicted next token: {}", output_token);
            token_id_list.push_back(entry.token_id);
        }

        const auto full_answer_string_opt = corpus.get_line(token_id_list);
        if (!full_answer_string_opt.has_value())
        {
            std::println("Input contains unknown tokens. Please try again.");
            return;
        }
        std::println("Full answer string: {}", *full_answer_string_opt);
    }

    void RLLM::train_mode(
        const std::optional<std::string>& input_filename,
        const std::string& output_filename,
        size_t num_layers,
        bool verbose,
        TrainingMethod method,
        std::optional<std::chrono::seconds> checkpointing_interval,
        int window_size,
        size_t num_epochs,
        const std::string& train_corpus_dir
    )
    {
        std::println("Training mode");
        set_nn_log_file("train.log");

        Corpus corpus{m_filters};
        corpus.load_files_from_dir(train_corpus_dir);
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(num_layers, corpus, stats);
        nn->set_training_method(method);
        nn->set_window_size(window_size);

        nn->train(verbose, num_epochs, input_filename, checkpointing_interval);

        stats.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm