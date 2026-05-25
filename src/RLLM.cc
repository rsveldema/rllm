#include <RLLM.hpp>

#include <iostream>
#include <print>
#include <string>

namespace rllm
{
    // maximum number of tokens to generate in response to a prompt before stopping
    static constexpr size_t MAX_NUM_ANSWER_TOKENS = 10;

    // Threshold for considering a predicted token as valid (not just noise).
    // This is a tunable hyperparameter.
    static constexpr float VALID_PREDICTION_THRESHOLD = 10.0f / 100.0f;

    struct PromptOptions
    {
        bool highest_prio_only = true;
    };




    void process_command(const std::string& _command, PromptOptions& options)
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
                 std::println("Number of token types in corpus: {}", static_cast<size_t>(TokenID::MAX));
             }},
            {{"/toggle_prio"}, "Toggle highest priority only mode", [&]() {
                 options.highest_prio_only = !options.highest_prio_only;
                 std::println(
                     "Toggled highest priority only mode. Now highest_prio_only is {}.", options.highest_prio_only
                 );
             }}
        };

        for (const auto& cmd : commands)
        {
            if (std::find(cmd.name.begin(), cmd.name.end(), command) != cmd.name.end())
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

    void RLLM::prompt_mode(const std::string& filename)
    {
        set_nn_log_file("prompt.log");
        Corpus corpus{m_filters};
        size_t _num_layers = 2; // overriden when loaded from file
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(_num_layers, corpus, stats);
        nn->load(filename);

        PromptOptions options;

        std::string line;
        while (true)
        {
            std::cout << "Enter input (or '/exit' to quit): ";
            if (!std::getline(std::cin, line) || line.starts_with("/") || line.empty())
            {
                process_command(line, options);
                continue;
            }

            // Process the input line
            auto token_id_list = corpus.get_token_ids(line);
            const auto full_string_opt = corpus.get_line(token_id_list);
            if (!full_string_opt.has_value())
            {
                std::println("Input contains unknown tokens. Please try again.");
                continue;
            }
            assert(full_string_opt.has_value());
            const auto& full_string = *full_string_opt;

            std::println("Input tokens: {}", full_string);

            const auto question_size = token_id_list.size();


            for (size_t iter = 0; iter < MAX_NUM_ANSWER_TOKENS; ++iter)
            {
                // Set the input layer of the neural network based on the token IDs
                // Propagate through the network to get the output
                nn->propagate_forward(token_id_list);

                // Get the output and convert it back to a token
                const auto output_token_id_lists = nn->get_best_output_token_ids(5);
                if (output_token_id_lists.empty())
                {
                    std::println("No output tokens predicted.");
                    break;
                }

                int ix = 0;
                int num_valid_tokens = 0;
                for (const auto& entry : output_token_id_lists)
                {
                    const auto predicted_token = nn->get_corpus().get_token_from_id(entry.token_id);
                    auto tok = nn->get_corpus().get_token_from_id(entry.token_id);
                    if (tok == "\n") {
                        tok = "\\n";
                    }
                    std::println(
                        "\t     prediction[{}]: '{}' (id: '{}'), {:.2f}%",
                        ix,
                        tok,
                        static_cast<int>(entry.token_id),
                        entry.activation * 100.0f
                    );
                    ix++;
                    if (entry.activation > VALID_PREDICTION_THRESHOLD)
                    {
                        num_valid_tokens++;
                    }
                }

                if (num_valid_tokens == 0)
                {
                    std::println("No output tokens predicted.");
                    break;
                }


                size_t random_index = 0;
                if (!options.highest_prio_only)
                {
                    random_index = static_cast<size_t>(rand()) % num_valid_tokens;
                }

                const auto& entry = output_token_id_lists[random_index];
                auto output_token = nn->get_corpus().get_token_from_id(entry.token_id);
                // Add the predicted token ID to the input for the
                // next iteration
                if (output_token == "<UNK>")
                {
                    std::println("Predicted next token is unknown. Stopping generation.");
                    break;
                }
                if (output_token == "\n") {
                    output_token = "\\n";
                }
                if (output_token == "\t") {
                    output_token = "\\t";
                }
                std::println("Predicted next token: {}", output_token);
                token_id_list.push_back(entry.token_id);
            }


            auto reply_id_list = token_id_list.substr(question_size);
            auto reply = corpus.get_line(reply_id_list);
            const auto full_answer_string_opt = corpus.get_line(token_id_list);
            if (!full_answer_string_opt.has_value())
            {
                std::println("Input contains unknown tokens. Please try again.");
                continue;
            }
            assert(full_answer_string_opt.has_value());
            const auto& full_answer_string = *full_answer_string_opt;
            std::println("Full answer string: {}", full_answer_string);
        }
    }

    void RLLM::train_mode(
        const std::optional<std::string>& input_filename,
        const std::string& output_filename,
        size_t num_layers,
        bool verbose,
        TrainingMethod method,
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

        nn->train(verbose, num_epochs, input_filename);

        stats.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm