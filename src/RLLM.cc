#include <RLLM.hpp>

#include <iostream>
#include <print>
#include <string>

namespace rllm
{
    RLLM::RLLM()
    {
        // Constructor implementation
    }

    void RLLM::prompt_mode(const std::string& filename)
    {
        Corpus corpus;
        size_t _num_layers = 2; // overriden when loaded from file
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(_num_layers, corpus, stats);
        nn->load(filename);

        std::string line;
        while (true)
        {
            std::cout << "Enter input (or 'exit' to quit): ";
            if (!std::getline(std::cin, line) || line == "exit")
            {
                break;
            }
            // Process the input line
            auto token_id_list = corpus.get_token_ids(line);
            const auto full_string_opt = corpus.get_line(token_id_list);
            if (! full_string_opt.has_value())
            {
                std::println("Input contains unknown tokens. Please try again.");
                continue;
            }
            assert(full_string_opt.has_value());
            const auto& full_string = *full_string_opt;

            std::println("Input tokens: {}", full_string);

            static constexpr size_t MAX_NUM_ANSWER_TOKENS = 10;

            for (size_t iter = 0; iter < MAX_NUM_ANSWER_TOKENS; ++iter)
            {
                // Set the input layer of the neural network based on the token IDs
                nn->set_input_layer(token_id_list);

                // Propagate through the network to get the output
                nn->propagate_forward();

                // Get the output and convert it back to a token
                const auto output_token_id_lists = nn->get_best_output_token_ids(5);
                if (output_token_id_lists.empty())
                {
                    std::println("No output tokens predicted.");
                    break;
                }

                int ix = 0;
                for (const auto& entry : output_token_id_lists)
                {
                    const auto predicted_token = nn->get_corpus().get_token_from_id(entry.token_id);
                    std::println(
                        "\t     prediction[{}]: '{}' (id: '{}'), {}",
                        ix,
                        nn->get_corpus().get_token_from_id(entry.token_id),
                        static_cast<int>(entry.token_id),
                        entry.activation
                    );
                    ix++;
                }

                const auto random_index = static_cast<size_t>(rand()) % output_token_id_lists.size();
                const auto output_token_id = output_token_id_lists[random_index].token_id;
                const auto output_token = nn->get_corpus().get_token_from_id(output_token_id);

                std::println("Predicted next token: {}", output_token);

                // Add the predicted token ID to the input for the
                // next iteration
                token_id_list.push_back(output_token_id);
            }
        }
    }

    void RLLM::train_mode(const std::string& filename, size_t num_layers, bool verbose)
    {
        std::println("Training mode");

        Corpus corpus;
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(num_layers, corpus, stats);

        nn->train(verbose);

        stats.print_statistics();

        nn->save(filename);
    }
} // namespace rllm