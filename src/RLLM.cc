#include <RLLM.hpp>

#include <iostream>
#include <print>
#include <string>

const size_t num_layers = 3; // Example number of layers

namespace rllm
{
    RLLM::RLLM()
    {
        // Constructor implementation
    }

    void RLLM::prompt_mode(const std::string& filename)
    {
        Corpus corpus;

        NeuralNetwork nn(num_layers);
        nn.load(filename);

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

            static constexpr size_t MAX_NUM_ANSWER_TOKENS = 10;

            for (int iter = 0; iter < MAX_NUM_ANSWER_TOKENS; ++iter)
            {
                // Set the input layer of the neural network based on the token IDs
                nn.set_input_layer(token_id_list);

                // Propagate through the network to get the output
                nn.propagate_forward();

                // Get the output and convert it back to a token
                const auto output_token_id_lists = nn.get_best_output_token_ids(5, corpus);
                if (output_token_id_lists.empty())
                {
                    std::println("No output tokens predicted.");
                    break;
                }
                const auto random_index = static_cast<size_t>(rand()) % output_token_id_lists.size();
                const auto output_token_id = output_token_id_lists[random_index].token_id;
                const auto output_token = corpus.get_token_from_id(output_token_id);

                std::println("Predicted next token: {}", output_token);

                // Add the predicted token ID to the input for the
                // next iteration
                token_id_list.push_back(output_token_id);
            }
        }
    }

    void RLLM::train_mode(const std::string& filename)
    {
        std::println("Training mode");

        Corpus corpus;

        NeuralNetwork nn(num_layers);

        nn.train(corpus);

        nn.save(filename);
    }
} // namespace rllm