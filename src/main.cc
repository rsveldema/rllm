#include <print>
#include <string>

#include <RLLM.hpp>

#include <cstdlib>
#include <cstring>
#include <ctime>



int main(int argc, char* argv[])
{
    std::srand(0);
    std::vector<std::string> filters;
    bool train_mode = false;
    const char* filename = "model.json";
    int num_layers = 4;
    bool verbose = false;
    rllm::NeuralNetwork::TrainingMethod method = rllm::NeuralNetwork::TrainingMethod::TWO_TOK;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--filter") == 0 && ((i + 1) < argc))
        {
            filters.push_back(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--layers") == 0 && ((i + 1) < argc))
        {
            num_layers = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--file") == 0 && ((i + 1) < argc))
        {
            filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "--method") == 0 && ((i + 1) < argc))
        {
            const std::string m = argv[++i];
            if (m == "two_tok")
                method = rllm::NeuralNetwork::TrainingMethod::TWO_TOK;
            else if (m == "three_tok")
                method = rllm::NeuralNetwork::TrainingMethod::THREE_TOK;
            else if (m == "increasingly_longer")
                method = rllm::NeuralNetwork::TrainingMethod::INCREASINGLY_LONGER_SEQUENCES;
            else if (m == "window2")
                method = rllm::NeuralNetwork::TrainingMethod::WINDOW2;
            else if (m == "window3")
                method = rllm::NeuralNetwork::TrainingMethod::WINDOW3;
            else
            {
                std::println("Unknown training method '{}'. Valid values: two_tok, three_tok, increasingly_longer, window2, window3", m);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--train") == 0)
        {
            train_mode = true;
        }
        else if (std::strcmp(argv[i], "--verbose") == 0)
        {
            verbose = true;
        }
        else
        {
            std::println(
                ""
                "Usage: {} [--train] [--file <filename>] [--verbose] [--filter <filter>]\n"
                "          [--method <two_tok|three_tok|increasingly_longer|window2|window3>]\n"
                "  --train         Run in training mode (default is prompt mode)\n"
                "  --file <filename>  Specify the model file to load/save (default is '{}')\n"
                "  --verbose       Enable verbose output\n"
                "  --filter <filter>  Specify a filter to apply\n"
                "  --method        Training method (default: two_tok)",
                argv[0],
                filename
            );
            return 1;
        }
    }

    rllm::RLLM llm(filters);
    if (train_mode)
    {
        llm.train_mode(filename, num_layers, verbose, method);
    }
    else
    {
        llm.prompt_mode(filename);
    }

    return 0;
}