#include <print>
#include <string>

#include <RLLM.hpp>

#include <cstdlib>
#include <cstring>
#include <ctime>

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
        case TrainingMethod::WINDOW:
            return "window";
        }
        return "UNKNOWN";
    }
}

int main(int argc, char* argv[])
{
    std::srand(0);
    std::vector<std::string> filters;
    bool train_mode = false;
    const char* filename = "model.json";
    int num_layers = 4;
    bool verbose = false;
    size_t num_epochs = 1000;
    auto method = rllm::TrainingMethod::TWO_TOK;
    int window_size = 2;

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
        else if (std::strcmp(argv[i], "--epochs") == 0 && ((i + 1) < argc))
        {
            num_epochs = static_cast<size_t>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--file") == 0 && ((i + 1) < argc))
        {
            filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "--method") == 0 && ((i + 1) < argc))
        {
            const std::string m = argv[++i];
            if (m == "two_tok")
                method = rllm::TrainingMethod::TWO_TOK;
            else if (m == "three_tok")
                method = rllm::TrainingMethod::THREE_TOK;
            else if (m == "increasingly_longer")
                method = rllm::TrainingMethod::INCREASINGLY_LONGER_SEQUENCES;
            else if (m.starts_with("window:"))
            {
                const int n = std::atoi(m.c_str() + 7);
                if (n < 2)
                {
                    std::println("window:<N> requires N >= 2, got '{}'", m);
                    return 1;
                }
                method = rllm::TrainingMethod::WINDOW;
                window_size = n;
            }
            else
            {
                std::println("Unknown training method '{}'. Valid values: two_tok, three_tok, increasingly_longer, window:<N>", m);
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
                "Usage: {} [--train] [--file <filename>] [--verbose] [--filter <filter>]\n"
                "          [--method <two_tok|three_tok|increasingly_longer|window:<N>>]\n"
                "  --train         Run in training mode (default is prompt mode)\n"
                "  --file <filename>  Specify the model file to load/save (default is '{}')\n"
                "  --verbose       Enable verbose output\n"
                "  --filter <filter>  Specify a filter to apply\n"
                "  --epochs <n>    Number of training epochs (default: {})\n"
                "  --method        Training method (default: {})\n"
                "  window:<N>      Sliding window of N tokens (N >= 2)",
                argv[0],
                filename,
                num_epochs,
                rllm::training_method_to_string(method)
            );
            return 1;
        }
    }

    rllm::RLLM llm(filters);
    if (train_mode)
    {
        llm.train_mode(filename, num_layers, verbose, method, window_size, num_epochs);
    }
    else
    {
        llm.prompt_mode(filename);
    }

    return 0;
}