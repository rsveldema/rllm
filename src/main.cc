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
                "  --train         Run in training mode (default is prompt mode)\n"
                "  --file <filename>  Specify the model file to load/save (default is '{}')\n"
                "  --verbose       Enable verbose output\n"
                "  --filter <filter>  Specify a filter to apply",
                argv[0],
                filename
            );
            return 1;
        }
    }

    rllm::RLLM llm(filters);
    if (train_mode)
    {
        llm.train_mode(filename, num_layers, verbose);
    }
    else
    {
        llm.prompt_mode(filename);
    }

    return 0;
}