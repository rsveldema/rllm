#include <print>
#include <string>

#include <RLLM.hpp>

#include <cstring>

int main(int argc, char* argv[])
{
    rllm::RLLM llm;
    bool train_mode = false;
    const char* filename = "model.dat";

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--file") == 0 && i + 1 < argc)
        {
            filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "--train") == 0)
        {
            train_mode = true;
        }
        else
        {
            std::println(
                ""
                "Usage: {} [--train] [--file <filename>]\n"
                "  --train         Run in training mode (default is prompt mode)\n"
                "  --file <filename>  Specify the model file to load/save (default is '{}')",
                argv[0],
                filename
            );
            return 1;
        }
    }

    if (train_mode)
    {
        llm.train_mode(filename);
    }
    else
    {
        llm.prompt_mode(filename);
    }

    return 0;
}