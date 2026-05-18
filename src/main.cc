#include <RLLM.hpp>

#include <cstring>

int main(int argc, char* argv[])
{
    rllm::RLLM llm;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--train") == 0) {
            llm.train_mode();
            return 0;
        }
    }

    llm.prompt_mode();
    return 0;
}