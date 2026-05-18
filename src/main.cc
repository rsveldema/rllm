#include <RLLM.hpp>

#include <cstring>

int main(int argc, char* argv[])
{
    rllm::RLLM predictor;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--train") == 0) {
            predictor.train_mode();
            return 0;
        }
    }

    predictor.prompt_mode();
    return 0;
}