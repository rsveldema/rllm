#include <print>
#include <string>

#include <parallel.hpp>
#include <RLLM.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>

int main(int argc, char* argv[])
{
    std::srand(0);
    parallel::init_parallel();
#ifdef NDEBUG
    std::println("Build type: Release (NDEBUG defined)");
#else
    std::println("Build type: Debug (NDEBUG not defined)");
#endif
    std::string train_corpus_dir = "training_data";
    std::vector<std::string> filters;
    bool train_mode = false;
    std::string output_filename = "model.json";
    std::optional<std::string> input_filename;
    std::optional<std::string> one_shot_prompt;
    int num_layers = 4;
    bool verbose = false;
    size_t num_epochs = 1000;
    auto method = rllm::TrainingMethod::TWO_TOK;
    int window_size = 2;
    std::optional<std::chrono::seconds> checkpointing_interval = std::chrono::seconds{120};

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
        else if (std::strcmp(argv[i], "--checkpoint-interval") == 0 && ((i + 1) < argc))
        {
            const int seconds = std::atoi(argv[++i]);
            if (seconds <= 0)
                checkpointing_interval = std::nullopt;
            else
                checkpointing_interval = std::chrono::seconds{seconds};
        }
        else if (std::strcmp(argv[i], "--train-dir") == 0 && ((i + 1) < argc))
        {
            train_corpus_dir = argv[++i];
        }
        else if (std::strcmp(argv[i], "--epochs") == 0 && ((i + 1) < argc))
        {
            num_epochs = static_cast<size_t>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "-o") == 0 && ((i + 1) < argc))
        {
            output_filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "-i") == 0 && ((i + 1) < argc))
        {
            input_filename = argv[++i];
        }        else if (std::strcmp(argv[i], "-c") == 0 && ((i + 1) < argc))
        {
            one_shot_prompt = argv[++i];
        }        else if (std::strcmp(argv[i], "--method") == 0 && ((i + 1) < argc))
        {
            const std::string m = argv[++i];
            if (m == "two_tok")
                method = rllm::TrainingMethod::TWO_TOK;
            else if (m == "three_tok")
                method = rllm::TrainingMethod::THREE_TOK;
            else if (m == "increasingly_longer")
                method = rllm::TrainingMethod::INCREASINGLY_LONGER_SEQUENCES;
            else if (m == "random_line_random_len")
                method = rllm::TrainingMethod::RANDOM_LINE_RANDOM_LEN;
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
                std::println(
                    "Unknown training method '{}'. Valid values: two_tok, three_tok, increasingly_longer, random_line_random_len, window:<N>", m
                );
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
                "          [--train-dir <directory>]\n"
                "          [--method <two_tok|three_tok|increasingly_longer|random_line_random_len|window:<N>>]\n"
                "  --train         Run in training mode (default is prompt mode)\n"
                "  --train-dir <directory>  Directory containing training text files (default is '{}')\n"
                "  -i <filename>  Specify the model file to load (trainer will init the model if not provided)\n"
                "  -c <string>    Run prompt mode with this string, print predictions, then exit\n"
                "  -o <filename>  Specify the model file to save (default is '{}')\n"
                "  --verbose       Enable verbose output\n"
                "  --filter <filter>  Specify a filter to apply\n"
                "  --epochs <n>    Number of training epochs (default: {})\n"
                "  --method        Training method (default: {})\n"
                "  --checkpoint-interval <seconds>  Extra timed checkpoint cadence; <=0 disables (default: {}s)\n"
                "  window:<N>      Sliding window of N tokens (N >= 2)",
                argv[0],
                train_corpus_dir,
                output_filename,
                num_epochs,
                rllm::training_method_to_string(method),
                checkpointing_interval.has_value() ? std::to_string(checkpointing_interval->count()) : "disabled"
            );
            return 1;
        }
    }

    rllm::RLLM llm(filters);
    if (train_mode)
    {
        llm.train_mode(
            input_filename,
            output_filename,
            num_layers,
            verbose,
            method,
            checkpointing_interval,
            window_size,
            num_epochs,
            train_corpus_dir
        );
    }
    else
    {
        llm.prompt_mode(input_filename ? *input_filename : output_filename, one_shot_prompt);
    }

    return 0;
}