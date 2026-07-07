#include <print>
#include <string>

#include <Prompter.hpp>
#include <RuntimeConfig.hpp>
#include <Trainer.hpp>
#include <parallel.hpp>
#include <rllm_vulkan_runtime.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <optional>
#include <ranges>
#include <vector>


struct CommandLineOption
{
    // if the command line has one of these:
    std::vector<std::string> options;
    std::string description;
    size_t required_args = 0;

    // then call the associated action with the remaining command line arguments
    std::function<void(const std::vector<std::string>&)> action;
};


struct CommandLineParser
{

    std::optional<std::string> train_corpus_dir;
    std::vector<std::string> filters;
    bool train_mode = false;
    std::string output_filename = "model.json";
    std::optional<std::string> input_filename;
    std::optional<std::string> one_shot_prompt;
    int num_layers = 4;
    bool verbose = false;
    size_t num_epochs = 1000;
    rllm::TrainingMethod method = rllm::TrainingMethod::TWO_TOK;
    int window_size = 2;
    std::optional<std::chrono::seconds> checkpointing_interval = std::chrono::seconds{120};
    std::string executable_name = "./rllm";
    size_t mtp_heads = 1;
    size_t learn_depth = rllm::NeuralNetwork::DEFAULT_LEARN_DEPTH;
    float learning_rate = rllm::NeuralNetwork::DEFAULT_LEARNING_RATE;
    std::optional<size_t> epoch_size;
    bool nan_finding_mode = false;

    static bool option_matches(const std::string& arg, const CommandLineOption& option)
    {
        return std::ranges::any_of(option.options, [&](const auto& name) {
            return name == arg;
        });
    }

    [[noreturn]] void print_usage_and_exit(const std::string& executable, int exit_code) const
    {
        std::println("Usage: {} ", executable);

        for (const auto& option : command_line_options)
        {
            std::string option_names;
            for (size_t i = 0; i < option.options.size(); ++i)
            {
                if (i > 0)
                    option_names += ", ";
                option_names += option.options[i];
            }

            std::println("  {:<28} {}", option_names, option.description);
        }

        std::exit(exit_code);
    }

    std::vector<CommandLineOption> command_line_options = {
        {.options = {"--filter"},
         .description = "Specify a filter to apply (can be used multiple times for multiple filters)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 filters.push_back(args[0]);
             }},
        {.options = {"--layers"},
         .description = std::format("Specify the number of layers in the model (default: {})", num_layers),
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 num_layers = std::atoi(args[0].c_str());
             }},
        {.options = {"--checkpoint-interval"},
         .description = "Extra timed checkpoint cadence; <=0 disables",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const int seconds = std::atoi(args[0].c_str());
                 if (seconds <= 0)
                     checkpointing_interval = std::nullopt;
                 else
                     checkpointing_interval = std::chrono::seconds{seconds};
             }},
        {.options = {"--learn-depth"},
         .description = std::format("Gradient-update passes per training example (default: {})", learn_depth),
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const int n = std::atoi(args[0].c_str());
                 if (n <= 0)
                 {
                     std::println("--learn-depth requires a positive integer, got '{}'", args[0]);
                     std::exit(1);
                 }
                 learn_depth = static_cast<size_t>(n);
             }},
        {.options = {"--learning-rate"},
         .description = std::format("Base learning rate before layer-count scaling (default: {})", learning_rate),
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const float rate = std::strtof(args[0].c_str(), &end);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(rate) || rate <= 0.0f)
                 {
                     std::println("--learning-rate requires a positive number, got '{}'", args[0]);
                     std::exit(1);
                 }
                 learning_rate = rate;
             }},
        {.options = {"--nan-finding"},
         .description = "Enable expensive NaN/range validation checks (default: disabled)",
         .action =
             [&](const std::vector<std::string>&) {
                 nan_finding_mode = true;
             }},
        {.options = {"--train-dir"},
         .description = "Directory containing training text files",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 train_corpus_dir = args[0];
             }},
        {.options = {"--epochs"},
         .description = "Number of training epochs",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 num_epochs = static_cast<size_t>(std::atoi(args[0].c_str()));
             }},
        {.options = {"--epoch-size"},
         .description = "Number of training lines to visit per line-based epoch (default: all)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const int n = std::atoi(args[0].c_str());
                 if (n <= 0)
                 {
                     std::println("--epoch-size requires a positive integer, got '{}'", args[0]);
                     std::exit(1);
                 }
                 epoch_size = static_cast<size_t>(n);
             }},
        {.options = {"-o"},
         .description = "Specify the model file to save",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 output_filename = args[0];
             }},
        {.options = {"-i"},
         .description = "Specify the model file to load",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 input_filename = args[0];
             }},
        {.options = {"-c"},
         .description = "Run prompt mode with this string, print predictions, then exit",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 one_shot_prompt = args[0];
             }},
        {.options = {"--method"},
         .description = std::format(
             "Training method. Valid values: two_tok, three_tok, increasingly_longer, random_line_random_len, "
             "window:<N> Sliding window of N tokens (N >= 2)"
         ),
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const std::string m = args[0];
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
                         std::exit(1);
                     }
                     method = rllm::TrainingMethod::WINDOW;
                     window_size = n;
                 }
                 else
                 {
                     std::println(
                         "Unknown training method '{}'. Valid values: two_tok, three_tok, increasingly_longer, "
                         "random_line_random_len, window:<N>",
                         m
                     );
                     std::exit(1);
                 }
             }},
        {.options = {"--mtp-heads"},
         .description = "Number of MTP heads to use for token generation during prompting (default: 1, max: 4)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const int n = std::atoi(args[0].c_str());
                 if (n < 1 || n > 4)
                 {
                     std::println("--mtp-heads requires a value between 1 and 4, got '{}'", args[0]);
                     std::exit(1);
                 }
                 mtp_heads = static_cast<size_t>(n);
             }},
        {.options = {"--train"},
         .description = "Run in training mode",
         .action =
             [&](const std::vector<std::string>&) {
                 train_mode = true;
             }},
        {.options = {"--verbose"},
         .description = "Enable verbose output",
         .action =
             [&](const std::vector<std::string>&) {
                 verbose = true;
             }},
        {.options = {"--help", "-h"}, .description = "Show this help", .action = [&](const std::vector<std::string>&) {
             print_usage_and_exit(executable_name, 0);
         }}

    };

    void parse(int argc, char* argv[])
    {
        executable_name = argv[0];
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];

            const auto option_it = std::ranges::find_if(command_line_options, [&](const auto& option) {
                return option_matches(arg, option);
            });

            if (option_it == command_line_options.end())
            {
                std::println("Unknown option: {}", arg);
                print_usage_and_exit(argv[0], 1);
            }

            if ((i + static_cast<int>(option_it->required_args)) >= argc)
            {
                std::println("Error: {} requires {} argument(s)", arg, option_it->required_args);
                print_usage_and_exit(argv[0], 1);
            }

            std::vector<std::string> args;
            args.reserve(option_it->required_args);
            for (size_t n = 0; n < option_it->required_args; ++n)
            {
                args.emplace_back(argv[i + 1 + static_cast<int>(n)]);
            }

            option_it->action(args);
            i += static_cast<int>(option_it->required_args);
        }

        if (train_mode && !train_corpus_dir.has_value())
        {
            std::println("Error: --train requires --train-dir <path>");
            print_usage_and_exit(argv[0], 1);
        }
    }
};

int main(int argc, char* argv[])
{
    std::srand(0);
    parallel::init_parallel();
#ifdef NDEBUG
    std::println("Build type: Release (NDEBUG defined)");
#else
    std::println("Build type: Debug (NDEBUG not defined)");
#endif

    // Generated Vulkan kernels are function-local statics and may destruct
    // during process teardown. Keep the session alive for the full process so
    // those destructors never see a dead VkDevice.
    auto* vulkan_session = new VulkanSession();
    rllm::vulkan_runtime::set_session(*vulkan_session);
    std::println("Offload type: Vulkan");
    parallel::print_vulkan_provider();

    CommandLineParser parser;
    parser.parse(argc, argv);
    rllm::set_nan_finding_mode_enabled(parser.nan_finding_mode);

    if (parser.train_mode)
    {
        rllm::Trainer trainer(parser.filters);
        trainer.train_mode(
            parser.input_filename,
            parser.output_filename,
            parser.num_layers,
            parser.verbose,
            parser.method,
            parser.checkpointing_interval,
            parser.window_size,
            parser.learn_depth,
            parser.learning_rate,
            parser.num_epochs,
            parser.epoch_size,
            parser.train_corpus_dir.value()
        );
    }
    else
    {
        rllm::Prompter prompter(parser.filters);
        prompter.prompt_mode(
            parser.input_filename ? *parser.input_filename : parser.output_filename,
            parser.one_shot_prompt,
            parser.mtp_heads
        );
    }

    return 0;
}
