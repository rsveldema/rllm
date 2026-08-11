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
#include <fstream>
#include <optional>
#include <ranges>
#include <vector>

#include <nlohmann/json.hpp>


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

    std::vector<std::string> train_corpus_dirs;
    std::vector<std::string> filters;
    bool train_mode = false;
    std::string output_filename = "model.json";
    std::optional<std::string> input_filename;
    std::optional<std::string> one_shot_prompt;
    int num_layers = 4;
    bool verbose = false;
    size_t num_epochs = 1000;
    rllm::TrainingMethod method = rllm::TrainingMethod::WINDOW;
    int window_size = 4;
    size_t window_stride = 1;
    std::optional<std::chrono::seconds> checkpointing_interval = std::chrono::seconds{120};
    std::chrono::seconds validation_interval{1800};
    std::string executable_name = "./rllm";
    size_t mtp_heads = 1;
    size_t learn_depth = rllm::TextTrainer::DEFAULT_LEARN_DEPTH;
    float learning_rate = rllm::TextTrainer::DEFAULT_LEARNING_RATE;
    float layer_learning_rate_multiplier = rllm::DEFAULT_DEPTH_LEARNING_RATE_MULTIPLIER;
    float warmup_percent = rllm::LoweringLearningRate::DEFAULT_WARMUP_PERCENT;
    rllm::LearningRateSchedule learning_rate_schedule = rllm::LearningRateSchedule::Lowering;
    float simulated_annealing_decay_factor = 0.8f;
    float simulated_annealing_initial_multiplier = 50.0f;
    size_t simulated_annealing_decay_epochs = 2;
    float simulated_annealing_min_multiplier = rllm::SimulatedAnnealingLearningRate::DEFAULT_MIN_MULTIPLIER;
    rllm::WeightInitializerType weight_initializer = rllm::WeightInitializerType::XavierInputProjections;
    rllm::FFNInitializerType ffn_initializer = rllm::FFNInitializerType::XavierInputProjections;
    rllm::EmbeddingInitializerType embedding_initializer = rllm::EmbeddingInitializerType::LegacyUniform;
    size_t micro_batch_size = 1;
    std::optional<size_t> epoch_size;
    size_t max_validation_windows = 4096;
    size_t validation_worst_count = 5;
    bool disable_early_stopping = false;
    bool disable_example_convergence = false;
    bool disable_training_diagnostics = false;
    bool reset_optimizer_state = false;
    bool restart_learning_rate_schedule = false;
    bool upgrade_mode = false;
    bool freeze_old_blocks = false;
    bool all_blocks_read_write = false;
    bool nan_finding_mode = false;
    std::optional<std::string> vulkan_device;

    void load_training_parameters(const std::string& filename)
    {
        const auto fail = [&](const std::string& message) {
            std::println("Error loading training parameters '{}': {}", filename, message);
            std::exit(1);
        };
        std::ifstream file{filename};
        if (!file)
            fail("cannot open file");
        const auto j = nlohmann::json::parse(file);
        if (j.value("version", 0) != 1)
            fail("unsupported version");

        num_layers = j.value("layers", num_layers);
        window_size = j.value("window_size", window_size);
        window_stride = j.value("window_stride", window_stride);
        learn_depth = j.value("learn_depth", learn_depth);
        learning_rate = j.value("learning_rate", learning_rate);
        layer_learning_rate_multiplier = j.value("layer_learning_rate_multiplier", layer_learning_rate_multiplier);
        if (!std::isfinite(layer_learning_rate_multiplier) ||
            layer_learning_rate_multiplier < 1.0f || layer_learning_rate_multiplier >= 2.0f)
            fail("layer_learning_rate_multiplier must be in [1, 2)");
        warmup_percent = j.value("warmup_percent", warmup_percent);
        if (!std::isfinite(warmup_percent) || warmup_percent <= 0.0f || warmup_percent > 100.0f)
            fail("warmup_percent must be in (0, 100]");
        simulated_annealing_decay_factor = j.value("simulated_annealing_decay_factor", simulated_annealing_decay_factor);
        simulated_annealing_initial_multiplier = j.value("simulated_annealing_initial_multiplier", simulated_annealing_initial_multiplier);
        simulated_annealing_decay_epochs = j.value("simulated_annealing_decay_epochs", simulated_annealing_decay_epochs);
        simulated_annealing_min_multiplier = j.value("simulated_annealing_min_multiplier", simulated_annealing_min_multiplier);
        micro_batch_size = j.value("micro_batch_size", micro_batch_size);
        if (micro_batch_size == 0 || micro_batch_size > static_cast<size_t>(rllm::BatchIndex::MAX))
            fail(std::format(
                "micro_batch_size must be between 1 and {}",
                static_cast<size_t>(rllm::BatchIndex::MAX)));
        num_epochs = j.value("epochs", num_epochs);
        if (j.contains("train_corpus_dirs"))
            train_corpus_dirs = j["train_corpus_dirs"].get<std::vector<std::string>>();
        else if (j.contains("train_corpus_dir"))
            train_corpus_dirs = {j["train_corpus_dir"].get<std::string>()};
        filters = j.value("filters", filters);
        if (j.contains("epoch_size"))
            epoch_size = j["epoch_size"].is_null() ? std::nullopt : std::optional<size_t>{j["epoch_size"].get<size_t>()};
        max_validation_windows = j.value("max_validation_windows", max_validation_windows);
        if (max_validation_windows == 0)
            fail("max_validation_windows must be positive");
        validation_worst_count = j.value("validation_worst_count", validation_worst_count);
        if (validation_worst_count == 0)
            fail("validation_worst_count must be positive");
        disable_early_stopping = j.value("disable_early_stopping", disable_early_stopping);
        disable_example_convergence = j.value("disable_example_convergence", disable_example_convergence);
        disable_training_diagnostics = j.value("disable_training_diagnostics", disable_training_diagnostics);
        reset_optimizer_state = j.value("reset_optimizer_state", reset_optimizer_state);
        restart_learning_rate_schedule = j.value("restart_learning_rate_schedule", restart_learning_rate_schedule);
        upgrade_mode = j.value("upgrade_mode", upgrade_mode);
        freeze_old_blocks = j.value("freeze_old_blocks", freeze_old_blocks);
        all_blocks_read_write = j.value("all_blocks_read_write", all_blocks_read_write);
        if (j.contains("checkpoint_interval_seconds"))
            checkpointing_interval = j["checkpoint_interval_seconds"].is_null() ? std::nullopt :
                std::optional<std::chrono::seconds>{std::chrono::seconds{j["checkpoint_interval_seconds"].get<long long>()}};
        if (j.contains("validation_interval_seconds"))
        {
            const auto seconds = j["validation_interval_seconds"].get<long long>();
            if (seconds <= 0)
                fail("validation_interval_seconds must be positive");
            validation_interval = std::chrono::seconds{seconds};
        }

        const auto method_name = j.value("method", "window");
        if (method_name == "window") method = rllm::TrainingMethod::WINDOW;
        else if (method_name == "reverse_window") method = rllm::TrainingMethod::REVERSE_WINDOW;
        else fail(std::format("unknown training method '{}'", method_name));

        const auto schedule = j.value("learning_rate_schedule", "lowering");
        if (schedule == "constant") learning_rate_schedule = rllm::LearningRateSchedule::Constant;
        else if (schedule == "lowering") learning_rate_schedule = rllm::LearningRateSchedule::Lowering;
        else if (schedule == "simulated_annealing") learning_rate_schedule = rllm::LearningRateSchedule::SimulatedAnnealing;
        else fail(std::format("unknown learning-rate schedule '{}'", schedule));

        const auto weight = j.value("weight_initializer", "xavier-input-projections");
        weight_initializer = weight == "xavier-uniform" ? rllm::WeightInitializerType::XavierUniform :
            weight == "legacy-uniform" ? rllm::WeightInitializerType::LegacyUniform : rllm::WeightInitializerType::XavierInputProjections;
        const auto ffn = j.value("ffn_initializer", "xavier-input-projections");
        ffn_initializer = ffn == "xavier-uniform" ? rllm::FFNInitializerType::XavierUniform :
            ffn == "legacy-uniform" ? rllm::FFNInitializerType::LegacyUniform : rllm::FFNInitializerType::XavierInputProjections;
        embedding_initializer = j.value("embedding_initializer", "legacy-uniform") == "variance-scaled-uniform" ?
            rllm::EmbeddingInitializerType::VarianceScaledUniform : rllm::EmbeddingInitializerType::LegacyUniform;
        std::println("Loaded training parameters '{}'", filename);
    }

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
        {.options = {"--training-parameters"},
         .description = "Load archived training settings from a checkpoint JSON; later options override them",
         .required_args = 1,
         .action = [&](const std::vector<std::string>& args) { load_training_parameters(args[0]); }},
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
        {.options = {"--upgrade-mode"},
         .description = "Increase a checkpoint's transformer-layer count",
         .action = [&](const std::vector<std::string>&) { upgrade_mode = true; }},
        {.options = {"--freeze-old-blocks"},
         .description = "With --upgrade-mode, keep the original transformer blocks read-only",
         .action = [&](const std::vector<std::string>&) {
             freeze_old_blocks = true;
             all_blocks_read_write = false;
         }},
        {.options = {"--all-blocks-read-write"},
         .description = "Clear a checkpoint's frozen boundary and train the complete model",
         .action = [&](const std::vector<std::string>&) {
             all_blocks_read_write = true;
             freeze_old_blocks = false;
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
        {.options = {"--validation-interval"},
         .description = "Timed validation cadence in seconds (default: 1800)",
         .required_args = 1,
         .action = [&](const std::vector<std::string>& args) {
             const int seconds = std::atoi(args[0].c_str());
             if (seconds <= 0)
             {
                 std::println("--validation-interval requires a positive number of seconds, got '{}'", args[0]);
                 std::exit(1);
             }
             validation_interval = std::chrono::seconds{seconds};
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
        {.options = {"--window-stride"},
         .description = "Token positions between supervised rows inside each window (default: 1)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const int n = std::atoi(args[0].c_str());
                 if (n <= 0)
                 {
                     std::println("--window-stride requires a positive integer, got '{}'", args[0]);
                     std::exit(1);
                 }
                 window_stride = static_cast<size_t>(n);
             }},
        {.options = {"--learning-rate"},
         .description = std::format("Base AdamW learning rate (default: {})", learning_rate),
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
        {.options = {"--layer-learning-rate-multiplier"},
         .description = "Output-side depth learning-rate multiplier in [1, 2); input side uses 2-M (default: 1.05)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const float multiplier = std::strtof(args[0].c_str(), &end);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE ||
                     !std::isfinite(multiplier) || multiplier < 1.0f || multiplier >= 2.0f)
                 {
                     std::println("--layer-learning-rate-multiplier requires a number in [1, 2), got '{}'", args[0]);
                     std::exit(1);
                 }
                 layer_learning_rate_multiplier = multiplier;
             }},
        {.options = {"--warmup-percent"},
         .description = "Lowering-schedule warmup as a percentage of planned epochs (default: 5)",
         .required_args = 1,
         .action = [&](const std::vector<std::string>& args) {
             char* end = nullptr;
             errno = 0;
             const float percent = std::strtof(args[0].c_str(), &end);
             if (end == args[0].c_str() || *end != '\0' || errno == ERANGE ||
                 !std::isfinite(percent) || percent <= 0.0f || percent > 100.0f)
             {
                 std::println("--warmup-percent requires a number in (0, 100], got '{}'", args[0]);
                 std::exit(1);
             }
             warmup_percent = percent;
         }},
        {.options = {"--weight-initializer"},
         .description = "Weight initializer: xavier-uniform, xavier-input-projections, or legacy-uniform (default: xavier-input-projections)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 if (args[0] == "xavier-uniform")
                     weight_initializer = rllm::WeightInitializerType::XavierUniform;
                 else if (args[0] == "xavier-input-projections")
                     weight_initializer = rllm::WeightInitializerType::XavierInputProjections;
                 else if (args[0] == "legacy-uniform")
                     weight_initializer = rllm::WeightInitializerType::LegacyUniform;
                 else
                 {
                     std::println("--weight-initializer must be 'xavier-uniform', 'xavier-input-projections', or 'legacy-uniform', got '{}'", args[0]);
                     std::exit(1);
                 }
             }},
        {.options = {"--ffn-initializer"},
         .description = "FFN initializer: xavier-uniform, xavier-input-projections, or legacy-uniform (default: xavier-input-projections)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 if (args[0] == "xavier-uniform")
                     ffn_initializer = rllm::FFNInitializerType::XavierUniform;
                 else if (args[0] == "xavier-input-projections")
                     ffn_initializer = rllm::FFNInitializerType::XavierInputProjections;
                 else if (args[0] == "legacy-uniform")
                     ffn_initializer = rllm::FFNInitializerType::LegacyUniform;
                 else
                 {
                     std::println("--ffn-initializer must be 'xavier-uniform', 'xavier-input-projections', or 'legacy-uniform', got '{}'", args[0]);
                     std::exit(1);
                 }
             }},
        {.options = {"--embedding-initializer"},
         .description = "Token embedding initializer: variance-scaled-uniform or legacy-uniform (default: legacy-uniform)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 if (args[0] == "variance-scaled-uniform")
                     embedding_initializer = rllm::EmbeddingInitializerType::VarianceScaledUniform;
                 else if (args[0] == "legacy-uniform")
                     embedding_initializer = rllm::EmbeddingInitializerType::LegacyUniform;
                 else
                 {
                     std::println("--embedding-initializer must be 'variance-scaled-uniform' or 'legacy-uniform', got '{}'", args[0]);
                     std::exit(1);
                 }
             }},
        {.options = {"--learning-rate-schedule"},
         .description = "Learning-rate schedule: constant, lowering, or simulated_annealing (default: lowering)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 if (args[0] == "constant")
                     learning_rate_schedule = rllm::LearningRateSchedule::Constant;
                 else if (args[0] == "lowering")
                     learning_rate_schedule = rllm::LearningRateSchedule::Lowering;
                 else if (args[0] == "simulated_annealing")
                     learning_rate_schedule = rllm::LearningRateSchedule::SimulatedAnnealing;
                 else
                 {
                     std::println("--learning-rate-schedule must be 'constant', 'lowering', or 'simulated_annealing', got '{}'", args[0]);
                     std::exit(1);
                 }
             }},
        {.options = {"--simulated-annealing-decay-factor"},
         .description = "Simulated-annealing learning-rate multiplier applied every two epochs (default: 0.8)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const float factor = std::strtof(args[0].c_str(), &end);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE ||
                     !std::isfinite(factor) || factor <= 0.0f || factor >= 1.0f)
                 {
                     std::println("--simulated-annealing-decay-factor requires a number between 0 and 1, got '{}'", args[0]);
                     std::exit(1);
                 }
                 simulated_annealing_decay_factor = factor;
             }},
        {.options = {"--simulated-annealing-initial-multiplier"},
         .description = "Initial simulated-annealing learning-rate multiplier (default: 50)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const float multiplier = std::strtof(args[0].c_str(), &end);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE ||
                     !std::isfinite(multiplier) || multiplier <= 0.0f)
                 {
                     std::println("--simulated-annealing-initial-multiplier requires a positive number, got '{}'", args[0]);
                     std::exit(1);
                 }
                 simulated_annealing_initial_multiplier = multiplier;
             }},
        {.options = {"--simulated-annealing-decay-epochs"},
         .description = "Epochs between simulated-annealing rate reductions (default: 2)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const unsigned long epochs = std::strtoul(args[0].c_str(), &end, 10);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE || epochs == 0)
                 {
                     std::println("--simulated-annealing-decay-epochs requires a positive integer, got '{}'", args[0]);
                     std::exit(1);
                 }
                 simulated_annealing_decay_epochs = static_cast<size_t>(epochs);
             }},
        {.options = {"--simulated-annealing-min-multiplier"},
         .description = "Minimum simulated-annealing rate as a base-rate multiplier (default: 0.02)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const float multiplier = std::strtof(args[0].c_str(), &end);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE ||
                     !std::isfinite(multiplier) || multiplier <= 0.0f)
                 {
                     std::println("--simulated-annealing-min-multiplier requires a positive number, got '{}'", args[0]);
                     std::exit(1);
                 }
                 simulated_annealing_min_multiplier = multiplier;
             }},
        {.options = {"--micro-batch-size"},
         .description = "Number of examples to group into one averaged gradient update (default: 1)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const unsigned long n = std::strtoul(args[0].c_str(), &end, 10);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE || n == 0 ||
                     n > static_cast<unsigned long>(rllm::BatchIndex::MAX))
                 {
                     std::println(
                         "--micro-batch-size requires an integer between 1 and {}, got '{}'",
                         static_cast<size_t>(rllm::BatchIndex::MAX), args[0]);
                     std::exit(1);
                 }
                 micro_batch_size = static_cast<size_t>(n);
             }},
        {.options = {"--max-validation-windows"},
         .description = "Maximum number of deterministic held-out windows to validate (default: 4096)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const unsigned long n = std::strtoul(args[0].c_str(), &end, 10);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE || n == 0)
                 {
                     std::println("--max-validation-windows requires a positive integer, got '{}'", args[0]);
                     std::exit(1);
                 }
                 max_validation_windows = static_cast<size_t>(n);
             }},
        {.options = {"--validation-worst-count"},
         .description = "Number of highest-loss validation predictions to log (default: 5)",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 char* end = nullptr;
                 errno = 0;
                 const unsigned long n = std::strtoul(args[0].c_str(), &end, 10);
                 if (end == args[0].c_str() || *end != '\0' || errno == ERANGE || n == 0)
                 {
                     std::println("--validation-worst-count requires a positive integer, got '{}'", args[0]);
                     std::exit(1);
                 }
                 validation_worst_count = static_cast<size_t>(n);
             }},
        {.options = {"--nan-finding"},
         .description = "Enable expensive NaN/range validation checks (default: disabled)",
         .action =
             [&](const std::vector<std::string>&) {
                 nan_finding_mode = true;
             }},
        {.options = {"--disable-early-stopping"},
         .description = "Continue through all requested epochs even when validation loss stops improving",
         .action =
             [&](const std::vector<std::string>&) {
                 disable_early_stopping = true;
             }},
        {.options = {"--disable-example-convergence"},
         .description = "Keep examples active after reaching the convergence threshold",
         .action =
             [&](const std::vector<std::string>&) {
                 disable_example_convergence = true;
             }},
        {.options = {"--disable-training-diagnostics"},
         .description = "Disable per-epoch optimizer and backward-pass diagnostics",
         .action =
             [&](const std::vector<std::string>&) {
                 disable_training_diagnostics = true;
             }},
        {.options = {"--reset-optimizer-state"},
         .description = "Resume model weights and training cursor but reset Adam moments and bias-correction step",
         .action =
             [&](const std::vector<std::string>&) {
                 reset_optimizer_state = true;
             }},
        {.options = {"--restart-learning-rate-schedule"},
         .description = "Resume model weights, optimizer state, and training cursor but restart the learning-rate schedule",
         .action =
             [&](const std::vector<std::string>&) {
                 restart_learning_rate_schedule = true;
             }},
        {.options = {"--vulkan-device"},
         .description = "Select a Vulkan device by a case-insensitive name substring",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 if (args[0].empty())
                 {
                     std::println("--vulkan-device requires a non-empty device name substring");
                     std::exit(1);
                 }
                 vulkan_device = args[0];
             }},
        {.options = {"--train-dir"},
         .description = "Training directory, optionally PATH:WEIGHT; repeat for a weighted source mixture",
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 train_corpus_dirs.push_back(args[0]);
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
             "Training method: window:<N> or reverse_window[:N], with metadata prefixes and all-position causal loss (N >= 4)"
         ),
         .required_args = 1,
         .action =
             [&](const std::vector<std::string>& args) {
                 const std::string m = args[0];
                 if (m.starts_with("window:"))
                 {
                     const int n = std::atoi(m.c_str() + 7);
                     if (n < 4)
                     {
                         std::println("window:<N> requires N >= 4, got '{}'", m);
                         std::exit(1);
                     }
                     method = rllm::TrainingMethod::WINDOW;
                     window_size = n;
                 }
                 else if (m == "reverse_window" || m.starts_with("reverse_window:"))
                 {
                     const int n = m == "reverse_window" ? window_size : std::atoi(m.c_str() + 15);
                     if (n < 4)
                     {
                         std::println("reverse_window:<N> requires N >= 4, got '{}'", m);
                         std::exit(1);
                     }
                     method = rllm::TrainingMethod::REVERSE_WINDOW;
                     window_size = n;
                 }
                 else
                 {
                     std::println(
                         "Unknown training method '{}'. Valid values: window:<N>, reverse_window[:N]",
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

        if (train_mode && train_corpus_dirs.empty())
        {
            std::println("Error: --train requires --train-dir <path>");
            print_usage_and_exit(argv[0], 1);
        }
    }
};

int main(int argc, char* argv[])
{
    std::srand(0);

    CommandLineParser parser;
    parser.parse(argc, argv);
    if (parser.vulkan_device)
        setenv("RLLM_VULKAN_DEVICE_SUBSTRING", parser.vulkan_device->c_str(), 1);

    parallel::init_parallel();

    // Generated Vulkan kernels are function-local statics and may destruct
    // during process teardown. Keep the session alive for the full process so
    // those destructors never see a dead VkDevice.
    auto* vulkan_session = new VulkanSession(
        false,
        parser.vulkan_device ? parser.vulkan_device->c_str() : nullptr
    );
    if (!vulkan_session->has_device())
        return 1;
    rllm::vulkan_runtime::set_session(*vulkan_session);
#ifdef NDEBUG
    std::println("Build type: Release (NDEBUG defined)");
#else
    std::println("Build type: Debug (NDEBUG not defined)");
#endif

    std::println("Offload type: Vulkan");
    parallel::print_vulkan_provider();
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
            parser.validation_interval,
            parser.window_size,
            parser.window_stride,
            parser.learn_depth,
            parser.learning_rate,
            parser.layer_learning_rate_multiplier,
            parser.warmup_percent,
            parser.learning_rate_schedule,
            parser.simulated_annealing_decay_factor,
            parser.simulated_annealing_initial_multiplier,
            parser.simulated_annealing_decay_epochs,
            parser.simulated_annealing_min_multiplier,
            parser.weight_initializer,
            parser.ffn_initializer,
            parser.embedding_initializer,
            parser.micro_batch_size,
            parser.num_epochs,
            parser.epoch_size,
            parser.max_validation_windows,
            parser.validation_worst_count,
            parser.disable_early_stopping,
            parser.disable_example_convergence,
            parser.disable_training_diagnostics,
            parser.reset_optimizer_state,
            parser.restart_learning_rate_schedule,
            parser.upgrade_mode,
            parser.freeze_old_blocks,
            parser.all_blocks_read_write,
            parser.train_corpus_dirs
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
