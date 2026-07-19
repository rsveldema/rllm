#include <Trainer.hpp>

#include <algorithm>
#include <fstream>
#include <print>
#include <string>
#include <type_traits>

#include <nlohmann/json.hpp>

#include <vulkan_session.hpp>

namespace rllm
{
    Trainer::Trainer(const std::vector<std::string>& filters)
        : m_filters(filters)
    {}

    void Trainer::train_mode(
        const std::optional<std::string>& input_filename,
        const std::string& output_filename,
        size_t num_layers,
        bool verbose,
        TrainingMethod method,
        std::optional<std::chrono::seconds> checkpointing_interval,
        int window_size,
        size_t window_stride,
        size_t learn_depth,
        float learning_rate,
        float layer_learning_rate_multiplier,
        LearningRateSchedule learning_rate_schedule,
        float simulated_annealing_decay_factor,
        float simulated_annealing_initial_multiplier,
        size_t simulated_annealing_decay_epochs,
        float simulated_annealing_min_multiplier,
        WeightInitializerType weight_initializer,
        FFNInitializerType ffn_initializer,
        EmbeddingInitializerType embedding_initializer,
        size_t micro_batch_size,
        size_t num_epochs,
        std::optional<size_t> epoch_size,
        bool disable_early_stopping,
        bool disable_example_convergence,
        const std::string& train_corpus_dir
    )
    {
        std::println(
            "Training mode: depth {}, learning rate {}, micro-batch size {}",
            learn_depth,
            learning_rate,
            micro_batch_size
        );
        if (disable_early_stopping)
            std::println("Early stopping disabled; all {} requested epochs will run", num_epochs);
        if (disable_example_convergence)
            std::println("Per-example convergence removal disabled");
        if (learning_rate > 0.25f)
        {
            std::println(
                "Warning: base learning rate {} is high for the current clipped optimizer and may saturate logits. "
                "Try --learning-rate 0.2 or lower if losses jump to ~38400.",
                learning_rate
            );
        }
        set_nn_log_file("train.log");
        ComputeKernelRegistry::instance().enableRegistrationLog("training-log.txt");

        Corpus corpus{m_filters};
        corpus.load_files_from_dir(train_corpus_dir);
        Statistics stats;

        auto nn = std::make_unique<TextTrainer>(num_layers, corpus, stats);
        const auto initializer_name = [](auto type) {
            using T = decltype(type);
            if constexpr (std::is_same_v<T, EmbeddingInitializerType>)
                return type == EmbeddingInitializerType::VarianceScaledUniform ? "variance-scaled-uniform" : "legacy-uniform";
            else
                return type == T::XavierUniform ? "xavier-uniform" :
                    type == T::XavierInputProjections ? "xavier-input-projections" : "legacy-uniform";
        };
        const char* schedule_name = learning_rate_schedule == LearningRateSchedule::Constant ? "constant" :
            learning_rate_schedule == LearningRateSchedule::Lowering ? "lowering" : "simulated_annealing";
        nlohmann::json training_parameters{
            {"version", 1},
            {"layers", num_layers},
            {"method", training_method_to_string(method)},
            {"window_size", window_size},
            {"window_stride", window_stride},
            {"learn_depth", learn_depth},
            {"learning_rate", learning_rate},
            {"learning_rate_schedule", schedule_name},
            {"layer_learning_rate_multiplier", layer_learning_rate_multiplier},
            {"simulated_annealing_decay_factor", simulated_annealing_decay_factor},
            {"simulated_annealing_initial_multiplier", simulated_annealing_initial_multiplier},
            {"simulated_annealing_decay_epochs", simulated_annealing_decay_epochs},
            {"simulated_annealing_min_multiplier", simulated_annealing_min_multiplier},
            {"weight_initializer", initializer_name(weight_initializer)},
            {"ffn_initializer", initializer_name(ffn_initializer)},
            {"embedding_initializer", initializer_name(embedding_initializer)},
            {"micro_batch_size", micro_batch_size},
            {"epochs", num_epochs},
            {"checkpoint_interval_seconds", checkpointing_interval ? nlohmann::json(checkpointing_interval->count()) : nlohmann::json(nullptr)},
            {"epoch_size", epoch_size ? nlohmann::json(*epoch_size) : nlohmann::json(nullptr)},
            {"disable_early_stopping", disable_early_stopping},
            {"disable_example_convergence", disable_example_convergence},
            {"train_corpus_dir", train_corpus_dir},
            {"filters", m_filters}
        };
        nn->set_training_parameters_json(training_parameters.dump(2));
        nn->set_training_method(method);
        nn->set_window_size(window_size);
        nn->set_window_stride(window_stride);
        nn->set_learn_depth(learn_depth);
        nn->set_learning_rate(learning_rate);
        nn->set_layer_learning_rate_multiplier(layer_learning_rate_multiplier);
        nn->set_learning_rate_schedule(learning_rate_schedule);
        nn->set_simulated_annealing_decay_factor(simulated_annealing_decay_factor);
        nn->set_simulated_annealing_initial_multiplier(simulated_annealing_initial_multiplier);
        nn->set_simulated_annealing_decay_epochs(simulated_annealing_decay_epochs);
        nn->set_simulated_annealing_min_multiplier(simulated_annealing_min_multiplier);
        nn->set_weight_initializer(weight_initializer);
        nn->set_ffn_initializer(ffn_initializer);
        nn->set_embedding_initializer(embedding_initializer);
        nn->set_micro_batch_size(micro_batch_size);
        nn->set_early_stopping_enabled(!disable_early_stopping);
        nn->set_example_convergence_enabled(!disable_example_convergence);

        nn->train(verbose, num_epochs, input_filename, checkpointing_interval, epoch_size);

        stats.print_statistics();
        parallel::statistics.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm
