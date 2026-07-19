#pragma once

#include <TextTrainer.hpp>

#include <chrono>
#include <optional>
#include <string>

namespace rllm
{
    class Trainer
    {
      public:
        Trainer(const std::vector<std::string>& filters);
        ~Trainer() = default;
        Trainer(const Trainer&) = delete;
        Trainer& operator=(const Trainer&) = delete;

        void train_mode(
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
        );
      private:
        const std::vector<std::string> m_filters;
    };

} // namespace rllm
