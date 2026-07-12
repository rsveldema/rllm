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
            size_t learn_depth,
            float learning_rate,
            size_t micro_batch_size,
            size_t num_epochs,
            std::optional<size_t> epoch_size,
            const std::string& train_corpus_dir
        );
      private:
        const std::vector<std::string> m_filters;
    };

} // namespace rllm
