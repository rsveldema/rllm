#pragma once

#include <NeuralNetwork.hpp>

#include <string>

namespace rllm
{
    class RLLM
    {
      public:
        RLLM(const std::vector<std::string>& filters);
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(
            const std::optional<std::string>& input_filename,
            const std::string& output_filename,
            size_t num_layers,
            bool verbose,
            TrainingMethod method,
            std::optional<size_t> checkpointing_interval,
            int window_size,
            size_t num_epochs,
            const std::string& train_corpus_dir
        );
        void prompt_mode(const std::string& input_filename, const std::optional<std::string>& one_shot_prompt = std::nullopt);

        struct PromptOptions
        {
            bool highest_prio_only = true;
        };

      private:
        void process_line(const std::string& line, Corpus& corpus, NeuralNetwork& nn, PromptOptions& options);

        const std::vector<std::string> m_filters;
    };

} // namespace rllm