#pragma once

#include <NeuralNetwork.hpp>

#include <optional>
#include <string>
#include <vector>

namespace rllm
{
    class Prompter
    {
      public:
        struct PromptOptions
        {
            bool highest_prio_only = true;
        };

        explicit Prompter(const std::vector<std::string>& filters);

        void prompt_mode(const std::string& filename, const std::optional<std::string>& one_shot_prompt);

      private:
        void process_line(const std::string& line, Corpus& corpus, NeuralNetwork& nn, PromptOptions& options);

        const std::vector<std::string> m_filters;
    };

} // namespace rllm
