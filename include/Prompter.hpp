#pragma once

#include <TextTrainer.hpp>

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
            // How many MTP heads to use for token generation (1 = head 0 only, up to MAX).
            size_t mtp_heads = 1;
        };

        explicit Prompter(const std::vector<std::string>& filters);

        void prompt_mode(const std::string& filename, const std::optional<std::string>& one_shot_prompt,
                         size_t mtp_heads = 1);

      private:
        void process_line(const std::string& line, Corpus& corpus, TextTrainer& nn, PromptOptions& options);

        const std::vector<std::string> m_filters;
    };

} // namespace rllm
