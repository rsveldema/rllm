#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <optional>

#include <nlohmann/json_fwd.hpp>
#include <LayerPrimitives.hpp>

namespace rllm
{
    struct WindowExample
    {
        CpuInputLine line;
        PositionIndex context_length;
    };

    /** Build next-token examples with explicit context lengths from corpus lines.
     * Windows never cross a line boundary. Each stride-selected token is the
     * primary target, followed by up to three additional MTP targets.
     */
    std::vector<WindowExample> make_line_windows(
        const std::vector<CpuInputLine>& lines,
        size_t window_size,
        size_t stride,
        bool reverse = false
    );

    class Corpus
    {
      public:
                struct TrainingSplit
                {
                        std::vector<CpuInputLine> training_lines;
                        std::vector<CpuInputLine> validation_lines;
                };

        Corpus(const std::vector<std::string>& filters);
        void load_files_from_dir(const std::string& train_corpus_dir);

        using visitor_fn_t       = std::function<void(const CpuInputLine&)>;
        using token_visitor_fn_t = std::function<void(TokenID)>;

        CpuInputLine get_token_ids(const std::string& text) const;
        Token get_token_from_id(TokenID id) const;
        std::optional<std::string> get_line(const CpuInputLine& line) const;

        std::vector<CpuInputLine> get_suitable_training_lines() const;
        TrainingSplit get_deterministic_training_split(size_t validation_percent = 20) const;

        void visit_lines(const visitor_fn_t& visitor) const
        {
            for (const auto& token_data : m_token_list)
                token_data.visit_lines(visitor);
        }

        // Iterate over every token in every file in corpus order.
        void visit_flat_tokens(const token_visitor_fn_t& visitor) const
        {
            for (const auto& token_data : m_token_list)
                token_data.visit_tokens(visitor);
        }

        size_t count_num_lines() const
        {
            size_t total = 0;
            for (const auto& token_data : m_token_list)
            {
                total += token_data.number_of_lines();
            }
            return total;
        }

      private:
        class TokenData
        {
          public:
            explicit TokenData(std::string filename)
                : filename(std::move(filename))
            {}

            void add(TokenID id)
            {
                m_tokens_in_file.push_back(id);
                if (m_lines.empty())
                {
                    m_lines.emplace_back();
                    m_lines.back().push_back(id);
                    return;
                }

                m_lines.back().push_back(id);
                if (id == TokenID::TOK_NEWLINE)
                {
                    m_lines.emplace_back();
                }
            }

            const CpuInputLine& get_training_input_line(size_t min_size) const
            {
                while (true)
                {
                    const auto random_index = static_cast<size_t>(rand()) % m_lines.size();
                    const CpuInputLine& result = m_lines[random_index];
                    if (static_cast<int>(result.size()) >= min_size)
                    {
                        return result;
                    }
                }
            }


            void visit_lines(const visitor_fn_t& visitor) const
            {
                for (const auto& line : m_lines)
                    visitor(line);
            }

            void visit_tokens(const token_visitor_fn_t& visitor) const
            {
                for (const auto tok : m_tokens_in_file)
                    visitor(tok);
            }

            size_t number_of_lines() const
            {
                return m_lines.size();
            }

          private:
            std::string filename;
            std::vector<CpuInputLine> m_lines; // positions of the token in the corpus
            std::vector<TokenID> m_tokens_in_file; // the actual token IDs in the file, in order
        };

        std::vector<TokenData> m_token_list;
        const std::vector<std::string>& m_filters;
        mutable size_t m_tokenization_errors = 0;
    };

} // namespace rllm
