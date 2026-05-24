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
    class Corpus
    {
      public:
        Corpus(const std::vector<std::string>& filters);
        void load_files_from_dir();

        using visitor_fn_t = std::function<void(const InputLine&)>;

        InputLine get_token_ids(const std::string& text) const;
        Token get_token_from_id(TokenID id) const;
        std::optional<std::string> get_line(const InputLine& line) const;

        std::vector<InputLine> get_suitable_training_lines() const;

        void visit_lines(const visitor_fn_t& visitor) const
        {
            for (const auto& token_data : m_token_list)
            {
                token_data.visit_lines(visitor);
            }
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

            InputLine get_training_input_line(size_t min_size) const
            {
                while (true)
                {
                    const auto random_index = static_cast<size_t>(rand()) % m_lines.size();
                    InputLine result = m_lines[random_index];
                    if (static_cast<int>(result.size()) >= min_size)
                    {
                        return result;
                    }
                }
            }


            void visit_lines(const visitor_fn_t& visitor) const
            {
                for (const auto& line : m_lines)
                {
                    visitor(line);
                }
            }

            size_t number_of_lines() const
            {
                return m_lines.size();
            }

          private:
            std::string filename;
            std::vector<InputLine> m_lines; // positions of the token in the corpus
            std::vector<TokenID> m_tokens_in_file; // the actual token IDs in the file, in order
        };

        std::vector<TokenData> m_token_list;
        const std::vector<std::string>& m_filters;
    };

} // namespace rllm
