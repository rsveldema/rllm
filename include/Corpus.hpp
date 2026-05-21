#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <LayerPrimitives.hpp>

namespace rllm
{
    class Corpus
    {
      public:
        Corpus();

        InputLine get_token_ids(const std::string& text) const;
        Token get_token_from_id(TokenID id) const;
        std::string get_line(const InputLine& line) const;
        InputLine get_training_input_line(size_t min_size) const;
        void save_token_map(const std::string& filename) const;

        size_t size() const
        {
            return m_token_to_id.size();
        }

      private:
        std::map<Token, TokenID> m_token_to_id;

        class TokenData
        {
          public:
            explicit TokenData(std::string filename)
                : filename(std::move(filename))
            {}

            void add(TokenID id)
            {
                if (m_data.empty())
                {
                    m_data.emplace_back();
                }
                m_data.back().push_back(id);
            }

            void next_line()
            {
                m_data.emplace_back();
            }

            InputLine get_training_input_line(size_t min_size) const
            {
                while (true)
                {
                    const auto random_index = static_cast<size_t>(rand()) % m_data.size();
                    InputLine result = m_data[random_index];
                    if (static_cast<int>(result.size()) >= min_size)
                    {
                        return result;
                    }
                }
            }

          private:
            std::string filename;
            std::vector<InputLine> m_data; // positions of the token in the corpus
        };

        std::vector<TokenData> m_token_list;
    };

} // namespace rllm
