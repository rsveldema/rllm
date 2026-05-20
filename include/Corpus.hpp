#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <cassert>

namespace rllm
{

    enum class TokenID : int32_t {
        UNKNOWN_TOKEN_ID = -1,
        START = 0,
        MAX = 4096
    };

    static inline TokenID inc(TokenID id)
    {
        assert(id != TokenID::UNKNOWN_TOKEN_ID);
        assert(id < TokenID::MAX);
        return static_cast<TokenID>(static_cast<int32_t>(id) + 1);
    }

    using Token = std::string;

    using InputLine = std::vector<TokenID>;

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
                const auto random_index = static_cast<size_t>(rand()) % m_data.size();
                InputLine result = m_data[random_index];

                while (result.size() < min_size)
                {
                    result.push_back(TokenID::UNKNOWN_TOKEN_ID);
                }

                return result;
            }

          private:
            std::string filename;
            std::vector<InputLine> m_data; // positions of the token in the corpus
        };

        std::vector<TokenData> m_token_list;
    };

} // namespace rllm
