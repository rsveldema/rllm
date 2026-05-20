
#include <Corpus.hpp>
#include <TokenIDFormatter.hpp>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string_view>


namespace rllm
{

    bool log_enabled = false;

#define LOG_INFO(...) \
    if (log_enabled) \
    { \
        std::println(__VA_ARGS__); \
    }


    static std::vector<std::string> tokenize(std::string_view text)
    {
        std::vector<std::string> tokens;
        std::string word;
        for (char c : text)
        {
            if (std::isspace(static_cast<unsigned char>(c)))
            {
                if (!word.empty())
                {
                    tokens.push_back(std::move(word));
                    word.clear();
                }
            }
            else if (std::ispunct(static_cast<unsigned char>(c)))
            {
                if (!word.empty())
                {
                    tokens.push_back(std::move(word));
                    word.clear();
                }
                tokens.push_back(std::string(1, c));
            }
            else
            {
                word += c;
            }
        }
        if (!word.empty())
        {
            tokens.push_back(std::move(word));
        }
        return tokens;
    }


    Corpus::Corpus()
    {
        const std::filesystem::path corpus_dir{"corpus"};
        if (!std::filesystem::exists(corpus_dir))
        {
            std::println(
                "Corpus directory '{}' does not exist. Please create it and add some text files for training.",
                corpus_dir.string()
            );
            abort();
            return;
        }

        TokenID next_id = TokenID::START;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(corpus_dir))
        {
            if (!entry.is_regular_file())
                continue;

            LOG_INFO("Processing file: {}", entry.path().c_str());

            TokenData& token_data = m_token_list.emplace_back(entry.path().string());

            std::ifstream file{entry.path()};
            std::string line;
            while (std::getline(file, line))
            {
                const auto tokens = tokenize(line);
                if (tokens.empty())
                    continue;
                token_data.next_line();
                for (const auto& token : tokens)
                {

                    const auto it = m_token_to_id.find(token);
                    if (it == m_token_to_id.end())
                    {
                        const auto new_id = inc(next_id);
                        m_token_to_id.emplace(token, new_id);

                        token_data.add(new_id);
                        LOG_INFO("new token: {} with ID: {}", token, new_id);
                    }
                    else
                    {
                        token_data.add(it->second);
                        LOG_INFO("Found token: {} with ID: {}", token, it->second);
                    }
                }
            }
        }

        std::println("Corpus initialized with {} unique tokens", m_token_to_id.size());

        save_token_map("token_map.json");
    }

    std::vector<TokenID> Corpus::get_token_ids(const std::string& text) const
    {
        std::vector<TokenID> ids;
        for (const auto& token : tokenize(text))
        {
            const auto it = m_token_to_id.find(token);
            if (it != m_token_to_id.end())
            {
                ids.push_back(it->second);
            }
            else
            {
                ids.push_back(TokenID::UNKNOWN_TOKEN_ID); // Use a special ID for unknown tokens
            }
        }
        return ids;
    }

    Token Corpus::get_token_from_id(TokenID id) const
    {
        for (const auto& [token, token_id] : m_token_to_id)
        {
            if (token_id == id)
            {
                return token;
            }
        }
        return {};
    }

    std::string Corpus::get_line(const InputLine& line) const
    {
        std::string result;
        for (const auto id : line)
        {
            if (!result.empty())
                result += ' ';
            result += get_token_from_id(id);
        }
        return result;
    }

    InputLine Corpus::get_training_input_line(size_t min_size) const
    {
        assert(!m_token_list.empty());

        const size_t random_index = static_cast<size_t>(rand()) % m_token_list.size();
        return m_token_list[random_index].get_training_input_line(min_size);
    }

} // namespace rllm