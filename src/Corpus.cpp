
#include <Corpus.hpp>
#include <JsonTensorHelpers.hpp>
#include <TokenIDFormatter.hpp>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
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
            else if (std::ispunct(static_cast<unsigned char>(c)) and c != '_')
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


    Corpus::Corpus(const std::vector<std::string>& filters)
        : m_filters(filters)
    {}

    void Corpus::load_files_from_dir()
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

            if (!m_filters.empty())
            {
                const auto filename = entry.path().filename().string();
                bool matches_filter = false;
                for (const auto& filter : m_filters)
                {
                    if (filename.find(filter) != std::string::npos)
                    {
                        matches_filter = true;
                        break;
                    }
                }
                if (!matches_filter)
                {
                    LOG_INFO("Skipping file '{}' due to filters", entry.path().string());
                    continue;
                }
            }

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
                        next_id = new_id;

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

    InputLine Corpus::get_token_ids(const std::string& text) const
    {
        InputLine ids;
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

    std::optional<std::string> Corpus::get_line(const InputLine& line) const
    {
        std::string result;
        for (const auto i : enum_iterator<PositionIndex>(line.size()))
        {
            if (!result.empty())
            {
                result += ' ';
            }
            if (line[i] == TokenID::UNKNOWN_TOKEN_ID)
            {
                return std::nullopt; // line contains unknown token ID, cannot convert to string
            }
            result += get_token_from_id(line[i]);
        }
        if (result.empty())
        {
            return std::nullopt; // empty line
        }
        return result;
    }

    /*
    InputLine Corpus::get_training_input_line(size_t min_size) const
    {
        assert(!m_token_list.empty());

        for (const auto& token_data : m_token_list)
        {
            return token_data.get_training_input_line(min_size);
        }

        assert(false);
        const size_t random_index = static_cast<size_t>(rand()) % m_token_list.size();
        return m_token_list[random_index].get_training_input_line(min_size);
    }
        */

    void Corpus::save_token_map(const std::string& filename) const
    {
        std::ofstream file{filename};
        file << save_token_map_json().dump(2) << '\n';
    }

    nlohmann::json Corpus::save_token_map_json() const
    {
        nlohmann::json j;
        for (const auto& [token, id] : m_token_to_id)
        {
            j[token] = static_cast<int32_t>(id);
        }
        return j;
    }

    void Corpus::load_token_map_json(const nlohmann::json& j)
    {
        m_token_to_id.clear();
        for (const auto& [token, id_j] : j.items())
        {
            m_token_to_id.emplace(token, static_cast<TokenID>(id_j.get<int32_t>()));
        }
    }

    std::vector<InputLine> Corpus::get_suitable_training_lines() const
    {
        std::vector<InputLine> training_lines;

        assert(!m_token_list.empty());
        this->visit_lines([&](const InputLine& line) {
            if (static_cast<int>(line.size()) < 2)
            {
                return; // skip too-short lines that can't be used for training
            }
            training_lines.push_back(line);
        });
        return training_lines;
    }


} // namespace rllm