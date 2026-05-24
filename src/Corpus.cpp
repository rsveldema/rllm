
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
            if (!file)
            {
                std::println("Failed to open file '{}'", entry.path().string());
                continue;
            }

            std::string line;
            while (std::getline(file, line))
            {
                const auto input_line = get_token_ids(line);
                for (const auto i : enum_iterator<PositionIndex>(input_line.size()))
                {
                    assert(input_line[i] >= TokenID::START);
                    assert(input_line[i] < TokenID::MAX);
                    token_data.add(input_line[i]);
                }
                token_data.add(TokenID::TOK_NEWLINE); // add a newline token at the end of each line
            }
        }
    }

    InputLine Corpus::get_token_ids(const std::string& text) const
    {
        InputLine result;

        size_t ix = 0;

        while (ix < text.size())
        {
            bool matched_token = false;
            for (const auto& token_id_and_string : tokenizer_map)
            {
                const auto& token_id = token_id_and_string.first;
                const auto& token_string = token_id_and_string.second;

                if (text.compare(ix, token_string.size(), token_string) == 0)
                {
                    result.push_back(token_id);
                    ix += token_string.size();
                    matched_token = true;
                    break;
                }
            }

            if (!matched_token)
            {
                // If no token matched, skip this character
                ix++;
                LOG_INFO("Warning: No token matched for character '{}', skipping it", text[ix - 1]);
            } else {
                LOG_INFO("Matched token '{}' at position {}", tokenizer_map[result.back()], ix);
            }
        }

        return result;
    }

    Token Corpus::get_token_from_id(TokenID id) const
    {
        if (id == TokenID::UNKNOWN_TOKEN_ID)
        {
            return "<UNK>";
        }
        assert(id < TokenID::MAX);
        return tokenizer_map[id];
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