
#include <Corpus.hpp>
#include <JsonTensorHelpers.hpp>
#include <TokenIDFormatter.hpp>

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <print>
#include <sstream>
#include <string_view>
#include <unordered_set>


namespace rllm
{
    namespace
    {
        void rebalance_training_split(Corpus::TrainingSplit& split)
        {
            const size_t total = split.training_lines.size() + split.validation_lines.size();
            if (total < 2)
                return;

            if (split.validation_lines.empty() && split.training_lines.size() > 1)
            {
                split.validation_lines.push_back(split.training_lines.back());
                split.training_lines.pop_back();
            }
            else if (split.training_lines.empty() && split.validation_lines.size() > 1)
            {
                split.training_lines.push_back(split.validation_lines.back());
                split.validation_lines.pop_back();
            }
        }
    }

    bool log_info_enabled = true;
    bool log_debug_enabled = true;

    static std::ofstream s_log_file{"tokenization.log"};

#ifdef LOG_INFO
#undef LOG_INFO
#endif

#ifdef LOG_ERROR
#undef LOG_ERROR
#endif

#define LOG_INFO(...) \
    if (log_info_enabled) \
    { \
        std::println(s_log_file, __VA_ARGS__); \
        s_log_file << std::flush; \
    }

#define LOG_ERROR(...) \
    { \
        std::println(s_log_file, __VA_ARGS__); \
        s_log_file << std::flush; \
    }

#define LOG_DEBUG(...) \
    if (log_debug_enabled) \
    { \
        std::println(s_log_file, __VA_ARGS__); \
        s_log_file << std::flush; \
    }

    Corpus::Corpus(const std::vector<std::string>& filters)
        : m_filters(filters)
    {}

    void Corpus::load_files_from_dir(const std::string& train_corpus_dir)
    {
        const std::filesystem::path corpus_dir{train_corpus_dir};
        if (!std::filesystem::exists(corpus_dir))
        {
            std::println(
                "Corpus directory '{}' does not exist. Please create it and add some text files for training.",
                corpus_dir.string()
            );
            abort();
            return;
        }

        LOG_INFO("Loading files from corpus directory: '{}'", corpus_dir.string());

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

            auto& token_data = m_token_list.emplace_back(entry.path().string());

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
                for (const auto i : enum_iterator1D<PositionIndex>(input_line.size()))
                {
                    assert(input_line[i] >= TokenID::START);
                    assert(input_line[i] < TokenID::MAX);
                    token_data.add(input_line[i]);
                }
                token_data.add(TokenID::TOK_NEWLINE); // add a newline token at the end of each line
            }
        }

        if (m_tokenization_errors > 0)
        {
            std::println(
                "Tokenization failed: {} character(s) could not be matched to any token. "
                "Check tokenization.log for details. "
                "Re-run create_tokenizer_map.py to regenerate the token map.",
                m_tokenization_errors
            );
            std::abort();
        }
    }

CpuInputLine Corpus::get_token_ids(const std::string& text) const
    {
        CpuInputLine result;

        size_t ix = 0;

        while (ix < text.size())
        {
            bool matched_token = false;
            for (const auto& token_id_and_string : tokenizer_map)
            {
                const auto& token_id = token_id_and_string.first;
                const auto& token_info = token_id_and_string.second;
                const auto token_len = std::strlen(token_info.str);

                if (text.compare(ix, token_len, token_info.str) == 0)
                {
                    if (token_info.end_of_word)
                    {
                        const size_t next_ix = ix + token_len;
                        if (next_ix < text.size())
                        {
                            const auto next_char = text[next_ix];
                            if (std::isalnum((unsigned char) next_char) || next_char == '_')
                            {
                                LOG_DEBUG(
                                    "Matched token '{}/{}' at position {}, but not at a word boundary, skipping it "
                                    "(remaining text: '{}')",
                                    token_info.str,
                                    token_id,
                                    ix,
                                    next_char
                                );
                                continue; // matched string, but not at a word boundary
                            }
                        }
                    }
                    result.push_back(token_id);
                    ix += token_len;
                    matched_token = true;
                    break;
                }
            }

            if (!matched_token)
            {
                // If no token matched, skip this character
                const auto ch = text[ix];
                if (! isspace(ch))
                {
                    // spaces have no explicit token, we just skip them without logging,
                    // but log other unmatched characters as warnings since they may
                    // indicate a problem with the tokenizer map.
                    LOG_ERROR("ERROR: No token matched for character '{}', skipping it", ch);
                    ++m_tokenization_errors;
                }
                ix++;
            }
            else
            {
                LOG_DEBUG("Matched token '{}'/{} at position {}", tokenizer_map[result.back()].str, result.back(), ix);
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
        if (id < TokenID::START || id >= TokenID::MAX)
        {
            return "<UNK>";
        }

        const auto it = tokenizer_map.find(id);
        if (it == tokenizer_map.end() || it->second.str == nullptr)
        {
            return "<UNK>";
        }

        return it->second.str;
    }

    std::optional<std::string> Corpus::get_line(const CpuInputLine& line) const
    {
        std::string result;
        for (const auto i : enum_iterator1D<PositionIndex>(line.size()))
        {
            auto& token_id = line[i];
            if (token_id == TokenID::UNKNOWN_TOKEN_ID)
            {
                return std::nullopt; // line contains unknown token ID, cannot convert to string
            }

            const auto it = tokenizer_map.find(token_id);
            if (it == tokenizer_map.end() || it->second.str == nullptr)
            {
                return std::nullopt;
            }

            const auto& token_info = it->second;
            result += get_token_from_id(token_id);
            if (token_info.end_of_word)
            {
                result += ' ';
            }
        }
        if (result.empty())
        {
            return std::nullopt; // empty line
        }
        return result;
    }

    std::vector<CpuInputLine> Corpus::get_suitable_training_lines() const
    {
        std::vector<CpuInputLine> training_lines;
        std::unordered_set<std::string> seen_training_line_keys;

        auto make_line_key = [](const CpuInputLine& line) {
            std::string key;
            key.reserve(static_cast<size_t>(line.size()) * 6);
            for (const auto i : enum_iterator1D<PositionIndex>(line.size()))
            {
                key += std::to_string(static_cast<int>(line[i]));
                key.push_back(',');
            }
            return key;
        };

        assert(!m_token_list.empty());
        this->visit_lines([&](const CpuInputLine& line) {
            // Strip the trailing TOK_NEWLINE that the corpus appends to every line.
            // Keeping it would make \n the dominant training target (it ends every line),
            // causing the model to collapse to always predicting \n.
            CpuInputLine stripped = line;
            if (!stripped.empty() && stripped.back() == TokenID::TOK_NEWLINE)
                stripped.pop_back();

            if (static_cast<int>(stripped.size()) < 2)
                return; // too short to produce a valid (input, target) pair

            const auto dedupe_key = make_line_key(stripped);
            if (!seen_training_line_keys.insert(dedupe_key).second)
                return; // duplicate line in corpus; already included once

            training_lines.push_back(stripped);
        });
        return training_lines;
    }

    Corpus::TrainingSplit Corpus::get_deterministic_training_split(size_t validation_percent) const
    {
        assert(validation_percent <= 100);

        TrainingSplit split;
        auto training_lines = get_suitable_training_lines();
        if (training_lines.empty())
            return split;

        if (validation_percent == 0)
        {
            split.training_lines = std::move(training_lines);
            return split;
        }

        for (auto& line : training_lines)
        {
            const size_t bucket = static_cast<size_t>(line.hash() % 100ull);
            if (bucket < validation_percent)
                split.validation_lines.push_back(std::move(line));
            else
                split.training_lines.push_back(std::move(line));
        }

        rebalance_training_split(split);
        return split;
    }


} // namespace rllm
