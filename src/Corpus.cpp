
#include <Corpus.hpp>
#include <JsonTensorHelpers.hpp>
#include <TokenIDFormatter.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <print>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>


namespace rllm
{
    TokenID language_token(SourceLanguage language)
    {
        switch (language)
        {
        case SourceLanguage::Cpp: return TokenID::LANG_CPP;
        case SourceLanguage::C: return TokenID::LANG_C;
        case SourceLanguage::Python: return TokenID::LANG_PYTHON;
        case SourceLanguage::Rust: return TokenID::LANG_RUST;
        case SourceLanguage::Java: return TokenID::LANG_JAVA;
        case SourceLanguage::Shell: return TokenID::LANG_SHELL;
        case SourceLanguage::Unknown: break;
        }
        return TokenID::LANG_CPP;
    }

    std::optional<SourceLanguage> parse_source_language(std::string_view name)
    {
        if (name == "cpp" || name == "c++" || name == "cc") return SourceLanguage::Cpp;
        if (name == "c") return SourceLanguage::C;
        if (name == "python" || name == "py") return SourceLanguage::Python;
        if (name == "rust" || name == "rs") return SourceLanguage::Rust;
        if (name == "java") return SourceLanguage::Java;
        if (name == "shell" || name == "sh" || name == "bash") return SourceLanguage::Shell;
        return std::nullopt;
    }

    SourceLanguage Corpus::TokenData::language() const
    {
        const auto extension = std::filesystem::path(filename).extension().string();
        if (extension == ".c") return SourceLanguage::C;
        if (extension == ".py") return SourceLanguage::Python;
        if (extension == ".rs") return SourceLanguage::Rust;
        if (extension == ".java") return SourceLanguage::Java;
        if (extension == ".sh") return SourceLanguage::Shell;
        if (extension == ".cpp" || extension == ".cc" || extension == ".cxx" ||
            extension == ".h" || extension == ".hpp") return SourceLanguage::Cpp;
        return SourceLanguage::Unknown;
    }
    namespace
    {
    template<typename Sequence>
    std::vector<WindowExample> make_sequence_windows(
        const std::vector<Sequence>& sequences,
        size_t window_size,
        size_t stride,
        bool reverse)
    {
        assert(window_size >= 2);
        assert(stride > 0);

        std::vector<WindowExample> windows;
        for (size_t sequence_index = 0; sequence_index < sequences.size(); ++sequence_index)
        {
            const auto& sequence = sequences[sequence_index];
            const size_t sequence_size = static_cast<size_t>(sequence.size());
            if (sequence_size < 2)
                continue;
            std::vector<bool> in_comment_at(sequence_size, false);
            size_t comment_depth = 0;
            for (size_t position = 0; position < sequence_size; ++position)
            {
                in_comment_at[position] = comment_depth > 0;
                const TokenID token = [&]() {
                    if constexpr (std::is_same_v<Sequence, CpuInputLine>)
                        return sequence[static_cast<PositionIndex>(position)];
                    else
                        return sequence[position];
                }();
                if (token == TokenID::BLOCK_COMMENT_START)
                    ++comment_depth;
                else if (token == TokenID::BLOCK_COMMENT_END && comment_depth > 0)
                    --comment_depth;
            }
            const size_t prediction_capacity = window_size - 1;
            const size_t block_span = prediction_capacity >= stride
                ? (prediction_capacity / stride) * stride
                : 1;
            const size_t block_advance = prediction_capacity >= stride ? block_span : stride;
            for (size_t first_target = 1; first_target < sequence_size; )
            {
                const size_t start = first_target - 1;
                const size_t block_predictions = std::min(block_span, sequence_size - first_target);
                const size_t end = first_target + block_predictions;
                CpuInputLine window;
                for (size_t position = start; position < end; ++position)
                {
                    if constexpr (std::is_same_v<Sequence, CpuInputLine>)
                        window.push_back(sequence[static_cast<PositionIndex>(position)]);
                    else
                        window.push_back(sequence[position]);
                }
                windows.push_back({
                    .line = std::move(window),
                    .context_length = static_cast<PositionIndex>(end - start - 1),
                    .source_index = sequence_index,
                    .starts_in_block_comment = in_comment_at[start]
                });
                first_target += block_advance;
            }
        }
        if (reverse)
            std::reverse(windows.begin(), windows.end());
        return windows;
    }
    }

    std::vector<WindowExample> make_line_windows(
        const std::vector<CpuInputLine>& lines, size_t window_size, size_t stride, bool reverse)
    {
        return make_sequence_windows(lines, window_size, stride, reverse);
    }

    std::vector<WindowExample> make_file_windows(
        const std::vector<FileTokenSequence>& files, size_t window_size, size_t stride, bool reverse)
    {
        return make_sequence_windows(files, window_size, stride, reverse);
    }

    FileWindowTrainingSplit make_file_window_training_split(
        const std::vector<FileTokenSequence>& files,
        size_t window_size,
        size_t stride,
        size_t validation_percent,
        size_t maximum_count,
        bool reverse)
    {
        assert(validation_percent <= 100);
        assert(maximum_count > 0);

        FileWindowTrainingSplit split;
        std::vector<std::vector<WindowExample>> validation_by_file;
        validation_by_file.reserve(files.size());
        for (size_t file_index = 0; file_index < files.size(); ++file_index)
        {
            const auto& file = files[file_index];
            auto windows = make_sequence_windows(
                std::vector<FileTokenSequence>{file}, window_size, stride, false);
            for (auto& window : windows)
                window.source_index = file_index;
            if (windows.size() < 2 || validation_percent == 0)
            {
                split.training_windows.insert(
                    split.training_windows.end(),
                    std::make_move_iterator(windows.begin()),
                    std::make_move_iterator(windows.end()));
                validation_by_file.emplace_back();
                continue;
            }

            size_t validation_count = windows.size() * validation_percent / 100;
            validation_count = std::clamp(validation_count, size_t{1}, windows.size() - 1);
            std::vector<bool> is_validation(windows.size(), false);
            for (size_t sample = 0; sample < validation_count; ++sample)
            {
                const size_t index = (sample + 1) * windows.size() / (validation_count + 1);
                is_validation[index] = true;
            }

            std::vector<WindowExample> file_validation;
            file_validation.reserve(validation_count);
            for (size_t index = 0; index < windows.size(); ++index)
            {
                auto& destination = is_validation[index]
                    ? file_validation
                    : split.training_windows;
                destination.push_back(std::move(windows[index]));
            }
            split.full_validation_window_count += file_validation.size();
            ++split.split_file_count;
            validation_by_file.push_back(std::move(file_validation));
        }

        std::vector<size_t> quotas(files.size(), 0);
        size_t assigned = 0;
        const size_t sample_count = std::min(maximum_count, split.full_validation_window_count);
        while (assigned < sample_count)
        {
            bool assigned_in_round = false;
            for (size_t file_index = 0;
                 file_index < validation_by_file.size() && assigned < sample_count;
                 ++file_index)
            {
                if (quotas[file_index] >= validation_by_file[file_index].size())
                    continue;
                ++quotas[file_index];
                ++assigned;
                assigned_in_round = true;
            }
            if (!assigned_in_round)
                break;
        }

        split.validation_windows.reserve(assigned);
        for (size_t file_index = 0; file_index < validation_by_file.size(); ++file_index)
        {
            auto& source = validation_by_file[file_index];
            const size_t quota = quotas[file_index];
            for (size_t sample = 0; sample < quota; ++sample)
            {
                const size_t source_index = sample * source.size() / quota;
                split.validation_windows.push_back(std::move(source[source_index]));
            }
        }
        if (reverse)
        {
            std::reverse(split.training_windows.begin(), split.training_windows.end());
            std::reverse(split.validation_windows.begin(), split.validation_windows.end());
        }
        return split;
    }

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
    bool log_debug_enabled = false;

    static std::ofstream s_log_file;

    void set_tokenization_log_file(const std::string& filename)
    {
        s_log_file.close();
        s_log_file.clear();
        s_log_file.open(filename, std::ios::trunc);
    }

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
    {
        if (!s_log_file.is_open())
            set_tokenization_log_file("tokenization.log");
    }

    void Corpus::load_files_from_dir(
        const std::string& train_corpus_dir,
        size_t source_index,
        double source_weight)
    {
        assert(source_weight > 0.0);
        if (m_source_weights.size() <= source_index)
            m_source_weights.resize(source_index + 1, 1.0);
        m_source_weights[source_index] = source_weight;
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

            auto& token_data = m_token_list.emplace_back(entry.path().string(), source_index);

            std::ifstream file{entry.path()};
            if (!file)
            {
                std::println("Failed to open file '{}'", entry.path().string());
                continue;
            }

            std::string line;
            CommentLexState comment_state;
            while (std::getline(file, line))
            {
                const auto input_line = get_token_ids(line, token_data.language(), comment_state);
                for (const auto i : enum_iterator1D<PositionIndex>(input_line.size()))
                {
                    assert(input_line[i] >= TokenID::START);
                    assert(input_line[i] < TokenID::MAX);
                    token_data.add(input_line[i]);
                }
                if (!comment_state.line_comment_on_last_line)
                    token_data.add(TokenID::TOK_NEWLINE);
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

    CpuInputLine Corpus::get_token_ids(
        const std::string& text,
        SourceLanguage language,
        CommentLexState& state) const
    {
        CpuInputLine result;
        state.line_comment_on_last_line = false;
        const auto append_text = [&](std::string_view part) {
            const auto tokens = get_token_ids(std::string{part});
            for (const auto position : enum_iterator1D<PositionIndex>(tokens.size()))
                result.push_back(tokens[position]);
        };

        const bool slash_comments = language == SourceLanguage::Cpp ||
            language == SourceLanguage::C || language == SourceLanguage::Rust ||
            language == SourceLanguage::Java;
        const bool hash_comments = language == SourceLanguage::Python ||
            language == SourceLanguage::Shell;

        size_t segment_start = 0;
        size_t position = 0;
        char quote = '\0';
        bool escaped = false;
        while (position < text.size())
        {
            if (state.block_depth > 0)
            {
                const bool nested_start = language == SourceLanguage::Rust &&
                    position + 1 < text.size() && text.compare(position, 2, "/*") == 0;
                const bool block_end = position + 1 < text.size() &&
                    text.compare(position, 2, "*/") == 0;
                if (!nested_start && !block_end)
                {
                    ++position;
                    continue;
                }
                append_text(std::string_view{text}.substr(segment_start, position - segment_start));
                if (nested_start)
                {
                    result.push_back(TokenID::BLOCK_COMMENT_START);
                    ++state.block_depth;
                }
                else
                {
                    result.push_back(TokenID::BLOCK_COMMENT_END);
                    --state.block_depth;
                }
                position += 2;
                segment_start = position;
                continue;
            }

            const char current = text[position];
            if (quote != '\0')
            {
                if (escaped)
                    escaped = false;
                else if (current == '\\')
                    escaped = true;
                else if (current == quote)
                    quote = '\0';
                ++position;
                continue;
            }
            if (current == '\'' || current == '"')
            {
                quote = current;
                ++position;
                continue;
            }

            const bool line_start =
                (slash_comments && position + 1 < text.size() && text.compare(position, 2, "//") == 0) ||
                (hash_comments && current == '#');
            const bool block_start = slash_comments && position + 1 < text.size() &&
                text.compare(position, 2, "/*") == 0;
            if (!line_start && !block_start)
            {
                ++position;
                continue;
            }

            append_text(std::string_view{text}.substr(segment_start, position - segment_start));
            const size_t delimiter_size = current == '#' ? 1 : 2;
            result.push_back(line_start
                ? TokenID::LINE_COMMENT_START
                : TokenID::BLOCK_COMMENT_START);
            position += delimiter_size;
            segment_start = position;
            if (line_start)
            {
                append_text(std::string_view{text}.substr(segment_start));
                result.push_back(TokenID::LINE_COMMENT_END);
                state.line_comment_on_last_line = true;
                return result;
            }
            state.block_depth = 1;
        }

        append_text(std::string_view{text}.substr(segment_start));
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
            if (static_cast<int>(line.size()) < 2)
                return; // too short to produce a valid (input, target) pair

            const auto dedupe_key = make_line_key(line);
            if (!seen_training_line_keys.insert(dedupe_key).second)
                return; // duplicate line in corpus; already included once

            training_lines.push_back(line);
        });
        return training_lines;
    }

    std::vector<FileTokenSequence> Corpus::get_file_token_sequences() const
    {
        std::vector<FileTokenSequence> files;
        for (const auto& token_data : m_token_list)
        {
            if (token_data.tokens().size() >= 2)
                files.push_back(token_data.tokens());
        }
        return files;
    }

    std::vector<size_t> Corpus::get_file_source_indices() const
    {
        std::vector<size_t> indices;
        for (const auto& token_data : m_token_list)
        {
            if (token_data.tokens().size() >= 2)
                indices.push_back(token_data.source_index());
        }
        return indices;
    }

    std::vector<SourceLanguage> Corpus::get_file_languages() const
    {
        std::vector<SourceLanguage> languages;
        for (const auto& token_data : m_token_list)
        {
            if (token_data.tokens().size() >= 2)
                languages.push_back(token_data.language());
        }
        return languages;
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

    Corpus::FileTrainingSplit Corpus::get_deterministic_file_split(size_t validation_percent) const
    {
        assert(validation_percent <= 100);

        FileTrainingSplit split;
        for (const auto& token_data : m_token_list)
        {
            const auto& tokens = token_data.tokens();
            if (tokens.size() < 2)
                continue;

            uint64_t hash = 1469598103934665603ull;
            for (const auto token : tokens)
            {
                hash ^= static_cast<uint64_t>(token);
                hash *= 1099511628211ull;
            }
            if (validation_percent != 0 && hash % 100ull < validation_percent)
                split.validation_files.push_back(tokens);
            else
                split.training_files.push_back(tokens);
        }

        const size_t total = split.training_files.size() + split.validation_files.size();
        if (total >= 2)
        {
            if (split.validation_files.empty())
            {
                split.validation_files.push_back(std::move(split.training_files.back()));
                split.training_files.pop_back();
            }
            else if (split.training_files.empty())
            {
                split.training_files.push_back(std::move(split.validation_files.back()));
                split.validation_files.pop_back();
            }
        }
        return split;
    }


} // namespace rllm
