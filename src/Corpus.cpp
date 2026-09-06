
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

    namespace
    {
        constexpr std::string_view loop_overflow_token = "<LOOP_OVERFLOW>";
        constexpr std::string_view local_overflow_token = "<LOCAL_OVERFLOW>";
        constexpr std::string_view param_overflow_token = "<PARAM_OVERFLOW>";
        constexpr std::string_view global_overflow_token = "<GLOBAL_OVERFLOW>";
        constexpr std::string_view field_access_token = "<FIELD_ACCESS_IDENT>";
        constexpr std::string_view legacy_identifier_token = "<IDENTIFIER>";
        constexpr std::string_view string_token = "<STRING>";
        constexpr std::string_view mcp_start_token = "<MCP>";
        constexpr std::string_view mcp_end_token = "</MCP>";

        bool identifier_start(char ch)
        {
            return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
        }

        bool identifier_continue(char ch)
        {
            return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
        }

        std::string_view language_name(SourceLanguage language)
        {
            switch (language)
            {
            case SourceLanguage::Cpp: return "cpp";
            case SourceLanguage::C: return "c";
            case SourceLanguage::Python: return "python";
            case SourceLanguage::Rust: return "rust";
            case SourceLanguage::Java: return "java";
            case SourceLanguage::Shell: return "shell";
            case SourceLanguage::Unknown: return "";
            }
            return "";
        }

        size_t identifier_end(std::string_view text, size_t start)
        {
            size_t end = start + 1;
            while (end < text.size() && identifier_continue(text[end]))
                ++end;
            return end;
        }

        size_t next_non_space(std::string_view text, size_t position)
        {
            while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position])))
                ++position;
            return position;
        }

        bool is_function_declaration_name(
            std::string_view text, size_t start, size_t end, SourceLanguage language)
        {
            const size_t next = next_non_space(text, end);
            if (next >= text.size() || text[next] != '(')
                return false;
            const size_t line_start = text.rfind('\n', start) == std::string_view::npos
                ? 0 : text.rfind('\n', start) + 1;
            const auto prefix = text.substr(line_start, start - line_start);
            if (language == SourceLanguage::Python)
                return prefix.find("def ") != std::string_view::npos;
            if (language == SourceLanguage::Rust)
                return prefix.find("fn ") != std::string_view::npos;
            if (language != SourceLanguage::Cpp && language != SourceLanguage::C &&
                language != SourceLanguage::Java)
                return false;
            const size_t boundary = prefix.find_last_of(";{}");
            const auto declaration = boundary == std::string_view::npos
                ? prefix : prefix.substr(boundary + 1);
            if (declaration.find('=') != std::string_view::npos ||
                declaration.find("return ") != std::string_view::npos ||
                declaration.find("throw ") != std::string_view::npos)
                return false;
            return std::ranges::any_of(declaration, identifier_start);
        }

        std::string_view identifier_category_token(
            std::string_view text, size_t start, size_t end)
        {
            if (std::isupper(static_cast<unsigned char>(text[start])))
                return "global";

            const auto prefix = text.substr(0, start);
            const auto suffix = text.substr(end);
            size_t for_position = std::string_view::npos;
            for (const auto spelling : {"for (", "for(", "for "})
            {
                const size_t candidate = prefix.rfind(spelling);
                if (candidate != std::string_view::npos &&
                    (for_position == std::string_view::npos || candidate > for_position))
                    for_position = candidate;
            }
            const size_t last_close_paren = prefix.rfind(')');
            if (for_position != std::string_view::npos &&
                (last_close_paren == std::string_view::npos || last_close_paren < for_position) &&
                prefix.substr(for_position).find(';') == std::string_view::npos)
            {
                const size_t next = next_non_space(text, end);
                if (next < text.size() &&
                    (text[next] == '=' || text[next] == ':' || text[next] == ';'))
                    return "loop";
                if (suffix.starts_with(" in "))
                    return "loop";
            }

            const size_t open_paren = prefix.rfind('(');
            if (open_paren != std::string_view::npos &&
                (last_close_paren == std::string_view::npos || last_close_paren < open_paren))
            {
                const auto declaration = prefix.substr(0, open_paren);
                const bool named_declaration = declaration.find("def ") != std::string_view::npos ||
                    declaration.find("fn ") != std::string_view::npos;
                size_t words = 0;
                bool in_word = false;
                for (const char ch : declaration)
                {
                    const bool word_char = identifier_continue(ch);
                    if (word_char && !in_word)
                        ++words;
                    in_word = word_char;
                }
                const bool expression_prefix = declaration.find('=') != std::string_view::npos ||
                    declaration.find('.') != std::string_view::npos ||
                    declaration.find("::") != std::string_view::npos;
                if (named_declaration || (words >= 2 && !expression_prefix))
                    return "param";
            }

            const size_t next = next_non_space(text, end);
            if (next < text.size() && text[next] == '=')
                return "local";
            return "global";
        }

        std::string assigned_identifier_token(
            std::string_view word,
            std::string_view category,
            IdentifierScopeState& state)
        {
            const std::string name{word};
            if (category == "param")
            {
                if (const auto pending = state.pending_parameters.find(name);
                    pending != state.pending_parameters.end())
                    return pending->second;
            }
            else if (category == "local" || category == "loop")
            {
                if (const auto existing = state.scopes.back().find(name);
                    existing != state.scopes.back().end())
                    return existing->second;
            }
            else
            {
                if (const auto pending = state.pending_parameters.find(name);
                    pending != state.pending_parameters.end())
                    return pending->second;
                for (auto scope = state.scopes.rbegin(); scope != state.scopes.rend(); ++scope)
                    if (const auto existing = scope->find(name); existing != scope->end())
                        return existing->second;
            }

            const std::string prefix = "<" + std::string{category == "loop" ? "LOOP_" :
                category == "local" ? "LOCAL_" :
                category == "param" ? "PARAM_" : "GLOBAL_"};
            const size_t capacity = category == "loop"
                ? static_cast<size_t>(IdentifierCategoryCount::LOOP_VARIABLES)
                : category == "local"
                    ? static_cast<size_t>(IdentifierCategoryCount::LOCALS)
                    : category == "param"
                        ? static_cast<size_t>(IdentifierCategoryCount::PARAMETERS)
                        : static_cast<size_t>(IdentifierCategoryCount::GLOBALS);
            std::vector<bool> used(capacity, false);
            const auto mark_used = [&](const auto& assignments) {
                for (const auto& [_, token] : assignments)
                {
                    if (!token.starts_with(prefix) || token.ends_with("OVERFLOW>"))
                        continue;
                    const auto digits = std::string_view{token}.substr(
                        prefix.size(), token.size() - prefix.size() - 1);
                    const size_t index = static_cast<size_t>(std::stoul(std::string{digits}));
                    if (index < used.size())
                        used[index] = true;
                }
            };
            for (const auto& scope : state.scopes)
                mark_used(scope);
            mark_used(state.pending_parameters);
            const auto overflow = category == "loop" ? loop_overflow_token :
                category == "local" ? local_overflow_token :
                category == "param" ? param_overflow_token : global_overflow_token;
            const auto available = std::ranges::find(used, false);
            const std::string token = available != used.end()
                ? prefix + std::to_string(static_cast<size_t>(available - used.begin())) + ">"
                : std::string{overflow};
            if (category == "global")
                state.scopes.front().emplace(name, token);
            else if (category == "param")
                state.pending_parameters.emplace(name, token);
            else
                state.scopes.back().emplace(name, token);
            return token;
        }

        size_t identifier_marker_length(std::string_view text, size_t position)
        {
            if (text.substr(position).starts_with(field_access_token))
                return field_access_token.size();
            for (const auto prefix : {"<LOOP_", "<LOCAL_", "<PARAM_", "<GLOBAL_"})
            {
                if (!text.substr(position).starts_with(prefix))
                    continue;
                const size_t end = text.find('>', position + std::string_view{prefix}.size());
                return end == std::string_view::npos ? 0 : end + 1 - position;
            }
            return 0;
        }
    }

    std::string abstract_source_for_model(
        std::string_view text, SourceLanguage language, IMCP& mcp,
        IdentifierScopeState* identifier_scopes)
    {
        IdentifierScopeState local_identifier_scopes;
        auto& scopes = identifier_scopes != nullptr
            ? *identifier_scopes : local_identifier_scopes;
        std::string out;
        out.reserve(text.size());
        bool expect_library_name = false;
        size_t i = 0;
        while (i < text.size())
        {
            bool matched_control = false;
            for (const auto marker : {string_token, mcp_start_token, mcp_end_token})
            {
                if (text.substr(i).starts_with(marker))
                {
                    out += marker;
                    i += marker.size();
                    matched_control = true;
                    break;
                }
            }
            if (matched_control)
                continue;
            if (const size_t marker_length = identifier_marker_length(text, i); marker_length > 0)
            {
                out += text.substr(i, marker_length);
                i += marker_length;
                continue;
            }
            if (text.substr(i).starts_with(legacy_identifier_token))
            {
                out += global_overflow_token;
                i += legacy_identifier_token.size();
                continue;
            }

            if (text.substr(i).starts_with("//"))
            {
                out += text.substr(i);
                break;
            }
            if (text.substr(i).starts_with("/*"))
            {
                const size_t end = text.find("*/", i + 2);
                const size_t length = end == std::string_view::npos ? text.size() - i : end + 2 - i;
                out += text.substr(i, length);
                i += length;
                continue;
            }
            if ((language == SourceLanguage::Python || language == SourceLanguage::Shell) && text[i] == '#')
            {
                out += text.substr(i);
                break;
            }
            if ((language == SourceLanguage::Cpp || language == SourceLanguage::C) && text[i] == '#')
            {
                size_t directive_end = i + 1;
                while (directive_end < text.size() && std::isspace(static_cast<unsigned char>(text[directive_end])))
                    ++directive_end;
                const size_t word_end = directive_end < text.size() && identifier_start(text[directive_end])
                    ? identifier_end(text, directive_end) : directive_end;
                const auto directive = text.substr(directive_end, word_end - directive_end);
                out += text.substr(i, word_end - i);
                i = word_end;
                if (directive == "include")
                {
                    while (i < text.size() && (text[i] == ' ' || text[i] == '\t'))
                        out += text[i++];
                    if (!text.substr(i).starts_with(mcp_start_token) && i < text.size() && (text[i] == '<' || text[i] == '"'))
                    {
                        const size_t literal_start = i;
                        const char close = text[i] == '<' ? '>' : '"';
                        const size_t end = text.find(close, i + 1);
                        if (end != std::string_view::npos)
                        {
                            const SourceContext context{text, literal_start, end + 1 - literal_start};
                            mcp.record_seen_string(context, context.value());
                            mcp.record_seen_mcp(context, context.value());
                            out += mcp_start_token;
                            out += string_token;
                            out += mcp_end_token;
                            i = end + 1;
                        }
                    }
                }
                continue;
            }
            if (text[i] == '"' || text[i] == '\'' || text[i] == '`')
            {
                const size_t literal_start = i;
                const char quote = text[i++];
                while (i < text.size())
                {
                    if (text[i] == '\\' && i + 1 < text.size())
                        i += 2;
                    else if (text[i++] == quote)
                        break;
                }
                const SourceContext context{text, literal_start, i - literal_start};
                mcp.record_seen_string(context, context.value());
                out += string_token;
                continue;
            }
            if (identifier_start(text[i]))
            {
                const size_t first_end = identifier_end(text, i);
                size_t chain_end = first_end;
                size_t part_count = 1;
                while (true)
                {
                    size_t separator = next_non_space(text, chain_end);
                    size_t after_separator = separator;
                    if (text.substr(separator).starts_with("::"))
                        after_separator += 2;
                    else if (text.substr(separator).starts_with("->"))
                        after_separator += 2;
                    else if (separator < text.size() && text[separator] == '.')
                        ++after_separator;
                    else
                        break;
                    after_separator = next_non_space(text, after_separator);
                    if (after_separator >= text.size() || !identifier_start(text[after_separator]))
                        break;
                    chain_end = identifier_end(text, after_separator);
                    ++part_count;
                }
                if (part_count > 1)
                {
                    if (is_function_declaration_name(text, i, chain_end, language))
                        scopes.pending_function_scope = true;
                    mcp.record_seen_mcp(
                        SourceContext{text, i, chain_end - i}, text.substr(i, chain_end - i));
                    out += mcp_start_token;
                    size_t position = i;
                    while (position < chain_end)
                    {
                        if (identifier_start(text[position]))
                        {
                            const size_t end = identifier_end(text, position);
                            mcp.record_seen_identifier(
                                SourceContext{text, position, end - position},
                                text.substr(position, end - position));
                            size_t separator_end = position;
                            while (separator_end > i && std::isspace(
                                static_cast<unsigned char>(text[separator_end - 1])))
                                --separator_end;
                            const bool field_access =
                                (separator_end > i && text[separator_end - 1] == '.') ||
                                (separator_end >= i + 2 &&
                                 (text.substr(separator_end - 2, 2) == "->" ||
                                  (text.substr(separator_end - 2, 2) == "::" &&
                                   (language == SourceLanguage::Cpp ||
                                    language == SourceLanguage::C))));
                            out += field_access ? field_access_token : assigned_identifier_token(
                                text.substr(position, end - position), "global", scopes);
                            position = end;
                        }
                        else
                            out += text[position++];
                    }
                    out += mcp_end_token;
                    i = chain_end;
                    expect_library_name = false;
                    continue;
                }

                const auto word = text.substr(i, first_end - i);
                if (is_language_keyword(word, language_name(language)))
                {
                    out += word;
                    expect_library_name = word == "import" || word == "from" || word == "use";
                    if (word == "class" || word == "struct")
                        scopes.pending_class_scope = true;
                }
                else if (expect_library_name ||
                         (next_non_space(text, first_end) < text.size() && text[next_non_space(text, first_end)] == '('))
                {
                    if (is_function_declaration_name(text, i, first_end, language))
                    {
                        const bool inside_class = std::ranges::any_of(scopes.class_scopes, [](bool value) {
                            return value;
                        });
                        const bool inside_function = std::ranges::any_of(scopes.function_scopes, [](bool value) {
                            return value;
                        });
                        if (!inside_class && !inside_function)
                            scopes.scopes.front().clear();
                        scopes.pending_function_scope = true;
                    }
                    const SourceContext context{text, i, first_end - i};
                    mcp.record_seen_identifier(context, word);
                    mcp.record_seen_mcp(context, word);
                    out += mcp_start_token;
                    out += assigned_identifier_token(word, "global", scopes);
                    out += mcp_end_token;
                    expect_library_name = false;
                }
                else
                {
                    mcp.record_seen_identifier(SourceContext{text, i, first_end - i}, word);
                    out += assigned_identifier_token(
                        word, identifier_category_token(text, i, first_end), scopes);
                }
                i = first_end;
                continue;
            }
            if (text[i] == '{')
            {
                scopes.scopes.emplace_back(std::move(scopes.pending_parameters));
                scopes.pending_parameters.clear();
                scopes.class_scopes.push_back(scopes.pending_class_scope);
                scopes.function_scopes.push_back(scopes.pending_function_scope);
                scopes.pending_class_scope = false;
                scopes.pending_function_scope = false;
            }
            else if (text[i] == '}' && scopes.scopes.size() > 1)
            {
                scopes.scopes.pop_back();
                scopes.class_scopes.pop_back();
                scopes.function_scopes.pop_back();
            }
            else if (text[i] == ';')
            {
                scopes.pending_parameters.clear();
                scopes.pending_class_scope = false;
                scopes.pending_function_scope = false;
            }
            out += text[i++];
        }
        return out;
    }

    std::string resolve_model_placeholders(std::string_view text, IMCP& mcp)
    {
        std::string result;
        result.reserve(text.size());
        size_t position = 0;
        while (position < text.size())
        {
            if (text.substr(position).starts_with(mcp_start_token))
            {
                const size_t end = text.find(mcp_end_token, position + mcp_start_token.size());
                if (end != std::string_view::npos)
                {
                    const size_t section_end = end + mcp_end_token.size();
                    const MCPContext context{text, position, section_end - position};
                    result += mcp.map_mcp(context);
                    position = section_end;
                    continue;
                }
            }
            const size_t identifier_length = identifier_marker_length(text, position);
            const auto identifier_marker = text.substr(position, identifier_length);
            const bool identifier = identifier_length > 0;
            const bool string = text.substr(position).starts_with(string_token);
            if (!identifier && !string)
            {
                result += text[position++];
                continue;
            }

            const auto marker = identifier ? identifier_marker : string_token;
            const MCPContext context{text, position, marker.size()};
            result += identifier ? mcp.map_identifier(context) : mcp.map_string(context);
            position += marker.size();
        }
        return result;
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
                if (is_validation[index])
                {
                    const size_t sequence_length = static_cast<size_t>(windows[index].line.size());
                    const size_t maximum_heads = static_cast<size_t>(MultiTokenPredictionIndex::MAX);
                    windows[index].context_length = static_cast<PositionIndex>(
                        std::max<size_t>(1, sequence_length - maximum_heads));
                }
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
        : m_filters(filters), m_mcp(m_default_mcp)
    {
        if (!s_log_file.is_open())
            set_tokenization_log_file("tokenization.log");
    }

    Corpus::Corpus(const std::vector<std::string>& filters, IMCP& mcp)
        : m_filters(filters), m_mcp(mcp)
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
        if (language == SourceLanguage::Python)
        {
            size_t indentation_position = 0;
            size_t indentation = 0;
            while (indentation_position < text.size() &&
                   (text[indentation_position] == ' ' || text[indentation_position] == '\t'))
            {
                indentation += text[indentation_position] == '\t' ? 4 : 1;
                ++indentation_position;
            }
            const bool has_code = indentation_position < text.size() && text[indentation_position] != '#';
            if (has_code)
            {
                while (state.identifier_scopes.indentation_levels.size() > 1 &&
                       indentation < state.identifier_scopes.indentation_levels.back())
                {
                    state.identifier_scopes.indentation_levels.pop_back();
                    state.identifier_scopes.scopes.pop_back();
                    state.identifier_scopes.class_scopes.pop_back();
                    state.identifier_scopes.function_scopes.pop_back();
                }
                if (indentation > state.identifier_scopes.indentation_levels.back())
                {
                    state.identifier_scopes.indentation_levels.push_back(indentation);
                    state.identifier_scopes.scopes.emplace_back(
                        std::move(state.identifier_scopes.pending_parameters));
                    state.identifier_scopes.pending_parameters.clear();
                    state.identifier_scopes.class_scopes.push_back(
                        state.identifier_scopes.pending_class_scope);
                    state.identifier_scopes.function_scopes.push_back(
                        state.identifier_scopes.pending_function_scope);
                    state.identifier_scopes.pending_class_scope = false;
                    state.identifier_scopes.pending_function_scope = false;
                }
            }
        }
        const auto append_raw_text = [&](std::string_view part) {
            const auto tokens = get_token_ids(std::string{part});
            for (const auto position : enum_iterator1D<PositionIndex>(tokens.size()))
                result.push_back(tokens[position]);
        };
        const auto append_source_text = [&](std::string_view part) {
            append_raw_text(abstract_source_for_model(
                part, language, m_mcp, &state.identifier_scopes));
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
                append_raw_text(std::string_view{text}.substr(segment_start, position - segment_start));
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

            append_source_text(std::string_view{text}.substr(segment_start, position - segment_start));
            const size_t delimiter_size = current == '#' ? 1 : 2;
            result.push_back(line_start
                ? TokenID::LINE_COMMENT_START
                : TokenID::BLOCK_COMMENT_START);
            position += delimiter_size;
            segment_start = position;
            if (line_start)
            {
                append_raw_text(std::string_view{text}.substr(segment_start));
                result.push_back(TokenID::LINE_COMMENT_END);
                state.line_comment_on_last_line = true;
                return result;
            }
            state.block_depth = 1;
        }

        if (state.block_depth > 0)
            append_raw_text(std::string_view{text}.substr(segment_start));
        else
            append_source_text(std::string_view{text}.substr(segment_start));
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
