#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <optional>

#include <nlohmann/json_fwd.hpp>
#include <LayerPrimitives.hpp>

namespace rllm
{
    enum class SourceLanguage
    {
        Cpp,
        C,
        Python,
        Rust,
        Java,
        Shell,
        Unknown
    };

    TokenID language_token(SourceLanguage language);
    std::optional<SourceLanguage> parse_source_language(std::string_view name);
    struct CommentLexState
    {
        size_t block_depth = 0;
        bool line_comment_on_last_line = false;
    };
    void set_tokenization_log_file(const std::string& filename);

    struct WindowExample
    {
        CpuInputLine line;
        PositionIndex context_length;
        size_t source_index = 0;
        bool starts_in_block_comment = false;
    };

    using FileTokenSequence = std::vector<TokenID>;

    /** Build bounded all-position training windows from corpus lines.
     * Windows never cross a line boundary. Consecutive windows overlap by one
     * token so every stride-selected next-token boundary can be supervised.
     * context_length is the number of input rows; the final token is a target.
     */
    std::vector<WindowExample> make_line_windows(
        const std::vector<CpuInputLine>& lines,
        size_t window_size,
        size_t stride,
        bool reverse = false
    );

    /** Build windows that may span lines but never cross a file boundary. */
    std::vector<WindowExample> make_file_windows(
        const std::vector<FileTokenSequence>& files,
        size_t window_size,
        size_t stride,
        bool reverse = false
    );

    struct FileWindowTrainingSplit
    {
        std::vector<WindowExample> training_windows;
        std::vector<WindowExample> validation_windows;
        size_t full_validation_window_count = 0;
        size_t split_file_count = 0;
    };

    /** Split the windows of every sufficiently large file between training
     * and validation. The validation cap is distributed across files. */
    FileWindowTrainingSplit make_file_window_training_split(
        const std::vector<FileTokenSequence>& files,
        size_t window_size,
        size_t stride,
        size_t validation_percent,
        size_t maximum_count,
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
                struct FileTrainingSplit
                {
                        std::vector<FileTokenSequence> training_files;
                        std::vector<FileTokenSequence> validation_files;
                };

        Corpus(const std::vector<std::string>& filters);
        void load_files_from_dir(
            const std::string& train_corpus_dir,
            size_t source_index = 0,
            double source_weight = 1.0);

        using visitor_fn_t       = std::function<void(const CpuInputLine&)>;
        using token_visitor_fn_t = std::function<void(TokenID)>;

        CpuInputLine get_token_ids(const std::string& text) const;
        CpuInputLine get_token_ids(
            const std::string& text,
            SourceLanguage language,
            CommentLexState& state) const;
        Token get_token_from_id(TokenID id) const;
        std::optional<std::string> get_line(const CpuInputLine& line) const;

        std::vector<CpuInputLine> get_suitable_training_lines() const;
        std::vector<FileTokenSequence> get_file_token_sequences() const;
        std::vector<size_t> get_file_source_indices() const;
        std::vector<SourceLanguage> get_file_languages() const;
        const std::vector<double>& source_weights() const { return m_source_weights; }
        TrainingSplit get_deterministic_training_split(size_t validation_percent = 20) const;
        FileTrainingSplit get_deterministic_file_split(size_t validation_percent = 20) const;

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
            TokenData(std::string filename, size_t source_index)
                : filename(std::move(filename)), m_source_index(source_index)
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

            const FileTokenSequence& tokens() const { return m_tokens_in_file; }
            size_t source_index() const { return m_source_index; }
            SourceLanguage language() const;

            size_t number_of_lines() const
            {
                return m_lines.size();
            }

          private:
            std::string filename;
            std::vector<CpuInputLine> m_lines; // positions of the token in the corpus
            std::vector<TokenID> m_tokens_in_file; // the actual token IDs in the file, in order
            size_t m_source_index;
        };

        std::vector<TokenData> m_token_list;
        std::vector<double> m_source_weights;
        const std::vector<std::string>& m_filters;
        mutable size_t m_tokenization_errors = 0;
    };

} // namespace rllm
