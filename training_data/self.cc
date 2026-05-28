#  Parallel backend selection
# cmake -DPARALLEL_BACKEND=openmp|fastfork|sequential  (default: fastfork)
set(PARALLEL_BACKEND "fastfork" CACHE STRING "Parallel backend: openmp, fastfork, or sequential")
set_property(CACHE PARALLEL_BACKEND PROPERTY STRINGS openmp fastfork sequential)

if(NOT PARALLEL_BACKEND MATCHES "^(openmp|fastfork|sequential)$")
    message(FATAL_ERROR "PARALLEL_BACKEND must be one of: openmp, fastfork, sequential (got '${PARALLEL_BACKEND}')")
endif()

if(PARALLEL_BACKEND STREQUAL "fastfork")
    message(STATUS "Parallel backend: fastfork")
elseif(PARALLEL_BACKEND STREQUAL "openmp")
    find_package(OpenMP REQUIRED)
    message(STATUS "Parallel backend: OpenMP")
else()
    message(STATUS "Parallel backend: sequential (single-threaded)")
endif()

#  Helper to create an rllm static library variant
# Usage: rllm_define_lib(<target> <extra compile flags   >)
function(rllm_define_lib target_name)
    add_library(${target_name} STATIC)

    target_sources(${target_name} PRIVATE
        RLLM.cc
        Corpus.cpp
        InputLayer.cc
        TransformerBlock.cc
        OutputLayer.cc
        NeuralNetwork.cc
        serialization.cc
        parallel.cc
        "${CMAKE_BINARY_DIR}/generated/tokenizer_map.cc"
    )

    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
    )

    # Warning flags common to all variants; optimization flags come from ARGN.
    target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic -Wstack-usage=10000 -fopenmp-simd -Wno-narrowing
        ${ARGN}
    )
    #target_compile_definitions(${target_name} PUBLIC RLLM_ENABLE_OVERFLOW_CHECK_ADD)

    add_dependencies(${target_name} generate_tokenizer_map)
    target_link_libraries(${target_name} PUBLIC nlohmann_json::nlohmann_json)

    if(PARALLEL_BACKEND STREQUAL "fastfork")
        target_sources(${target_name} PRIVATE fastfork/fastfork.cc)
        target_compile_definitions(${target_name} PUBLIC USE_FASTFORK)
        target_link_libraries(${target_name} PUBLIC hwloc)
    elseif(PARALLEL_BACKEND STREQUAL "openmp")
        target_compile_definitions(${target_name} PUBLIC USE_OPENMP)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

rllm_define_lib(rllm_lib_opt   -march=native -O3)
rllm_define_lib(rllm_lib_debug -O0 -g)

add_library(rllm_lib INTERFACE)
target_link_libraries(rllm_lib INTERFACE
    $<IF:$<CONFIG:Release>,rllm_lib_opt,rllm_lib_debug>
)

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
    bool log_info_enabled = true;
    bool log_debug_enabled = true;

    static std::ofstream s_log_file{"tokenization.log"};

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
                for (const auto i : enum_iterator<PositionIndex>(input_line.size()))
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
                if (ch != ' ')
                {
                    // spaces have no explicit token, we just skip them without logging,
                    // but log other unmatched characters as warnings since they may
                    // indicate a problem with the tokenizer map.
                    LOG_ERROR("Warning: No token matched for character '{}', skipping it", ch);
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
        assert(id < TokenID::MAX);
        return tokenizer_map[id].str;
    }

    std::optional<std::string> Corpus::get_line(const InputLine& line) const
    {
        std::string result;
        for (const auto i : enum_iterator<PositionIndex>(line.size()))
        {
            auto& token_id = line[i];
            auto& token_info = tokenizer_map[token_id];
            if (token_id == TokenID::UNKNOWN_TOKEN_ID)
            {
                return std::nullopt; // line contains unknown token ID, cannot convert to string
            }
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

    std::vector<InputLine> Corpus::get_suitable_training_lines() const
    {
        std::vector<InputLine> training_lines;
        std::unordered_set<std::string> seen_training_line_keys;

        auto make_line_key = [](const InputLine& line) {
            std::string key;
            key.reserve(static_cast<size_t>(line.size()) * 6);
            for (const auto i : enum_iterator<PositionIndex>(line.size()))
            {
                key += std::to_string(static_cast<int>(line[i]));
                key.push_back(',');
            }
            return key;
        };

        assert(!m_token_list.empty());
        this->visit_lines([&](const InputLine& line) {
            // Strip the trailing TOK_NEWLINE that the corpus appends to every line.
            // Keeping it would make \n the dominant training target (it ends every line),
            // causing the model to collapse to always predicting \n.
            InputLine stripped = line;
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


} // namespace rllm#include <InputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rllm
{
    void InputLayer::reset_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok][d] = RLMM_ZERO;
    }

    void InputLayer::set_random_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok][d] = static_cast<rlmm_float>(get_random_value(-0.1f, 0.1f));
    }

    void InputLayer::propagate_forward(
        const InputLine& input,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        h.set_rows(static_cast<PositionIndex>(input.size()));

        const float D = static_cast<float>(EmbeddingDimension::MAX);
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const TokenID tok = input[pos];
            const auto& embed = m_embeddings[tok];

            for (const auto di : enum_iterator<EmbeddingDimension>())
            {
                const int di_int = static_cast<int>(di);
                const float emb_val = embed[di];
                const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(di_int & ~1) / D);
                const float pe = (di_int % 2 == 0) ? std::sin(static_cast<float>(pos) * freq)
                                                   : std::cos(static_cast<float>(pos) * freq);
                h[pos, di] = static_cast<rlmm_float>(emb_val + pe);
            }
        }
    }

    void InputLayer::propagate_backward(
        const InputLine& input,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dh,
        float learning_rate
    )
    {
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const auto tok = input[pos];
            auto& embed = m_embeddings[tok];

            for (const auto di : enum_iterator<EmbeddingDimension>())
            {
                auto& e = embed[di];
                e = math::clamp(e + learning_rate * static_cast<rlmm_float>(dh[pos, di]), RLMM_NEG_ONE, RLMM_ONE);
            }
        }
    }
} // namespace rllm
#include <print>
#include <string>

#include <parallel.hpp>
#include <RLLM.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>

int main(int argc, char* argv[])
{
    std::srand(0);
    parallel::init_parallel();
#ifdef NDEBUG
    std::println("Build type: Release (NDEBUG defined)");
#else
    std::println("Build type: Debug (NDEBUG not defined)");
#endif
    std::string train_corpus_dir = "training_data";
    std::vector<std::string> filters;
    bool train_mode = false;
    std::string output_filename = "model.json";
    std::optional<std::string> input_filename;
    std::optional<std::string> one_shot_prompt;
    int num_layers = 4;
    bool verbose = false;
    size_t num_epochs = 1000;
    auto method = rllm::TrainingMethod::TWO_TOK;
    int window_size = 2;
    std::optional<std::chrono::seconds> checkpointing_interval = std::chrono::seconds{120};

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--filter") == 0 && ((i + 1) < argc))
        {
            filters.push_back(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--layers") == 0 && ((i + 1) < argc))
        {
            num_layers = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--checkpoint-interval") == 0 && ((i + 1) < argc))
        {
            const int seconds = std::atoi(argv[++i]);
            if (seconds <= 0)
                checkpointing_interval = std::nullopt;
            else
                checkpointing_interval = std::chrono::seconds{seconds};
        }
        else if (std::strcmp(argv[i], "--train-dir") == 0 && ((i + 1) < argc))
        {
            train_corpus_dir = argv[++i];
        }
        else if (std::strcmp(argv[i], "--epochs") == 0 && ((i + 1) < argc))
        {
            num_epochs = static_cast<size_t>(std::atoi(argv[++i]));
        }
        else if (std::strcmp(argv[i], "-o") == 0 && ((i + 1) < argc))
        {
            output_filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "-i") == 0 && ((i + 1) < argc))
        {
            input_filename = argv[++i];
        }        else if (std::strcmp(argv[i], "-c") == 0 && ((i + 1) < argc))
        {
            one_shot_prompt = argv[++i];
        }        else if (std::strcmp(argv[i], "--method") == 0 && ((i + 1) < argc))
        {
            const std::string m = argv[++i];
            if (m == "two_tok")
                method = rllm::TrainingMethod::TWO_TOK;
            else if (m == "three_tok")
                method = rllm::TrainingMethod::THREE_TOK;
            else if (m == "increasingly_longer")
                method = rllm::TrainingMethod::INCREASINGLY_LONGER_SEQUENCES;
            else if (m == "random_line_random_len")
                method = rllm::TrainingMethod::RANDOM_LINE_RANDOM_LEN;
            else if (m.starts_with("window:"))
            {
                const int n = std::atoi(m.c_str() + 7);
                if (n < 2)
                {
                    std::println("window:<N> requires N >= 2, got '{}'", m);
                    return 1;
                }
                method = rllm::TrainingMethod::WINDOW;
                window_size = n;
            }
            else
            {
                std::println(
                    "Unknown training method '{}'. Valid values: two_tok, three_tok, increasingly_longer, random_line_random_len, window:<N>", m
                );
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--train") == 0)
        {
            train_mode = true;
        }
        else if (std::strcmp(argv[i], "--verbose") == 0)
        {
            verbose = true;
        }
        else
        {
            std::println(
                "Usage: {} [--train] [--file <filename>] [--verbose] [--filter <filter>]\n"
                "          [--train-dir <directory>]\n"
                "          [--method <two_tok|three_tok|increasingly_longer|random_line_random_len|window:<N>>]\n"
                "  --train         Run in training mode (default is prompt mode)\n"
                "  --train-dir <directory>  Directory containing training text files (default is '{}')\n"
                "  -i <filename>  Specify the model file to load (trainer will init the model if not provided)\n"
                "  -c <string>    Run prompt mode with this string, print predictions, then exit\n"
                "  -o <filename>  Specify the model file to save (default is '{}')\n"
                "  --verbose       Enable verbose output\n"
                "  --filter <filter>  Specify a filter to apply\n"
                "  --epochs <n>    Number of training epochs (default: {})\n"
                "  --method        Training method (default: {})\n"
                "  --checkpoint-interval <seconds>  Extra timed checkpoint cadence; <=0 disables (default: {}s)\n"
                "  window:<N>      Sliding window of N tokens (N >= 2)",
                argv[0],
                train_corpus_dir,
                output_filename,
                num_epochs,
                rllm::training_method_to_string(method),
                checkpointing_interval.has_value() ? std::to_string(checkpointing_interval->count()) : "disabled"
            );
            return 1;
        }
    }

    rllm::RLLM llm(filters);
    if (train_mode)
    {
        llm.train_mode(
            input_filename,
            output_filename,
            num_layers,
            verbose,
            method,
            checkpointing_interval,
            window_size,
            num_epochs,
            train_corpus_dir
        );
    }
    else
    {
        llm.prompt_mode(input_filename ? *input_filename : output_filename, one_shot_prompt);
    }

    return 0;
}#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>

namespace rllm
{
    const char* training_method_to_string(TrainingMethod method)
    {
        switch (method)
        {
        case TrainingMethod::TWO_TOK:
            return "two_tok";
        case TrainingMethod::THREE_TOK:
            return "three_tok";
        case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
            return "increasingly_longer";
        case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
            return "random_line_random_len";
        case TrainingMethod::WINDOW:
            return "window";
        }
        return "UNKNOWN";
    }
} // namespace rllm

static std::ofstream s_nn_log;
#define LOG_INFO(...) (s_nn_log << std::format(__VA_ARGS__) << '\n' << std::flush)


#define LOG_INFO_EVERY_N(...) \
    do \
    { \
        static int counter = 0; \
        if (counter++ % 100 == 0) \
        { \
            LOG_INFO(__VA_ARGS__); \
        } \
    } while (0)

namespace rllm
{
    void set_nn_log_file(const std::string& filename)
    {
        if (s_nn_log.is_open())
            s_nn_log.close();
        s_nn_log.open(filename);
    }
} // namespace rllm

namespace rllm
{
    // Number of gradient-update passes over all layers per training example per epoch.
    constexpr size_t NUMBER_OF_LAYER_VISITS_PER_EXAMPLE = 8;

    // Layers

    struct NeuralNetworkForwardWorkspace
    {
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h;
        fixed_size_vector<rlmm_float, EmbeddingDimension> h_last;
        explicit NeuralNetworkForwardWorkspace(PositionIndex seq_len) : h(seq_len) {}
    };

    void NeuralNetwork::propagate_forward(const InputLine& input)
    {
        assert(!m_transformer_blocks.empty());

        m_last_input = input;
        m_seq_len = input.size();

        auto ws = std::make_unique<NeuralNetworkForwardWorkspace>(m_seq_len);

        m_input_layer.propagate_forward(input, ws->h);

        for (auto& block : m_transformer_blocks)
            block.forward(ws->h, m_seq_len);

        m_last_hidden = ws->h;

        // Project the last-position hidden state to vocabulary logits.
        // Given a string of N tokens, the model learns to predict the N+1'th token,
        // so the final output is based on the hidden state at the last input position.
        const auto last_pos = dec(m_seq_len);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->h_last[d] = ws->h[last_pos, d];
        m_output_layer.forward_from_hidden(ws->h_last);
    }


    // NeuralNetwork

    static void try_add_to_top_k(std::vector<OutputToken>& top_k, TokenID id, float value, size_t k)
    {
        if (top_k.size() < k)
        {
            top_k.emplace_back(id, value);
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
        else if (!top_k.empty() && value >= top_k.back().activation)
        {
            // Use >= so ties don't always leave the initial token IDs (0,1,2,...) in place.
            top_k.back() = {id, value};
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
    }

    static bool timed_checkpoint_due(
        const std::optional<std::chrono::seconds>& checkpointing_interval,
        std::chrono::steady_clock::time_point& last_checkpoint_at
    )
    {
        if (!checkpointing_interval.has_value())
            return false;

        const auto now = std::chrono::steady_clock::now();
        if ((now - last_checkpoint_at) < checkpointing_interval.value())
            return false;

        last_checkpoint_at = now;
        return true;
    }

    // Returns top-K tokens selected by logit, with probabilities normalized
    // over the returned top-K set. This keeps prompt-time probabilities stable
    // and informative even when the full-vocabulary softmax is extremely sharp.
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k) const
    {
        assert(!m_transformer_blocks.empty());
        if (top_k == 0)
            return {};

        // First, keep only top-K by raw logit (preserves correct rank order).
        std::vector<OutputToken> top_k_pairs;
        for (const auto i : enum_iterator<TokenID>())
        {
            const auto logit = m_output_layer.m_inputs[i];
            try_add_to_top_k(top_k_pairs, i, logit, top_k);
        }

        if (top_k_pairs.empty())
            return top_k_pairs;

        // Clamp the logit gap so no token in top-K is numerically invisible.
        // Without capping, a collapsed model produces 100% / 0.000e-25% which is
        // useless for display and prevents diverse sampling.
        // A gap cap of 5 nats allows at most a ~148:1 ratio between any two entries.
        static constexpr double MAX_LOGIT_GAP = 5.0;
        const double best_logit = static_cast<double>(top_k_pairs.front().activation);
        for (auto& entry : top_k_pairs)
        {
            const double capped = std::max(static_cast<double>(entry.activation), best_logit - MAX_LOGIT_GAP);
            entry.activation = static_cast<float>(capped);
        }

        // Stable softmax over the gap-capped logits.
        double sum_exp = 0.0;
        for (const auto& entry : top_k_pairs)
            sum_exp += std::exp(static_cast<double>(entry.activation) - best_logit);

        for (auto& entry : top_k_pairs)
        {
            const double p = std::exp(static_cast<double>(entry.activation) - best_logit) / sum_exp;
            entry.activation = static_cast<float>(p);
        }

        return top_k_pairs;
    }


    struct BackwardPropWorkspace
    {
        fixed_size_vector<rlmm_float, TokenID>                                   output_layer_delta;
        fixed_size_vector<rlmm_float, EmbeddingDimension>                        h_last_vec;
        fixed_size_vector<rlmm_float, EmbeddingDimension>                        dh_last;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>    dh;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>    din;
        explicit BackwardPropWorkspace(PositionIndex seq_len) : dh(seq_len), din(seq_len) {}
    };

    void NeuralNetwork::propagate_backward(const Score& score)
    {
        static constexpr float LEARNING_RATE = 0.003f;

        auto ws = std::make_unique<BackwardPropWorkspace>(m_seq_len);

        m_output_layer.compute_deltas(score, ws->output_layer_delta);

        // Backpropagate through LM head  get dL/dh_last
        const auto last_pos = dec(m_seq_len);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->h_last_vec[d] = m_last_hidden[last_pos, d];
        ws->dh_last = m_output_layer.backward_and_update(ws->output_layer_delta, ws->h_last_vec, LEARNING_RATE);

        // Initialise full-sequence gradient: zero everywhere except the last position
        ws->dh.fill(RLMM_ZERO);
        ws->din.fill(RLMM_ZERO);
        for (const auto d : enum_iterator<EmbeddingDimension>())
            ws->dh[last_pos, d] = ws->dh_last[d];

        // Backward through transformer blocks in reverse order
        for (int i = static_cast<int>(m_transformer_blocks.size()) - 1; i >= 0; --i)
        {
            m_transformer_blocks[i].backward(ws->dh, ws->din, LEARNING_RATE);
            ws->dh = ws->din;
        }

        // Update token embeddings
        m_input_layer.propagate_backward(m_last_input, ws->dh, LEARNING_RATE);
    }


    void NeuralNetwork::set_random_weights_and_connections()
    {
        m_input_layer.set_random_embeddings();
        for (auto& block : m_transformer_blocks)
            block.randomize();
        m_output_layer.set_random_weights();
    }


    void NeuralNetwork::dump_top_predictions()
    {
        int prediction_index = 0;
        const auto predicted_token_id_lists = get_best_output_token_ids(5);
        for (const auto& entry : predicted_token_id_lists)
        {
            const auto predicted_token = m_corpus.get_token_from_id(entry.token_id);
            LOG_INFO(
                "\t prediction[{} of {}] / pred:'{}' (id: '{}'), {}",
                prediction_index,
                predicted_token_id_lists.size(),
                predicted_token,
                entry.token_id,
                entry.activation
            );
            prediction_index++;
        }
    }

    // Compute cross-entropy loss: -log(softmax(logits)[target])
    // Only considers the active corpus tokens, not the full TokenID::MAX space.
    float NeuralNetwork::compute_loss(TokenID expected_output_token) const
    {
        const int n_tok = static_cast<int>(TokenID::MAX);
        float max_val = m_output_layer.m_inputs[TokenID::START];
#pragma omp simd reduction(max : max_val)
        for (int i = 0; i < n_tok; ++i)
            max_val = math::max(max_val, m_output_layer.m_inputs[static_cast<TokenID>(i)]);

        float sum_exp = 0.0f;
#pragma omp simd reduction(+ : sum_exp)
        for (int i = 0; i < n_tok; ++i)
        {
            const float term = std::exp(m_output_layer.m_inputs[static_cast<TokenID>(i)] - max_val);
            OVERFLOW_CHECK_ADD(sum_exp, term);
            sum_exp += term;
        }

        const float log_prob = m_output_layer.m_inputs[expected_output_token] - max_val - std::log(sum_exp);
        return -log_prob;
    }

    void NeuralNetwork::train_with_increasingly_longer_sequences(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations
    )
    {
        for (const auto& line_substring_length : enum_iterator<PositionIndex>(line_of_file.size()))
        {
            const auto line = line_of_file.sub_array(line_substring_length);
            if (line.empty())
                continue; // skip empty lines that can't be used for training

            const auto full_string_opt = m_corpus.get_line(line);
            assert(full_string_opt.has_value());
            const auto& full_string = *full_string_opt;

            if (static_cast<int>(line.size()) < 2)
            {
                continue; // skip too-short lines that can't be used for training
            }

            LOG_INFO("Training on line: '{}'", full_string);

            do_training(line, verbose, max_iterations);
        }
    }

    void NeuralNetwork::train_with_up_to_N(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations,
        int num_tokens
    )
    {
        assert(num_tokens >= 2);

        // If the line is too short for num_tokens, fall back to fewer tokens
        // (minimum 2 so there is always at least one input token and one target).
        if (static_cast<int>(line_of_file.size()) < num_tokens)
        {
            if (num_tokens > 2)
                train_with_up_to_N(line_of_file, verbose, max_iterations, num_tokens - 1);
            return;
        }

        const auto train_input = line_of_file.sub_array(static_cast<PositionIndex>(num_tokens));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line: '{}'", full_string);

        do_training(train_input, verbose, max_iterations);
    }

    void NeuralNetwork::train_with_random_len_from_start(
        const InputLine& line_of_file,
        bool verbose,
        size_t max_iterations,
        std::mt19937& rng
    )
    {
        const int line_len = static_cast<int>(line_of_file.size());
        if (line_len < 2)
            return;

        // Avoid over-training the shortest prefix for lines that have richer context.
        // Example: for "# include A", sampling length 2 trains only "# -> include"
        // and can drown out "# include -> A". For line_len >= 3, keep randomization
        // but require at least 3 tokens so the continuation target is learned.
        const int min_len = (line_len >= 3) ? 3 : 2;
        std::uniform_int_distribution<int> len_dist(min_len, line_len);
        const int random_len = len_dist(rng);

        const auto train_input = line_of_file.sub_array(static_cast<PositionIndex>(random_len));
        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());

        LOG_INFO("Training on random line prefix ({} toks): '{}'", random_len, *full_string_opt);

        // Reward longer correct sequences with a larger optimization budget.
        // len=2 keeps the base budget; longer prefixes scale up proportionally
        // (capped to avoid runaway per-example compute on large corpora).
        const size_t length_factor = std::max<size_t>(1, static_cast<size_t>(random_len) - 1);
        const size_t capped_factor = std::min<size_t>(length_factor, 4);
        const size_t effective_max_iterations = max_iterations * capped_factor;

        do_training(train_input, verbose, effective_max_iterations);
    }

    void NeuralNetwork::train_random_line_random_len_epoch(
        size_t epoch,
        const std::vector<InputLine>& training_lines,
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval,
        std::chrono::steady_clock::time_point& last_checkpoint_at,
        std::mt19937& rng
    )
    {
        const auto total_lines = training_lines.size();
        if (total_lines == 0)
            return;

        // Visit each line exactly once per epoch (in random order).
        // Sampling with replacement over-emphasizes some lines while skipping others,
        // which causes unstable learning and apparent forgetting.
        std::vector<size_t> line_indices(total_lines);
        std::iota(line_indices.begin(), line_indices.end(), 0);
        std::shuffle(line_indices.begin(), line_indices.end(), rng);

        for (size_t lines_visited = 1; lines_visited <= total_lines; ++lines_visited)
        {
            const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

            if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
            {
                std::println(
                    "creating timed checkpoint at epoch {}, line {}, total lines visited {}",
                    epoch,
                    lines_visited,
                    total_lines
                );
                save(std::format("models/checkpoint-{}.json", epoch * total_lines + lines_visited));
            }

            LOG_INFO(
                "Epoch[{}%] random-line[{}]: {:0.2f}% done",
                epoch / static_cast<float>(num_epochs) * 100.0f,
                lines_visited,
                progress * 100.0f
            );

            const auto& random_line = training_lines[line_indices[lines_visited - 1]];
            train_with_random_len_from_start(
                random_line,
                verbose,
                NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size(),
                rng
            );
        }
    }


    void NeuralNetwork::do_line_based_training(
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        std::vector<InputLine> training_lines = m_corpus.get_suitable_training_lines();

        // Multi-epoch training with shuffling prevents catastrophic forgetting:
        // each example only gets 8*num_layers gradient updates per
        // pass, so no single example can overwrite all the others.
        std::mt19937 rng{42};
        const auto total_lines = training_lines.size();
        auto last_checkpoint_at = std::chrono::steady_clock::now();

        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            if (m_training_method == TrainingMethod::RANDOM_LINE_RANDOM_LEN)
            {
                train_random_line_random_len_epoch(
                    epoch,
                    training_lines,
                    verbose,
                    num_epochs,
                    checkpointing_interval,
                    last_checkpoint_at,
                    rng
                );

                std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                save(std::format("models/checkpoint-epoch-{}.json", epoch));
                continue;
            }

            std::shuffle(training_lines.begin(), training_lines.end(), rng);

            size_t lines_visited = 0;
            for (const auto& line_of_file : training_lines)
            {
                lines_visited++;
                const float progress = static_cast<float>(lines_visited) / static_cast<float>(total_lines);

                if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                {
                    std::println(
                        "creating timed checkpoint at epoch {}, line {}, total lines visited {}",
                        epoch,
                        lines_visited,
                        total_lines
                    );
                    save(std::format("models/checkpoint-{}.json", epoch * total_lines + lines_visited));
                }

                LOG_INFO(
                    "Epoch[{}%] line[{}]: {:0.2f}% done",
                    epoch / static_cast<float>(num_epochs) * 100.0f,
                    lines_visited,
                    progress * 100.0f
                );

                switch (m_training_method)
                {
                case TrainingMethod::TWO_TOK:
                    train_with_up_to_N(
                        line_of_file, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size(), 2
                    );
                    break;

                case TrainingMethod::THREE_TOK:
                    train_with_up_to_N(
                        line_of_file, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size(), 3
                    );
                    break;

                case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                    train_with_increasingly_longer_sequences(
                        line_of_file, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size()
                    );
                    break;

                case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
                    // Handled in the random-line loop above.
                    assert(false);
                    break;

                case TrainingMethod::WINDOW:
                    // window methods don't use the line-based loop; handled separately below
                    assert(false);
                    break;
                }
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            save(std::format("models/checkpoint-epoch-{}.json", epoch));
        }
    }

    void NeuralNetwork::train_with_window(
        int window_size,
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        assert(window_size >= 2);

        // Collect the full flat token sequence from every corpus file.
        std::vector<TokenID> tokens;
        m_corpus.visit_flat_tokens([&](TokenID tok) {
            tokens.push_back(tok);
        });

        if (tokens.size() < static_cast<size_t>(window_size))
            return;

        // Each valid start index yields one training example.
        const size_t num_windows = tokens.size() - static_cast<size_t>(window_size) + 1;
        std::vector<size_t> indices(num_windows);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng{42};
        size_t total_windows_trained = 0;
        auto last_checkpoint_at = std::chrono::steady_clock::now();
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            LOG_INFO("Epoch[{}%]: {:0.2f}% done", epoch / static_cast<float>(num_epochs) * 100.0f, 0.0f);

            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t j = 0; j < num_windows; ++j)
            {
                const float progress = static_cast<float>(j) / static_cast<float>(num_windows);

                InputLine window;
                int current_try_len = 2 + random_int(0, window_size - 2); // random length between 2 and window_size
                for (int k = 0; k < current_try_len; ++k)
                {
                    window.push_back(tokens[indices[j] + static_cast<size_t>(k)]);
                }
                assert(window.size() == static_cast<PositionIndex>(current_try_len));

                total_windows_trained++;

                if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                {
                    std::println(
                        "creating timed checkpoint at epoch {}, window {}, total windows trained {}",
                        epoch,
                        j,
                        total_windows_trained
                    );
                    save(std::format("models/checkpoint-{}.json", total_windows_trained));
                }


                if (total_windows_trained % 100 == 0)
                {
                    const auto line_opt = m_corpus.get_line(window);
                    LOG_INFO(
                        "Epoch[{}%] window[{}]: {:0.2f}% done for '{}', successes: {}, failures: {}",
                        epoch / static_cast<float>(num_epochs) * 100.0f,
                        j,
                        progress * 100.0f,
                        line_opt.has_value() ? line_opt->c_str() : "unknown",
                        m_stats.num_learning_successes(),
                        m_stats.num_learning_failures()
                    );
                }

                do_training(window, verbose, NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * m_transformer_blocks.size());
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            save(std::format("models/checkpoint-epoch-{}.json", epoch));
        }
    }

    void NeuralNetwork::do_whole_corpus_window_based_training(
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        // Window methods operate on the flat token stream rather than per-line.
        assert(m_training_method == TrainingMethod::WINDOW);
        train_with_window(m_window_size, verbose, num_epochs, checkpointing_interval);
    }


    void NeuralNetwork::train(
        bool verbose,
        size_t num_epochs,
        const std::optional<std::string>& input_filename,
        const std::optional<std::chrono::seconds>& checkpointing_interval
    )
    {
        Statistics::TotalLearnRecorderScope total_learn_recorder_scope(m_stats);

        if (input_filename)
        {
            if (!load(*input_filename))
            {
                std::println("Failed to load model from '{}'", *input_filename);
                std::exit(1);
            }
            LOG_INFO("Loaded model from '{}'", *input_filename);
        }
        else
        {
            std::println("No input model specified, starting with random weights.");
            set_random_weights_and_connections();
        }

        const size_t vocab       = static_cast<size_t>(TokenID::MAX);
        const size_t d_model     = static_cast<size_t>(EmbeddingDimension::MAX);
        const size_t d_ff        = static_cast<size_t>(FFDimension::MAX);
        const size_t n_layers    = m_transformer_blocks.size();
        const size_t params_embed    = vocab * d_model;           // token embeddings
        const size_t params_lm_head  = vocab * d_model;           // LM head
        const size_t params_attn     = 4 * d_model * d_model;     // W_q, W_k, W_v, W_o per block
        const size_t params_ffn      = 3 * d_model * d_ff;        // W_gate, W_up, W_down per block (2*d_ff*d_model + d_model*d_ff)
        const size_t params_per_block = params_attn + params_ffn;
        const size_t total_params    = params_embed + params_lm_head + n_layers * params_per_block;

        LOG_INFO(
            "Training the neural network...\n"
            "\t $num_layers: {}\n"
            "\t $corpus_size: {}\n"
            "\t $total_params: {} ({:.2f}M)  [embed:{} lm_head:{} blocks:{}x{}]\n"
            "\t convergence threshold: {:.6f}\n"
            "\t fires nothing CE loss:  {:.6f}\n"
            "\t steps per example per epoch: {}\n"
            "\t num epochs: {}\n"
            "\t training method: {}\n",
            n_layers,
            vocab,
            total_params, static_cast<float>(total_params) / 1e6f,
            params_embed, params_lm_head, n_layers, params_per_block,
            m_convergence_threshold,
            m_fires_nothing_ce_loss,
            NUMBER_OF_LAYER_VISITS_PER_EXAMPLE * n_layers,
            num_epochs,
            training_method_to_string(m_training_method)
        );

        if (training_method_is_line_based())
        {
            do_line_based_training(verbose, num_epochs, checkpointing_interval);
        }
        else
        {
            do_whole_corpus_window_based_training(verbose, num_epochs, checkpointing_interval);
        }
    }


    void NeuralNetwork::do_training(const InputLine& train_output, bool verbose, size_t max_iterations)
    {
        // given a training example line [5,34,3,4,1], we use the first N-1 tokens
        // as input ([5,34,3,4]) and the last token as the expected output (1).
        assert(static_cast<int>(train_output.size()) >= 2);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        const bool trace_enabled = []() {
            const char* v = std::getenv("RLLM_TRACE_INCLUDE");
            return v != nullptr && std::string(v) != "0";
        }();
        const bool trace_this_example = trace_enabled && (full_string.find("#include") != std::string::npos);

        auto trace_probes = [&](const char* phase, size_t iter, float loss_value) {
            if (!trace_this_example)
                return;

            static const std::array<std::string, 2> probes = {"#", "#include"};
            for (const auto& probe : probes)
            {
                const auto probe_ids = m_corpus.get_token_ids(probe);
                if (probe_ids.empty())
                    continue;

                propagate_forward(probe_ids);
                const auto top5 = get_best_output_token_ids(5);

                if (top5.empty())
                {
                    LOG_INFO(
                        "[TRACE] phase='{}' iter={} train='{}' probe='{}' top1='<none>' loss={:.6f}",
                        phase,
                        iter,
                        full_string,
                        probe,
                        loss_value
                    );
                    continue;
                }

                const auto top1_tok = m_corpus.get_token_from_id(top5.front().token_id);
                LOG_INFO(
                    "[TRACE] phase='{}' iter={} train='{}' probe='{}' top1='{}' p={:.6f} loss={:.6f}",
                    phase,
                    iter,
                    full_string,
                    probe,
                    top1_tok,
                    top5.front().activation,
                    loss_value
                );

                for (size_t k = 0; k < top5.size(); ++k)
                {
                    LOG_INFO(
                        "[TRACE]   top{} token='{}' id={} p={:.6f}",
                        k,
                        m_corpus.get_token_from_id(top5[k].token_id),
                        top5[k].token_id,
                        top5[k].activation
                    );
                }
            }
        };

        // In multi-epoch training each call gets a small fixed budget (max_iterations).
        // We run all steps unconditionally; convergence emerges across epochs.
        float loss = 0.0f;
        for (size_t i = 0; i < max_iterations; ++i)
        {
            Score score;
            propagate_forward(train_input);
            m_output_layer.compute_score(score, expected_output_token);
            propagate_backward(score);
            loss = compute_loss(expected_output_token);

            if (trace_this_example && i < 4)
                trace_probes("post_step", i + 1, loss);

            if (loss < m_convergence_threshold)
            {
                LOG_INFO_EVERY_N(
                    "Convergence reached after {} steps for expected '{}', full string: '{}', input size: {}",
                    i + 1,
                    m_corpus.get_token_from_id(expected_output_token),
                    full_string,
                    static_cast<size_t>(train_input.size())
                );
                m_stats.record_learning_success();
                return;
            }


            if (verbose && i % 25 == 0)
            {
                const auto expected_token = m_corpus.get_token_from_id(expected_output_token);
                LOG_INFO(
                    "Training iteration[{}], wanted: '{}' ({}), full string: '{}'",
                    i,
                    expected_token,
                    expected_output_token,
                    full_string
                );
                LOG_INFO("  Loss: {:.6f}", loss);
                dump_top_predictions();
            }
        }

        LOG_INFO(
            "Steps exhausted ({}) for this line. loss = {:.6f}, threshold = {:.6f}, expected token: '{}' ({}), full "
            "string: '{}', input size: {}.",
            max_iterations,
            loss,
            m_convergence_threshold,
            m_corpus.get_token_from_id(expected_output_token),
            expected_output_token,
            full_string,
            static_cast<size_t>(train_input.size())
        );
        m_stats.record_learning_failure();
    }


} // namespace rllm
#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

namespace rllm
{
    static constexpr float MOMENTUM_BETA = 0.9f;
    static constexpr float GRAD_CLIP = 1.0f;
    static constexpr float VEL_CLIP = 0.1f;
    static constexpr float WEIGHT_CLAMP = 2.0f;

    OutputLayer::OutputLayer(const Corpus& corpus)
        : m_corpus(corpus)
    {}

    void OutputLayer::set_random_weights()
    {
        const int D = static_cast<int>(EmbeddingDimension::MAX);
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (const auto v : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                W_lm_head[v, d] = get_random_value(-scale, scale);
        V_lm_head.fill(RLMM_ZERO);
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last)
    {
        m_inputs.fill(RLMM_ZERO);
        for (const auto v : enum_iterator<TokenID>())
        {
            float sum = 0.f;
            for (const auto d : enum_iterator<EmbeddingDimension>())
                sum += h_last[d] * W_lm_head[v, d];
            m_inputs[v] = sum;
        }
    }

    // Returns dL/dh_last[D] and updates W_lm_head.
    fixed_size_vector<rlmm_float, EmbeddingDimension> OutputLayer::backward_and_update(
        const fixed_size_vector<rlmm_float, TokenID>& delta,
        const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
        float learning_rate
    )
    {
        fixed_size_vector<rlmm_float, EmbeddingDimension> dh;

        for (const auto v : enum_iterator<TokenID>())
        {
            const float dv = delta[v];
            for (const auto d : enum_iterator<EmbeddingDimension>())
            {
                dh[d] += dv * W_lm_head[v, d];
                const float g = math::clamp(dv * h_last[d], -GRAD_CLIP, GRAD_CLIP);
                V_lm_head[v, d] = math::clamp(
                    MOMENTUM_BETA * V_lm_head[v, d] + learning_rate * g,
                    -VEL_CLIP,
                    VEL_CLIP
                );
                W_lm_head[v, d] = math::clamp(
                    W_lm_head[v, d] + V_lm_head[v, d],
                    -WEIGHT_CLAMP,
                    WEIGHT_CLAMP
                );
            }
        }
        return dh;
    }


    void OutputLayer::rms_normalize_inputs()
    {
        constexpr float eps = 1e-6f;
        const float n = static_cast<float>(TokenID::MAX);
        float sum_sq = 0.0f;
        for (const auto i : enum_iterator<TokenID>())
            sum_sq += m_inputs[i] * m_inputs[i];
        const float rms = std::sqrt(sum_sq / n + eps);
        for (const auto i : enum_iterator<TokenID>())
            m_inputs[i] /= rms;
    }

    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        // Label smoothing (   =0.1): instead of a one-hot target, each non-target
        // token gets a small positive gradient of    /V. This prevents the model
        // from driving non-target logits to - and collapsing to one token.
        static constexpr float LABEL_SMOOTHING = 0.1f;
        const float smooth = LABEL_SMOOTHING / static_cast<float>(static_cast<int>(TokenID::MAX));

        float max_val = m_inputs[TokenID::START];
        for (const auto i : enum_iterator<TokenID>())
            max_val = math::max(max_val, m_inputs[i]);

        float sum_exp = 0.0f;
        for (const auto i : enum_iterator<TokenID>())
        {
            score.values[i] = std::exp(m_inputs[i] - max_val);
            sum_exp += score.values[i];
        }

        // delta[i] = smooth - softmax[i]  (small positive floor for all non-targets)
        for (const auto i : enum_iterator<TokenID>())
            score.values[i] = smooth - score.values[i] / sum_exp;

        // delta[expected] += (1 - LABEL_SMOOTHING)
        score.values[expected_output_token] += (RLMM_ONE - LABEL_SMOOTHING);
    }

    void OutputLayer::compute_deltas(const Score& score, fixed_size_vector<rlmm_float, TokenID>& deltas) const
    {
        deltas = score.values;
    }

} // namespace rllm
#include <parallel.hpp>
#include <print>

#if defined(USE_OPENMP)

namespace parallel {
    void init_parallel() {
          // OMP initialises its thread pool automatically
        std::println("Using OpenMP with {} threads", get_max_threads());
    }
}

#elif defined(USE_FASTFORK)

namespace parallel {
    void init_parallel() {
        fastfork::init();
        std::println("Using FastFork with {} threads", get_max_threads());
    }
}

#else // no OpenMP/FastFork: sequential execution

namespace parallel {
    void init_parallel() {
        std::println("Using sequential execution");
    }
}
#endif #include <RLLM.hpp>

#include <algorithm>
#include <iostream>
#include <print>
#include <string>

namespace rllm
{
    // maximum number of tokens to generate in response to a prompt before stopping
    static constexpr size_t MAX_NUM_ANSWER_TOKENS = 10;

    // Threshold for considering a predicted token as valid (not just noise).
    // This is a tunable hyperparameter.
    static constexpr float VALID_PREDICTION_THRESHOLD = 0.5f / 100.0f;

    void process_command(const std::string& _command, RLLM::PromptOptions& options, NeuralNetwork& nn)
    {
        const auto command = _command.empty() ? "/help" : _command;

        struct command_entry
        {
            std::vector<std::string_view> name;
            std::string_view description;
            std::function<void()> action;
        };

        std::vector<command_entry> commands = {
            {{"/help", "/h", "/?"},
             "Show this help message",
             [&]() {
                 std::println("Available commands:");
                 for (const auto& cmd : commands)
                 {
                     std::println("  {}: {}", cmd.name[0], cmd.description);
                 }
             }},
            {{"/exit"},
             "Exit the prompt mode",
             [&]() {
                 std::println("Exiting prompt mode.");
                 std::exit(0);
             }},
            {{"/info"},
             "Show information about the loaded model",
             [&]() {
                 std::println("Model information:");
                 std::println("  Vocabulary size:      {}", static_cast<size_t>(TokenID::MAX));
                 std::println("  Max context window:   {} tokens", static_cast<size_t>(PositionIndex::MAX));
                 std::println("  Embedding dimensions: {}", static_cast<size_t>(EmbeddingDimension::MAX));
                 std::println("  Attention heads:      {}", static_cast<size_t>(HeadsIndex::MAX));
                 std::println("  Transformer layers:   {}", nn.get_transformer_block_count());
             }},
            {{"/toggle_prio"}, "Toggle highest priority only mode", [&]() {
                 options.highest_prio_only = !options.highest_prio_only;
                 std::println(
                     "Toggled highest priority only mode. Now highest_prio_only is {}.", options.highest_prio_only
                 );
             }},
            {{"/hidden"}, "Mean-pool last hidden state over sequence (contextual embedding): /hidden <string>", [&]() {
                 const auto space = command.find(' ');
                 if (space == std::string::npos)
                 {
                     std::println("Usage: /hidden <string>");
                     return;
                 }
                 const std::string arg = command.substr(space + 1);
                 const auto& corpus = nn.get_corpus();
                 const auto token_ids = corpus.get_token_ids(arg);
                 nn.propagate_forward(token_ids);
                 const auto mean_vec = nn.get_last_hidden_mean();
                 constexpr size_t D = static_cast<size_t>(EmbeddingDimension::MAX);
                 constexpr size_t COLS = 8;
                 std::println("Hidden state embedding for '{}' ({} tokens, mean-pooled):", arg, static_cast<size_t>(token_ids.size()));
                 for (size_t i = 0; i < D; i += COLS)
                 {
                     std::print("  [{:3d}]", i);
                     for (size_t j = i; j < std::min(i + COLS, D); ++j)
                         std::print("  {:+.4f}", static_cast<float>(mean_vec[static_cast<EmbeddingDimension>(j)]));
                     std::println("");
                 }
             }},
            {{"/embed"}, "Print the raw input embedding vector(s) for a given string: /embed <string>", [&]() {
                 const auto space = command.find(' ');
                 if (space == std::string::npos)
                 {
                     std::println("Usage: /embed <string>");
                     return;
                 }
                 const std::string arg = command.substr(space + 1);
                 const auto& corpus = nn.get_corpus();
                 const auto& input_layer = nn.get_input_layer();
                 const auto token_ids = corpus.get_token_ids(arg);
                 constexpr size_t D = static_cast<size_t>(EmbeddingDimension::MAX);
                 constexpr size_t COLS = 8;
                 for (const auto pos : enum_iterator<PositionIndex>(token_ids.size()))
                 {
                     const TokenID tok = token_ids[pos];
                     const auto token_str = corpus.get_token_from_id(tok);
                     std::println("Embedding for token '{}' (id {}):", token_str, static_cast<size_t>(tok));
                     const auto& emb = input_layer.get_embedding(tok);
                     for (size_t i = 0; i < D; i += COLS)
                     {
                         std::print("  [{:3d}]", i);
                         for (size_t j = i; j < std::min(i + COLS, D); ++j)
                             std::print("  {:+.4f}", static_cast<float>(emb[static_cast<EmbeddingDimension>(j)]));
                         std::println("");
                     }
                 }
             }}
        };

        for (const auto& cmd : commands)
        {
            // Match either the exact command or the command followed by a space (for argument-taking commands).
            const bool exact = std::find(cmd.name.begin(), cmd.name.end(), command) != cmd.name.end();
            const bool prefix = std::any_of(cmd.name.begin(), cmd.name.end(), [&](std::string_view name) {
                return command.starts_with(std::string(name) + " ");
            });
            if (exact || prefix)
            {
                cmd.action();
                return;
            }
        }

        std::println("Unknown command: '{}'", command);
    }


    RLLM::RLLM(const std::vector<std::string>& filters)
        : m_filters(filters)
    {
        // Constructor implementation
    }

    void RLLM::prompt_mode(const std::string& filename, const std::optional<std::string>& one_shot_prompt)
    {
        set_nn_log_file("prompt.log");
        Corpus corpus{m_filters};
        size_t _num_layers = 2; // overriden when loaded from file
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(_num_layers, corpus, stats);


        std::println("Loading '{}'...", filename);
        if (! nn->load(filename))
        {
            std::println("Failed to load model from file: '{}'", filename);
            return;
        }
        std::println("Loaded.");

        PromptOptions options;

        if (one_shot_prompt.has_value())
        {
            process_line(*one_shot_prompt, corpus, *nn, options);
        }
        else
        {
            std::string line;
            while (true)
            {
                std::println("Enter input (or '/exit' to quit): ");
                std::print("> ");
                if (!std::getline(std::cin, line))
                {
                    std::println("Exiting prompt mode.");
                    break;
                }
                process_line(line, corpus, *nn, options);
            }
        }
    }

    void RLLM::process_line(const std::string& line, Corpus& corpus, NeuralNetwork& nn, PromptOptions& options)
    {
        if (line.starts_with("/") || line.empty())
        {
            process_command(line, options, nn);
            return;
        }

        auto token_id_list = corpus.get_token_ids(line);
        const auto full_string_opt = corpus.get_line(token_id_list);
        if (!full_string_opt.has_value())
        {
            std::println("Input contains unknown tokens. Please try again.");
            return;
        }
        std::println("Input tokens: {}", *full_string_opt);

        const auto question_size = token_id_list.size();

        for (size_t iter = 0; iter < MAX_NUM_ANSWER_TOKENS; ++iter)
        {
            nn.propagate_forward(token_id_list);

            const auto output_token_id_lists = nn.get_best_output_token_ids(5);
            if (output_token_id_lists.empty())
            {
                std::println("No output tokens predicted.");
                break;
            }

            int ix = 0;
            int num_valid_tokens = 0;
            float top_k_probability_sum = 0.0f;
            for (const auto& entry : output_token_id_lists)
                top_k_probability_sum += entry.activation;

            for (const auto& entry : output_token_id_lists)
            {
                auto tok = nn.get_corpus().get_token_from_id(entry.token_id);
                if (tok == "\n")
                    tok = "\\n";
                const float global_percent = entry.activation * 100.0f;
                const float top_k_percent = (top_k_probability_sum > 0.0f)
                    ? (entry.activation / top_k_probability_sum) * 100.0f
                    : 0.0f;

                if (global_percent >= 0.01f)
                {
                    std::println(
                        "\t     prediction[{}]: '{}' (id: '{}'), {:.6f}% (top-5 {:.2f}%)",
                        ix,
                        tok,
                        static_cast<int>(entry.token_id),
                        global_percent,
                        top_k_percent
                    );
                }
                else
                {
                    std::println(
                        "\t     prediction[{}]: '{}' (id: '{}'), {:.3e}% (top-5 {:.2f}%)",
                        ix,
                        tok,
                        static_cast<int>(entry.token_id),
                        global_percent,
                        top_k_percent
                    );
                }
                ix++;
                if (entry.activation > VALID_PREDICTION_THRESHOLD)
                    num_valid_tokens++;
            }

            if (num_valid_tokens == 0)
            {
                std::println("No output tokens predicted.");
                break;
            }

            size_t random_index = 0;
            if (!options.highest_prio_only)
                random_index = static_cast<size_t>(rand()) % num_valid_tokens;

            const auto& entry = output_token_id_lists[random_index];
            auto output_token = nn.get_corpus().get_token_from_id(entry.token_id);
            if (output_token == "<UNK>")
            {
                std::println("Predicted next token is unknown. Stopping generation.");
                break;
            }
            if (output_token == "\n")  output_token = "\\n";
            if (output_token == "\t")  output_token = "\\t";
            std::println("Predicted next token: {}", output_token);
            token_id_list.push_back(entry.token_id);
        }

        auto reply_id_list = token_id_list.sub_array(question_size);
        const auto full_answer_string_opt = corpus.get_line(token_id_list);
        if (!full_answer_string_opt.has_value())
        {
            std::println("Input contains unknown tokens. Please try again.");
            return;
        }
        std::println("Full answer string: {}", *full_answer_string_opt);
    }

    void RLLM::train_mode(
        const std::optional<std::string>& input_filename,
        const std::string& output_filename,
        size_t num_layers,
        bool verbose,
        TrainingMethod method,
        std::optional<std::chrono::seconds> checkpointing_interval,
        int window_size,
        size_t num_epochs,
        const std::string& train_corpus_dir
    )
    {
        std::println("Training mode");
        set_nn_log_file("train.log");

        Corpus corpus{m_filters};
        corpus.load_files_from_dir(train_corpus_dir);
        Statistics stats;

        auto nn = std::make_unique<NeuralNetwork>(num_layers, corpus, stats);
        nn->set_training_method(method);
        nn->set_window_size(window_size);

        nn->train(verbose, num_epochs, input_filename, checkpointing_interval);

        stats.print_statistics();

        nn->save(output_filename);
    }
} // namespace rllm#include <RLLM.hpp>
#include <JsonTensorHelpers.hpp>

#include <fstream>
#include <format>
#include <chrono>
#include <nlohmann/json.hpp>
#include <print>
#include <stdexcept>
#include <string>


namespace rllm
{
    namespace
    {
        // Stable checksum of tokenizer contents to detect model/tokenizer mismatch.
        uint64_t tokenizer_signature()
        {
            uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
            for (const auto& [id, info] : tokenizer_map)
            {
                const auto id_val = static_cast<uint64_t>(static_cast<int32_t>(id));
                h ^= id_val;
                h *= 1099511628211ULL;

                for (const char* p = info.str; *p != '\0'; ++p)
                {
                    h ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
                    h *= 1099511628211ULL;
                }

                h ^= static_cast<uint64_t>(info.end_of_word ? 1 : 0);
                h *= 1099511628211ULL;
            }
            return h;
        }
    } // namespace


    void InputLayer::load(const nlohmann::json& j)
    {
        if (!j.contains("embeddings")) return; // backwards compat: skip if absent

        const auto& emb_j = j.at("embeddings");
        for (size_t t = 0; t < static_cast<size_t>(TokenID::MAX); ++t)
        {
            const auto& row = emb_j.at(t);
            for (size_t d = 0; d < static_cast<size_t>(EmbeddingDimension::MAX); ++d)
                m_embeddings[static_cast<TokenID>(t)][static_cast<EmbeddingDimension>(d)] = row.at(d).template get<float>();
        }
    }

    nlohmann::json InputLayer::save() const
    {
        auto emb_j = nlohmann::json::array();
        for (size_t t = 0; t < static_cast<size_t>(TokenID::MAX); ++t)
        {
            auto row = nlohmann::json::array();
            for (size_t d = 0; d < static_cast<size_t>(EmbeddingDimension::MAX); ++d)
                row.push_back(m_embeddings[static_cast<TokenID>(t)][static_cast<EmbeddingDimension>(d)]);
            emb_j.push_back(std::move(row));
        }
        return {{"embeddings", std::move(emb_j)}};
    }

    void OutputLayer::load(const nlohmann::json& j)
    {
        m_inputs.fill(RLMM_ZERO);
        if (j.contains("W_lm_head"))
        {
            const auto& w_j = j.at("W_lm_head");
            size_t i = 0;
            for (const auto t : enum_iterator<TokenID>())
                for (const auto d : enum_iterator<EmbeddingDimension>())
                    W_lm_head[t, d] = w_j.at(i++).template get<float>();
        }
        V_lm_head.fill(RLMM_ZERO);
    }

    nlohmann::json OutputLayer::save() const
    {
        auto w_j = nlohmann::json::array();
        w_j.get_ref<nlohmann::json::array_t&>().reserve(W_lm_head.ROWS * W_lm_head.COLS);
        for (const auto t : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                w_j.push_back(W_lm_head[t, d]);
        return {{"W_lm_head", std::move(w_j)}};
    }


    bool NeuralNetwork::load(const std::string& filename)
    {
        try
        {
            std::ifstream file{filename};
            if (!file) {
                return false;
            }

            const auto j = nlohmann::json::parse(file);

            const auto version = j.value("version", 0);
            if (version != 1)
            {
                throw std::runtime_error("Unsupported model version");
            }

            const auto expected_vocab_size = static_cast<size_t>(TokenID::MAX);
            if (j.contains("tokenizer_vocab_size"))
            {
                const auto model_vocab_size = j.at("tokenizer_vocab_size").template get<size_t>();
                if (model_vocab_size != expected_vocab_size)
                {
                    throw std::runtime_error(
                        std::format(
                            "Tokenizer vocab size mismatch (model={}, runtime={}). "
                            "Rebuild/regenerate tokenizer map and retrain or load a compatible model.",
                            model_vocab_size,
                            expected_vocab_size
                        )
                    );
                }
            }

            if (j.contains("tokenizer_signature"))
            {
                const auto model_sig = j.at("tokenizer_signature").template get<uint64_t>();
                const auto runtime_sig = tokenizer_signature();
                if (model_sig != runtime_sig)
                {
                    throw std::runtime_error(
                        "Tokenizer signature mismatch between model and runtime tokenizer map. "
                        "Rebuild/regenerate tokenizer map and retrain or load a compatible model."
                    );
                }
            }

            m_input_layer.load(j.at("input_layer"));

            if (j.contains("transformer_blocks"))
            {
                const auto& blocks_j = j.at("transformer_blocks");
                m_transformer_blocks.clear();
                m_transformer_blocks.reserve(blocks_j.size());
                for (const auto& b : blocks_j)
                {
                    m_transformer_blocks.emplace_back();
                    m_transformer_blocks.back().load(b);
                }
            }

            m_output_layer.load(j.at("output_layer"));
        }
        catch (const std::exception& e)
        {
            std::println("Failed to load model '{}': {}", filename, e.what());
            std::abort();
        }
        return true;
    }

    void NeuralNetwork::save(const std::string& filename) const
    {
        const auto save_start = std::chrono::steady_clock::now();

        nlohmann::json j;
        j["version"] = 1;
        j["tokenizer_vocab_size"] = static_cast<size_t>(TokenID::MAX);
        j["tokenizer_signature"] = tokenizer_signature();

        j["input_layer"] = m_input_layer.save();

        auto blocks = nlohmann::json::array();
        for (const auto& block : m_transformer_blocks)
            blocks.push_back(block.save());
        j["transformer_blocks"] = std::move(blocks);

        j["output_layer"] = m_output_layer.save();

        std::ofstream file{filename};
        file << j.dump(2) << '\n';

        const auto save_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - save_start).count();
        std::println("Saved model '{}' in {:.3f} seconds", filename, save_elapsed);
    }
} // namespace rllm
#include <JsonTensorHelpers.hpp>
#include <RandomHelpers.hpp>
#include <TransformerBlock.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>

namespace rllm
{
    static_assert(
        static_cast<int>(EmbeddingDimension::MAX) % static_cast<int>(HeadsIndex::MAX) == 0,
        "EmbeddingDimension::MAX must be divisible by HeadsIndex::MAX"
    );

    //  randomize

    void TransformerBlock::randomize()
    {
        const float sd = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(EmbeddingDimension::MAX)));
        const float sf = 1.0f / std::sqrt(static_cast<float>(static_cast<int>(FFDimension::MAX)));

        W_q.fill_rand(-sd, sd);
        W_k.fill_rand(-sd, sd);
        W_v.fill_rand(-sd, sd);
        W_o.fill_rand(-sd, sd);
        W_gate.fill_rand(-sd, sd);
        W_up.fill_rand(-sd, sd);
        W_down.fill_rand(-sf, sf);

        V_q.fill(RLMM_ZERO);
        V_k.fill(RLMM_ZERO);
        V_v.fill(RLMM_ZERO);
        V_o.fill(RLMM_ZERO);
        V_gate.fill(RLMM_ZERO);
        V_up.fill(RLMM_ZERO);
        V_down.fill(RLMM_ZERO);
    }

    //  normalisation

    // y_t = x_t / rms(x_t)  for each row t
    void TransformerBlock::rms_norm(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& y
    )
    {
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        PARFOR(t, enum_iterator<PositionIndex>(x.num_rows()))
            float sq = 0.f;
            for (const auto i : enum_iterator<EmbeddingDimension>())
            {
                const auto val = x[t, i];
                sq += val * val;
            }
            const float inv = 1.0f / std::sqrt(sq / fd + eps);
            for (const auto i : enum_iterator<EmbeddingDimension>())
                y[t, i] = x[t, i] * inv;
        ENDFOR
    }

    // dx += dL/dx  given dy = dL/dy and the original x (not the normalised y).
    // Per row:  dx_j += (1/rms) * (dy_j  -  y_j * mean(dy  y))
    void TransformerBlock::rms_norm_backward(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dy,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dx
    )
    {
        constexpr float eps = 1e-6f;
        constexpr float fd = static_cast<float>(EmbeddingDimension::MAX);

        PARFOR(t, enum_iterator<PositionIndex>(x.num_rows()))
            float sq = 0.f;
            for (const auto i : enum_iterator<EmbeddingDimension>())
                sq += x[t, i] * x[t, i];
            const float inv = 1.0f / std::sqrt(sq / fd + eps);

            // dot = mean(dy  y) = (1/d) * sum_i dy[i] * x[i] * inv
            float dot = 0.f;
            for (const auto i : enum_iterator<EmbeddingDimension>())
                dot += dy[t, i] * x[t, i] * inv;
            dot /= fd;

            for (const auto i : enum_iterator<EmbeddingDimension>())
                dx[t, i] += inv * (dy[t, i] - x[t, i] * inv * dot);
        ENDFOR
    }

    //  attention helpers

    /** In-place causal softmax over a [T  T] score matrix (row i only attends
    * to positions j  i).  stride is the distance between rows in floats.
    *
    * @param x The [T  T] matrix of scores to softmax in-place.  Only the top-left [T  T] block is accessed and modified; the rest of the matrix is left as-is to avoid unnecessary writes.
    * @param T The sequence length (the active size of the [T  T] block).
    */
    void
    TransformerBlock::causal_softmax(flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& x, PositionIndex T)
    {
        PARFOR(i, enum_iterator<PositionIndex>(T))
            float max_val = x[i, PositionIndex::START];
            for (const auto j : enum_iterator<PositionIndex>(inc(PositionIndex::START), inc(i)))
                max_val = math::max(max_val, x[i, j]);
            float sum_exp = 0.f;
            // Compute and sum the exponentials for the active [T  T] block of the row; leave the rest of the row as-is
            // to avoid unnecessary writes.
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
            {
                x[i, j] =  static_cast<rlmm_float>(std::exp(x[i, j] - max_val));
                sum_exp += x[i, j];
            }
            const float inv = 1.0f / sum_exp;
            // scale the active [T  T] block of the row; leave the rest of the row as-is to avoid unnecessary writes
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                x[i, j] *=  static_cast<rlmm_float>(inv);

            // clear from the active [T  T] block to the end of the row
            // to avoid stale values affecting the backward pass
            for (const auto j : enum_iterator<PositionIndex>(inc(i), T))
                x[i, j] = static_cast<rlmm_float>(0.f);
        ENDFOR
    }

    /** dscores[T   T] +=    L/   scores  via the softmax Jacobian.
     * dp/dscores use stride T; p uses p_stride (the cached matrix's row stride).
     */
    void TransformerBlock::softmax_backward(
        const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dp,
        const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& p,
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dscores,
        PositionIndex T
    )
    {
        PARFOR(i, enum_iterator<PositionIndex>(T))
            float dot = 0.f;
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                dot += dp[i, j] * p[i, j];
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                dscores[i, j] += p[i, j] * (dp[i, j] - dot);
        ENDFOR
    }

    //  SwiGLU backward

    void TransformerBlock::swiglu_backward(
        PositionIndex seq,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_ffn_act,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_gate_pre,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_up_pre
    )
    {
        PARFOR_2D(t, f, enum_iterator2D<PositionIndex, FFDimension>(seq))
            const float g = gate_pre[t, f];
            const float sg = 1.0f / (1.0f + std::exp(-g)); // sigma(g)
            const float silu = g * sg;
            // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
            // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
            const float dsilu_dg = sg * (1.0f + g * (1.0f - sg));
            d_gate_pre[t, f] = d_ffn_act[t, f] * up_pre[t, f] * dsilu_dg;
            d_up_pre[t, f] = d_ffn_act[t, f] * silu;
        ENDFOR
    }

    //  SGD + momentum

    //  forward workspace
    // All large fixed-size matrices live here so they are heap-allocated via
    // unique_ptr and do not blow the stack (~21 MB of combined fixed-size arrays).
    struct ForwardWorkspace
    {
        PositionIndex seq_len;
        // Activations cached for the backward pass
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_in;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_norm_attn;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> Q, K, V;
        // Per-head softmax weight matrices; only the top-left [T  T] block is live.
        fixed_size_vector<flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>, HeadsIndex> attn_w;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> attn_concat;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_mid;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> h_norm_ff;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> gate_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> up_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> ffn_act;
        // Temporaries used only within forward() itself
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> attn_proj;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> ffn_out;

        explicit ForwardWorkspace(PositionIndex seq)
            : seq_len(seq)
            , h_in(seq)
            , h_norm_attn(seq)
            , Q(seq)
            , K(seq)
            , V(seq)
            , attn_concat(seq)
            , h_mid(seq)
            , h_norm_ff(seq)
            , gate_pre(seq)
            , up_pre(seq)
            , ffn_act(seq)
            , attn_proj(seq)
            , ffn_out(seq)
        {}
    };

    //  destructor
    // Defined here so ForwardWorkspace is complete when unique_ptr deleter fires.
    TransformerBlock::TransformerBlock() = default;
    TransformerBlock::~TransformerBlock() = default;
    TransformerBlock::TransformerBlock(TransformerBlock&&) noexcept = default;
    TransformerBlock& TransformerBlock::operator=(TransformerBlock&&) noexcept = default;

    //  forward pass

    void
    TransformerBlock::forward(flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len)
    {
        constexpr int Dh = static_cast<int>(HeadDimension::MAX);

        if (!m_fwd_ws || m_fwd_ws->seq_len != seq_len)
            m_fwd_ws = std::make_unique<ForwardWorkspace>(seq_len);
        auto& ws = *m_fwd_ws;

        ws.h_in = h;

        //  1. Pre-norm (attention)
        rms_norm(h, ws.h_norm_attn);

        //  2. Q / K / V projections
        PARSECTIONS_BEGIN
            matmul_ABt(ws.h_norm_attn, W_q, ws.Q);
        PARSECTION
            matmul_ABt(ws.h_norm_attn, W_k, ws.K);
        PARSECTION
            matmul_ABt(ws.h_norm_attn, W_v, ws.V);
        PARSECTIONS_END

        //  3. Multi-head causal self-attention
        const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        ws.attn_concat.fill(RLMM_ZERO);

        PARFOR(hi, enum_iterator<HeadsIndex>())
            auto& scores_mat = ws.attn_w[hi];
            scores_mat.set_size(seq_len, seq_len);

            const auto hi_int = static_cast<size_t>(hi);
            const auto hStart = static_cast<EmbeddingDimension>(hi_int * static_cast<size_t>(HeadDimension::MAX));
            const auto hEnd = static_cast<EmbeddingDimension>((hi_int + 1) * static_cast<size_t>(HeadDimension::MAX));

            // scores_mat[i, j] = Q[i, hi*Dh..]  K[j, hi*Dh..] * scale
            for (const auto i : enum_iterator<PositionIndex>(seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(seq_len))
                {
                    float dot = 0.f;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        dot += ws.Q[i, d] * ws.K[j, d];
                    scores_mat[i, j] = dot * scale;
                }
            }

            causal_softmax(scores_mat, seq_len);

            // attn_concat[i, d] = sum_j scores_mat[i,j] * V[j, d]  (causal: j  i)
            for (const auto i : enum_iterator<PositionIndex>(seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                {
                    const float w = scores_mat[i, j];
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws.attn_concat[i, d] += w * ws.V[j, d];
                }
            }
        ENDFOR

        //  4. Output projection + residual
        matmul_ABt(ws.attn_concat, W_o, ws.attn_proj);

        PARFOR_2D(t, d, enum_iterator2D<PositionIndex, EmbeddingDimension>(seq_len))
            ws.h_mid[t, d] = h[t, d] + ws.attn_proj[t, d];
        ENDFOR

        //  5. Pre-norm (FFN)
        rms_norm(ws.h_mid, ws.h_norm_ff);

        //  6. SwiGLU FFN
        PARSECTIONS_BEGIN
            matmul_ABt(ws.h_norm_ff, W_gate, ws.gate_pre);
        PARSECTION
            matmul_ABt(ws.h_norm_ff, W_up, ws.up_pre);
        PARSECTIONS_END

        PARFOR_2D(t, f, enum_iterator2D<PositionIndex, FFDimension>(seq_len))
            const float g = ws.gate_pre[t, f];
            const float silu = g / (1.0f + std::exp(-g));
            ws.ffn_act[t, f] = silu * ws.up_pre[t, f];
        ENDFOR

        matmul_ABt(ws.ffn_act, W_down, ws.ffn_out);

        //  7. Residual
        PARFOR_2D(t, d, enum_iterator2D<PositionIndex, EmbeddingDimension>(seq_len))
            h[t, d] = ws.h_mid[t, d] + ws.ffn_out[t, d];
        ENDFOR
    }

    //  backward temporaries
    // All large matrices live here so they are heap-allocated via unique_ptr and
    // do not blow the stack (~21 MB combined).
    struct BackwardWorkspace
    {
        // FFN backward
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_mid;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_ffn_act;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension> dW_down;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_gate_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_up_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_norm_ff;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> dW_gate;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> dW_up;
        // Attention backward
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_attn_concat;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_o;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_Q;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_K;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_V;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_q;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_k;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_v;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_norm_attn;
        // Scratch buffer shared by both matmul accumulation loops
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> tmp;
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex> d_scores;
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex> d_raw;

        explicit BackwardWorkspace(PositionIndex seq)
            : d_h_mid(seq)
            , d_ffn_act(seq)
            , d_gate_pre(seq)
            , d_up_pre(seq)
            , d_h_norm_ff(seq)
            , d_attn_concat(seq)
            , d_Q(seq)
            , d_K(seq)
            , d_V(seq)
            , d_h_norm_attn(seq)
            , tmp(seq)
            , d_scores(seq, seq)
            , d_raw(seq, seq)
        {}

        // Zero only the fields that accumulate across the backward pass.
        // Fields fully overwritten before use (d_h_mid, d_ffn_act, d_gate_pre,
        // d_up_pre, d_attn_concat, tmp, d_scores, d_raw) are left untouched.
        void reset()
        {
            dW_down.fill(RLMM_ZERO);
            d_h_norm_ff.fill(RLMM_ZERO);
            dW_gate.fill(RLMM_ZERO);
            dW_up.fill(RLMM_ZERO);
            dW_o.fill(RLMM_ZERO);
            d_Q.fill(RLMM_ZERO);
            d_K.fill(RLMM_ZERO);
            d_V.fill(RLMM_ZERO);
            dW_q.fill(RLMM_ZERO);
            dW_k.fill(RLMM_ZERO);
            dW_v.fill(RLMM_ZERO);
            d_h_norm_attn.fill(RLMM_ZERO);
        }
    };

    //  backward pass

    void TransformerBlock::backward(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dout,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& din,
        float learning_rate
    )
    {
        const PositionIndex seq = m_fwd_ws->seq_len;
        if (!m_bwd_ws || m_bwd_ws->d_h_mid.num_rows() != seq)
            m_bwd_ws = std::make_unique<BackwardWorkspace>(seq);
        m_bwd_ws->reset();
        auto* ws = m_bwd_ws.get();
        auto& fwd = *m_fwd_ws;

        //  FFN backward
        // h_out = h_mid + ffn_out   d_h_mid += dout,  d_ffn_out = dout (same buffer)
        ws->d_h_mid = dout; // copy first to avoid overwriting dout before it's used in dW_down

        // d_ffn_act = d_ffn_out @ W_down   (W_down[D  Dff])
        matmul_AB(dout, W_down, ws->d_ffn_act);

        // dW_down[D, Dff] += dout^T @ ffn_act
        matmul_AtB_acc(dout, fwd.ffn_act, ws->dW_down);

        // SwiGLU backward: d(silu(g)*u) / dg, du
        swiglu_backward(fwd.seq_len, fwd.gate_pre, fwd.up_pre, ws->d_ffn_act, ws->d_gate_pre, ws->d_up_pre);

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        matmul_AB(ws->d_gate_pre, W_gate, ws->tmp);
        ws->d_h_norm_ff.element_wise_add(ws->tmp);
        matmul_AB(ws->d_up_pre, W_up, ws->tmp);
        ws->d_h_norm_ff.element_wise_add(ws->tmp);

        // weight gradients for gate, up
        PARSECTIONS_BEGIN
            matmul_AtB_acc(ws->d_gate_pre, fwd.h_norm_ff, ws->dW_gate);
        PARSECTION
            matmul_AtB_acc(ws->d_up_pre, fwd.h_norm_ff, ws->dW_up);
        PARSECTIONS_END

        // RMSNorm backward for FFN: d_h_mid += rms_bwd(d_h_norm_ff, h_mid)
        rms_norm_backward(ws->d_h_norm_ff, fwd.h_mid, ws->d_h_mid);

        //  Attention backward
        // h_mid = h_in + attn_proj   d_attn_proj = d_h_mid (passed through residual)
        //                             d_h_in (residual part) = d_h_mid (accumulated below)

        // d_attn_concat = d_attn_proj @ W_o
        matmul_AB(ws->d_h_mid, W_o, ws->d_attn_concat);
        matmul_AtB_acc(ws->d_h_mid, fwd.attn_concat, ws->dW_o);

        // Per-head backward
        const float scale = 1.0f / std::sqrt(static_cast<float>(static_cast<size_t>(HeadDimension::MAX)));
        for (const auto hi : enum_iterator<HeadsIndex>())
        {
            ws->d_raw.fill(RLMM_ZERO); // d_raw accumulates per head; reset each iteration

            const auto& scores_mat = fwd.attn_w[hi];
            const auto hi_int = static_cast<size_t>(hi);
            const auto hStart = static_cast<EmbeddingDimension>(hi_int * static_cast<size_t>(HeadDimension::MAX));
            const auto hEnd = static_cast<EmbeddingDimension>((hi_int + 1) * static_cast<size_t>(HeadDimension::MAX));

            // d_V_h[j, d] += sum_i scores_mat[i,j] * d_attn_concat[i, d]
            for (const auto j : enum_iterator<PositionIndex>(fwd.seq_len))
            {
                for (const auto i : enum_iterator<PositionIndex>(j, fwd.seq_len)) // only non-masked rows
                {
                    const float w = scores_mat[i, j];
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws->d_V[j, d] += w * ws->d_attn_concat[i, d];
                }
            }

            // d_scores[i,j] = d_attn_concat[i, d]  V[j, d]
            for (const auto i : enum_iterator<PositionIndex>(fwd.seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                {
                    float dot = 0.f;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        dot += ws->d_attn_concat[i, d] * fwd.V[j, d];
                    ws->d_scores[i, j] = dot;
                }
            }

            // Backward through causal softmax
            softmax_backward(ws->d_scores, scores_mat, ws->d_raw, fwd.seq_len);

            // d_Q and d_K
            for (const auto i : enum_iterator<PositionIndex>(fwd.seq_len))
            {
                for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                {
                    const float ds = ws->d_raw[i, j] * scale;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws->d_Q[i, d] += ds * fwd.K[j, d];
                }
            }
            for (const auto j : enum_iterator<PositionIndex>(fwd.seq_len))
            {
                for (const auto i : enum_iterator<PositionIndex>(j, fwd.seq_len))
                {
                    const float ds = ws->d_raw[i, j] * scale;
                    for (const auto d : enum_iterator<EmbeddingDimension>(hStart, hEnd))
                        ws->d_K[j, d] += ds * fwd.Q[i, d];
                }
            }
        }

        // Weight gradients for W_q, W_k, W_v
        PARSECTIONS_BEGIN
            matmul_AtB_acc(ws->d_Q, fwd.h_norm_attn, ws->dW_q);
        PARSECTION
            matmul_AtB_acc(ws->d_K, fwd.h_norm_attn, ws->dW_k);
        PARSECTION
            matmul_AtB_acc(ws->d_V, fwd.h_norm_attn, ws->dW_v);
        PARSECTIONS_END

        // d_h_norm_attn = d_Q @ W_q  +  d_K @ W_k  +  d_V @ W_v
        matmul_AB(ws->d_Q, W_q, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);
        matmul_AB(ws->d_K, W_k, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);
        matmul_AB(ws->d_V, W_v, ws->tmp);
        ws->d_h_norm_attn.element_wise_add(ws->tmp);

        // d_h_in: residual from d_h_mid + RMSNorm backward
        din = ws->d_h_mid;
        rms_norm_backward(ws->d_h_norm_attn, fwd.h_in, din);

        //  Weight updates
        sgd_update(W_q, V_q, ws->dW_q, learning_rate);
        sgd_update(W_k, V_k, ws->dW_k, learning_rate);
        sgd_update(W_v, V_v, ws->dW_v, learning_rate);
        sgd_update(W_o, V_o, ws->dW_o, learning_rate);
        sgd_update(W_gate, V_gate, ws->dW_gate, learning_rate);
        sgd_update(W_up, V_up, ws->dW_up, learning_rate);
        sgd_update(W_down, V_down, ws->dW_down, learning_rate);
    }

    //  serialisation

    nlohmann::json TransformerBlock::save() const
    {
        using namespace json_helpers;
        return {
            {"W_q", serialize_matrix(W_q)},
            {"W_k", serialize_matrix(W_k)},
            {"W_v", serialize_matrix(W_v)},
            {"W_o", serialize_matrix(W_o)},
            {"W_gate", serialize_matrix(W_gate)},
            {"W_up", serialize_matrix(W_up)},
            {"W_down", serialize_matrix(W_down)},
        };
    }

    void TransformerBlock::load(const nlohmann::json& j)
    {
        using namespace json_helpers;
        deserialize_matrix(j.at("W_q"), W_q);
        deserialize_matrix(j.at("W_k"), W_k);
        deserialize_matrix(j.at("W_v"), W_v);
        deserialize_matrix(j.at("W_o"), W_o);
        deserialize_matrix(j.at("W_gate"), W_gate);
        deserialize_matrix(j.at("W_up"), W_up);
        deserialize_matrix(j.at("W_down"), W_down);

        // Reset momentum on load  do not persist transient training state
        V_q.fill(RLMM_ZERO);
        V_k.fill(RLMM_ZERO);
        V_v.fill(RLMM_ZERO);
        V_o.fill(RLMM_ZERO);
        V_gate.fill(RLMM_ZERO);
        V_up.fill(RLMM_ZERO);
        V_down.fill(RLMM_ZERO);
    }

} // namespace rllm
#pragma once

#include <array>
#include <atomic>
#include <optional>

namespace fastfork
{
    // Dmitry Vyukov's MPMC bounded queue (lock-free).
    // N must be a power of two.
    // try_push returns false when full  (t is left unchanged).
    // try_pop  returns false when empty.
    template<typename T, int N>
    class CircularQueue
    {
        static_assert((N & (N - 1)) == 0, "N must be a power of two");
        static constexpr unsigned MASK = static_cast<unsigned>(N) - 1u;

        struct Slot
        {
            std::atomic<unsigned> sequence{0};
            T item{};
        };

        std::array<Slot, N>               m_slots{};
        alignas(64) std::atomic<unsigned> m_enqueue_pos{0};
        alignas(64) std::atomic<unsigned> m_dequeue_pos{0};

    public:
        CircularQueue() noexcept
        {
            for (unsigned i = 0; i < static_cast<unsigned>(N); ++i)
                m_slots[i].sequence.store(i, std::memory_order_relaxed);
        }

        // Returns false if full; t is NOT moved from in that case.
        bool try_push(T&& t)
        {
            unsigned pos = m_enqueue_pos.load(std::memory_order_relaxed);
            for (;;)
            {
                Slot& slot = m_slots[pos & MASK];
                const unsigned seq  = slot.sequence.load(std::memory_order_acquire);
                const auto     diff = static_cast<int>(seq - pos);
                if (diff == 0)
                {
                    if (m_enqueue_pos.compare_exchange_weak(
                            pos, pos + 1, std::memory_order_relaxed))
                        break;
                    // pos refreshed by CAS on failure
                }
                else if (diff < 0)
                    return false; // full
                else
                    pos = m_enqueue_pos.load(std::memory_order_relaxed);
            }
            m_slots[pos & MASK].item = std::move(t);
            m_slots[pos & MASK].sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        // Returns false if empty.
        bool try_pop(T& t)
        {
            unsigned pos = m_dequeue_pos.load(std::memory_order_relaxed);
            for (;;)
            {
                Slot& slot = m_slots[pos & MASK];
                const unsigned seq  = slot.sequence.load(std::memory_order_acquire);
                const auto     diff = static_cast<int>(seq - (pos + 1));
                if (diff == 0)
                {
                    if (m_dequeue_pos.compare_exchange_weak(
                            pos, pos + 1, std::memory_order_relaxed))
                        break;
                }
                else if (diff < 0)
                    return false; // empty
                else
                    pos = m_dequeue_pos.load(std::memory_order_relaxed);
            }
            t = std::move(m_slots[pos & MASK].item);
            m_slots[pos & MASK].sequence.store(
                pos + static_cast<unsigned>(N), std::memory_order_release);
            return true;
        }

        // Approximate occupancy (racy; used only as a batch-steal hint).
        int approx_size() const noexcept
        {
            const unsigned enq = m_enqueue_pos.load(std::memory_order_relaxed);
            const unsigned deq = m_dequeue_pos.load(std::memory_order_relaxed);
            return static_cast<int>(enq - deq);
        }
    };

} // namespace fastfork


namespace fastfork
{
    // Chase-Lev work-stealing deque (fixed-size, single-producer / multi-consumer).
    //
    // Interface
    //   push_bottom(T&&)  bool   owner-only LIFO push; false if full (t unchanged)
    //   pop_bottom(T&)    bool   owner-only LIFO pop;  false if empty
    //   steal_top(T&)     bool   any-thread FIFO steal; false if empty / lost CAS
    //
    // Correctness: the (t == b) "last element" race between pop_bottom and
    // steal_top is resolved by a seq_cst CAS on m_top; the loser returns false
    // without touching the item.  Items live in std::optional<T> so that moves
    // happen only after ownership has been established.
    template<typename T, unsigned N>
    class alignas(64) WSDeque
    {
        static_assert((N & (N - 1)) == 0, "N must be a power of two");
        static constexpr int64_t MASK = static_cast<int64_t>(N) - 1;

        struct Slot { std::optional<T> val; };

        Slot                             m_slots[N];
        alignas(64) std::atomic<int64_t> m_bottom{0};
        alignas(64) std::atomic<int64_t> m_top{0};

    public:
        WSDeque() = default;
        WSDeque(const WSDeque&)            = delete;
        WSDeque& operator=(const WSDeque&) = delete;

        // Owner only  push to bottom (LIFO end).  Returns false if full; t unchanged.
        bool push_bottom(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>)
        {
            const int64_t b  = m_bottom.load(std::memory_order_relaxed);
            const int64_t tp = m_top.load(std::memory_order_acquire);
            if (b - tp >= static_cast<int64_t>(N)) return false;
            m_slots[b & MASK].val = std::move(t);
            std::atomic_thread_fence(std::memory_order_release);
            m_bottom.store(b + 1, std::memory_order_relaxed);
            return true;
        }

        // Owner only  pop from bottom (LIFO).  Returns false if empty.
        bool pop_bottom(T& item) noexcept(std::is_nothrow_move_assignable_v<T>)
        {
            const int64_t b = m_bottom.load(std::memory_order_relaxed) - 1;
            m_bottom.store(b, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            int64_t t = m_top.load(std::memory_order_relaxed);

            if (t > b)                          // deque is empty
            {
                m_bottom.store(b + 1, std::memory_order_relaxed);
                return false;
            }
            if (t == b)                         // last element  may race with steal_top
            {
                m_bottom.store(b + 1, std::memory_order_relaxed);
                if (!m_top.compare_exchange_strong(t, t + 1,
                        std::memory_order_seq_cst, std::memory_order_relaxed))
                    return false;               // stealer won
            }
            item = std::move(*m_slots[b & MASK].val);
            m_slots[b & MASK].val.reset();
            return true;
        }

        // Any thread  steal from top (FIFO).  Returns false if empty or CAS lost.
        bool steal_top(T& item) noexcept(std::is_nothrow_move_assignable_v<T>)
        {
            int64_t t = m_top.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const int64_t b = m_bottom.load(std::memory_order_acquire);
            if (t >= b) return false;
            if (!m_top.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed))
                return false;
            item = std::move(*m_slots[t & MASK].val);
            m_slots[t & MASK].val.reset();
            return true;
        }

        // Approximate occupancy  racy, used only as a batch-steal hint.
        int64_t approx_size() const noexcept
        {
            const int64_t b = m_bottom.load(std::memory_order_relaxed);
            const int64_t t = m_top.load(std::memory_order_relaxed);
            return std::max(int64_t{0}, b - t);
        }
    };

} // namespace fastfork
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace fastfork
{
    using task_t = std::function<void()>;

    void init();

    // Returns the maximum number of threads available for parallel execution.
    int get_max_threads();

    // how many threads to use for parallel regions; default is number of hardware threads
    void set_num_threads(int n);

    // returns thread i of N
    int get_thread_num();


    // Fork a task to run in parallel. The task will be executed by a worker thread.
    // it is allowed for a task to fork additional tasks, which will also be executed in parallel.
    // We are NUMA aware, so tasks may be scheduled on different NUMA nodes, and we will attempt
    // to schedule tasks on the same NUMA node as the data they access.
    void fork_task(task_t t);

    // Scoped batch handle.
    //   fork_task(ctx, t)   increments ctx, wraps t to decrement on completion.
    //   ctx destructor      calls wait_local(ctx) so the batch always drains
    //                        before ctx goes out of scope.
    // Outer in-flight tasks are not counted, so nested fork/wait is deadlock-free.
    class Context
    {
    public:
        Context() = default;
        ~Context();                         // defined in fastfork.cc; calls wait_local
        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        void operator++() noexcept { m_n.fetch_add(1, std::memory_order_relaxed); }
        void operator--() noexcept { m_n.fetch_sub(1, std::memory_order_release); }
        bool empty()  const noexcept { return m_n.load(std::memory_order_acquire) == 0; }

    private:
        std::atomic<int> m_n{0};
    };

    // Fork a task and register it with a Context batch.
    void fork_task(Context& ctx, task_t t);

    // Participate in work-stealing until ctx is empty.
    void wait_local(Context& ctx);

    // Per-worker statistics (indexed by thread ID 0..get_max_threads()-1).
    struct WorkerStats
    {
        uint64_t tasks_executed_own{0}; // tasks run from thread's own queue
        uint64_t tasks_stolen{0};       // tasks run stolen from another thread
        uint64_t idle_polls{0};         // full steal_order scans that found nothing
        uint64_t tasks_enqueued{0};     // tasks deposited into this thread's queue
    };

    std::vector<WorkerStats> get_worker_stats();
    void                     reset_worker_stats();
}#pragma once

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
        void load_files_from_dir(const std::string& train_corpus_dir);

        using visitor_fn_t       = std::function<void(const InputLine&)>;
        using token_visitor_fn_t = std::function<void(TokenID)>;

        InputLine get_token_ids(const std::string& text) const;
        Token get_token_from_id(TokenID id) const;
        std::optional<std::string> get_line(const InputLine& line) const;

        std::vector<InputLine> get_suitable_training_lines() const;

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
                    visitor(line);
            }

            void visit_tokens(const token_visitor_fn_t& visitor) const
            {
                for (const auto tok : m_tokens_in_file)
                    visitor(tok);
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
        mutable size_t m_tokenization_errors = 0;
    };

} // namespace rllm
#pragma once

#include <cstddef>

namespace rllm
{
    /** iterates from Enum1::START to Enum1::MAX and Enum2::START to Enum2::MAX
     * Conventiently assumes that Enum1 and Enum2 have contiguous values starting from START and that MAX is one past the last valid value.
    */
    template <typename Enum1, typename Enum2>
    class enum_iterator2D
    {
      public:
        enum_iterator2D(Enum1 end1 = Enum1::MAX, Enum2 end2 = Enum2::MAX)
            : m_pos(0)
            , m_end((static_cast<size_t>(end1) - static_cast<size_t>(Enum1::START)) *
                    (static_cast<size_t>(end2) - static_cast<size_t>(Enum2::START)))
            , m_size2(static_cast<size_t>(end2) - static_cast<size_t>(Enum2::START))
        {}

        std::pair<Enum1, Enum2> operator*() const
        {
            return {
                static_cast<Enum1>(static_cast<size_t>(Enum1::START) + m_pos / m_size2),
                static_cast<Enum2>(static_cast<size_t>(Enum2::START) + m_pos % m_size2)
            };
        }

        enum_iterator2D& operator++()
        {
            ++m_pos;
            return *this;
        }

        bool operator!=(const enum_iterator2D& other) const
        {
            return m_pos != other.m_pos;
        }

        void operator+=(size_t offset)
        {
            m_pos = (m_pos + offset) % m_end;
        }

        int operator-(const enum_iterator2D& other) const
        {
            return static_cast<int>(m_pos) - static_cast<int>(other.m_pos);
        }

        enum_iterator2D begin() const { return {0,     m_end, m_size2}; }
        enum_iterator2D end()   const { return {m_end, m_end, m_size2}; }

        // Number of inner (v2) values per outer (v1) step.
        size_t inner_size() const noexcept { return m_size2; }
        // Number of distinct v1 (outer) rows.
        size_t outer_size() const noexcept { return m_end / m_size2; }

        // Lightweight range over one outer row  suitable for range-for.
        struct RowRange
        {
            enum_iterator2D m_begin, m_end;
            enum_iterator2D begin() const noexcept { return m_begin; }
            enum_iterator2D end()   const noexcept { return m_end;   }
        };

        RowRange row_range(size_t outer_idx) const noexcept
        {
            const size_t start = outer_idx * m_size2;
            return {iter_at(start), iter_at(start + m_size2)};
        }

        // Range over a rectangular block [outer_begin, outer_end) x [inner_begin, inner_end).
        // Iterates row-major: for each outer row, scans all inner columns in the block.
        struct BlockRange
        {
            struct iterator
            {
                size_t m_inner_begin, m_inner_end;
                size_t m_oi, m_ii; // current outer/inner indices (relative to Enum::START)

                std::pair<Enum1, Enum2> operator*() const noexcept
                {
                    return {
                        static_cast<Enum1>(static_cast<size_t>(Enum1::START) + m_oi),
                        static_cast<Enum2>(static_cast<size_t>(Enum2::START) + m_ii)
                    };
                }

                iterator& operator++() noexcept
                {
                    if (++m_ii >= m_inner_end)
                    {
                        m_ii = m_inner_begin;
                        ++m_oi;
                    }
                    return *this;
                }

                bool operator!=(const iterator& other) const noexcept
                {
                    return m_oi != other.m_oi || m_ii != other.m_ii;
                }
            };

            size_t m_outer_begin, m_outer_end;
            size_t m_inner_begin, m_inner_end;

            iterator begin() const noexcept
            {
                return {m_inner_begin, m_inner_end, m_outer_begin, m_inner_begin};
            }
            iterator end() const noexcept
            {
                return {m_inner_begin, m_inner_end, m_outer_end, m_inner_begin};
            }
        };

        BlockRange block_range(size_t outer_begin, size_t outer_end,
                               size_t inner_begin, size_t inner_end) const noexcept
        {
            return {outer_begin, outer_end, inner_begin, inner_end};
        }

      private:
        enum_iterator2D iter_at(size_t pos) const noexcept
        {
            return {pos, m_end, m_size2};
        }

        enum_iterator2D(size_t pos, size_t end, size_t size2)
            : m_pos(pos), m_end(end), m_size2(size2)
        {}

        size_t m_pos;    // flat index in [0, m_end)
        size_t m_end;    // total iterations = size1 * size2
        size_t m_size2;  // stride of the inner dimension
    };
} // namespace rllm
#pragma once

#include <cstddef>

namespace rllm
{
    template <typename Enum>
    class enum_iterator
    {
      public:
        enum_iterator(Enum end = Enum::MAX)
                        : m_start(Enum::START)
                        , m_current(Enum::START)
            , m_end(end)
        {}

        enum_iterator(Enum start, Enum end)
                        : m_start(start)
                        , m_current(start)
            , m_end(end)
        {}

        Enum operator*() const
        {
            return m_current;
        }
        enum_iterator& operator++()
        {
            m_current = inc(m_current);
            return *this;
        }

        bool operator!=(const enum_iterator& other) const
        {
            return m_current != other.m_current;
        }

        void operator+=(size_t offset)
        {
            m_current = static_cast<Enum>((static_cast<size_t>(m_current) + offset) % static_cast<size_t>(m_end));
        }

        int operator-(const enum_iterator& other) const
        {
            return static_cast<int>(m_current) - static_cast<int>(other.m_current);
        }

        enum_iterator begin() const
        {
            return enum_iterator{m_start, m_end};
        }

        enum_iterator end() const
        {
            return enum_iterator{m_end, m_end};
        }

      private:
                Enum m_start;
        Enum m_current;
        Enum m_end;
    };
} // namespace rllm
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>
#include <RandomHelpers.hpp>

namespace rllm
{
    /** We assume that the ElementType supports default construction, copy assignment, and arithmetic operations.
     * Should be sth like float/float16/int8, not a complex struct.  The X and Y template parameters are the enum
     * types for the row and column indices, respectively, and are only used to determine the matrix dimensions
     * and provide type safety for indexing.
     */
    template <typename ElementType, typename X, typename Y>
    class fixed_size_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        fixed_size_matrix()
        {
            m_data.fill(ElementType{});
        }
        ~fixed_size_matrix() = default;

        inline void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        inline ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline void fill(ElementType value)
        {
            m_data.fill(value);
        }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            for (auto& v : m_data)
                v = static_cast<ElementType>(get_random_value(lo, hi));
        }

        inline void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        inline void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        inline void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] += delta;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
    };
} // namespace rllm
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

#include <Range.hpp>
#include <enum_iterator.hpp>

namespace rllm
{
    template <typename T, typename LengthType>
    class fixed_size_vector
    {
      public:
        fixed_size_vector()
        {
            if constexpr (std::is_trivially_default_constructible_v<T>)
                m_data.fill(T{});
            // else: T's own default constructor already initialises each element.
        }

        ~fixed_size_vector() = default;

        float get_highest_value(const LengthType length) const
        {
            float max_value = std::numeric_limits<float>::lowest();
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto val = m_data[static_cast<size_t>(i)];
                if (val > max_value)
                {
                    max_value = val;
                }
            }
            return max_value;
        }

        /** Normalize the values using the softmax function.
         * Each element is <= 1 and >= 0, and the sum of all elements is 1.
         * This is useful for converting the output of the neural network into
         * a probability distribution over the tokens.
         */
        void normalize_using_softmax(const LengthType length)
        {
            const auto max_value = get_highest_value(length);

            float sum_exp = 0.0f;
            for (const auto i : enum_iterator<LengthType>(length))
            {
                sum_exp += std::exp(m_data[static_cast<size_t>(i)] - max_value);
            }

            for (const auto i : enum_iterator<LengthType>(length))
            {
                m_data[static_cast<size_t>(i)] = std::exp(m_data[static_cast<size_t>(i)] - max_value) / sum_exp;
            }
        }

        fixed_size_vector sub_array(LengthType length) const
        {
            assert(length <= len);
            fixed_size_vector result;
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto tok = m_data[static_cast<size_t>(i)];
                result.push_back(tok);
            }
            result.len = length;
            return result;
        }

        void push_back(T value)
        {
            assert(len < LengthType::MAX);
            m_data[static_cast<size_t>(len)] = value;
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        const T& back() const
        {
            assert(len > LengthType::START);
            return m_data[static_cast<size_t>(len) - 1];
        }

        void pop_back()
        {
            assert(len > LengthType::START);
            len = static_cast<LengthType>(static_cast<size_t>(len) - 1);
        }

        bool empty() const
        {
            return len == LengthType::START;
        }

        LengthType size() const
        {
            return len;
        }

        T& operator[](LengthType index)
        {
            return m_data[static_cast<size_t>(index)];
        }

        const T& operator[](LengthType index) const
        {
            return m_data[static_cast<size_t>(index)];
        }

        void fill(T value)
        {
            m_data.fill(value);
        }

        /** add a value to an element at index with clamping */
        void add_with_clamp(LengthType index, T delta, Range<T> range)
        {
            auto& cell = m_data[static_cast<size_t>(index)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        /** add a value to an element at index without clamping */
        void add_no_clamp(LengthType index, T delta)
        {
            m_data[static_cast<size_t>(index)] += delta;
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        using token_vector_data_t = std::array<T, static_cast<size_t>(LengthType::MAX)>;
        token_vector_data_t m_data;
        LengthType len = LengthType::START;
    };
} // namespace rllm
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>

namespace rllm
{
    /** the number of columns can vary, the number of rows are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_cols_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_cols_matrix()
            : m_cols(Y::MAX)
        {
            m_data.fill(ElementType{});
        }

        flexible_cols_matrix(Y cols)
            : m_cols(cols)
        {
            m_data.fill(ElementType{});
        }

        ~flexible_cols_matrix() = default;

        void set_cols(Y cols)
        {
            assert(static_cast<size_t>(cols) <= COLS);
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        // Adds each element of other (must have the same runtime dimensions) into this matrix.
        void element_wise_add(const flexible_cols_matrix& other)
        {
            assert(ROWS == other.ROWS && m_cols == other.m_cols);
            const size_t n = static_cast<size_t>(ROWS) * static_cast<size_t>(m_cols);
#pragma omp simd
            for (size_t i = 0; i < n; ++i)
                m_data[i] += other.m_data[i];
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            auto& cell = m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] += delta;
        }

        const X num_rows() const
        {
            return ROWS;
        }

        Y num_cols() const
        {
            return m_cols;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
        Y m_cols;
    };

} // namespace rllm
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>

namespace rllm
{
    /** the number of columns AND rows can vary */
    template <typename ElementType, typename X, typename Y>
    class flexible_rows_cols_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_rows_cols_matrix()
            : m_rows(X::MAX), m_cols(Y::MAX)
        {
            m_data.fill(ElementType{});
        }

        flexible_rows_cols_matrix( X rows, Y cols)
            : m_rows(rows), m_cols(cols)
        {
            m_data.fill(ElementType{});
        }

        ~flexible_rows_cols_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            m_rows = rows;
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        // Adds each element of other (must have the same runtime dimensions) into this matrix.
        void element_wise_add(const flexible_rows_cols_matrix& other)
        {
            assert(static_cast<size_t>(m_rows) == static_cast<size_t>(other.m_rows) && static_cast<size_t>(m_cols) == static_cast<size_t>(other.m_cols));
            const size_t n = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
#pragma omp simd
            for (size_t i = 0; i < n; ++i)
                m_data[i] += other.m_data[i];
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            auto& cell = m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] += delta;
        }

        X num_rows() const
        {
            return m_rows;
        }

        Y num_cols() const
        {
            return m_cols;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
        X m_rows;
        Y m_cols;
    };

} // namespace rllm
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>

namespace rllm
{
    /** the number of rows can vary, the number of columns are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_rows_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_rows_matrix()
            : m_rows(X::MAX)
        {
            m_data.fill(ElementType{});
        }

        flexible_rows_matrix(X rows)
            : m_rows(rows)
        {
            m_data.fill(ElementType{});
        }

        ~flexible_rows_matrix() = default;

        void set_rows(X rows)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            m_rows = rows;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        // Adds each element of other (must have the same runtime dimensions) into this matrix.
        void element_wise_add(const flexible_rows_matrix& other)
        {
            assert(m_rows == other.m_rows);
            const size_t n = static_cast<size_t>(m_rows) * COLS;
#pragma omp simd
            for (size_t i = 0; i < n; ++i)
                m_data[i] += other.m_data[i];
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] += delta;
        }

        X num_rows() const
        {
            return m_rows;
        }

        constexpr Y num_cols() const
        {
            return Y::MAX;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
        X m_rows;
    };

} // namespace rllm
#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <vector>

namespace rllm
{
    // InputLayer converts an InputLine (sequence of token IDs) into a
    // flat hidden-state vector h[seq_len  EmbeddingDimension::MAX].
    // Each position receives its learned token embedding plus a
    // fixed sinusoidal positional encoding.
    class InputLayer
    {
      public:
        InputLayer()
        {
            reset_embeddings();
        }
        ~InputLayer() = default;
        InputLayer(const InputLayer&) = delete;
        InputLayer& operator=(const InputLayer&) = delete;

        // Fill h[seq_len  D_MODEL] with (token_embedding + positional_encoding).
        void propagate_forward(const InputLine& input,
                flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h) const;

        // Update token embeddings using dh[seq_len  D_MODEL] =    L/   h.
        // Positional encodings are fixed (sinusoidal), so only embeddings change.
        void propagate_backward(
            const InputLine& input,
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dh,
            float learning_rate
        );

        void set_random_embeddings();

        // Returns the raw learned embedding for a single token (without positional encoding).
        const fixed_size_vector<rlmm_float_small, EmbeddingDimension>& get_embedding(TokenID tok) const
        {
            return m_embeddings[tok];
        }

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        // m_embeddings[token_id][d]  learned embedding for dimension d of token_id.
        fixed_size_vector<fixed_size_vector<rlmm_float_small, EmbeddingDimension>, TokenID> m_embeddings;

        void reset_embeddings();
    };

} // namespace rllm
#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace rllm::json_helpers
{
    template <typename Enum>
    constexpr size_t enum_max()
    {
        return static_cast<size_t>(Enum::MAX);
    }

    template <typename Enum, typename T>
    nlohmann::json serialize_vector(const fixed_size_vector<T, Enum>& v)
    {
        auto out = nlohmann::json::array();
        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            out.push_back(v[static_cast<Enum>(i)]);
        }
        return out;
    }

    template <typename Enum, typename T>
    void deserialize_vector(const nlohmann::json& j, fixed_size_vector<T, Enum>& v)
    {
        if (!j.is_array() || j.size() != enum_max<Enum>())
        {
            throw std::runtime_error("Invalid vector shape in model JSON");
        }

        for (size_t i = 0; i < enum_max<Enum>(); ++i)
        {
            v[static_cast<Enum>(i)] = j.at(i).template get<T>();
        }
    }

    template <typename T, typename X, typename Y>
    nlohmann::json serialize_matrix(const fixed_size_matrix<T, X, Y>& m)
    {
        auto rows = nlohmann::json::array();

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            auto cols = nlohmann::json::array();
            for (size_t y = 0; y < enum_max<Y>(); ++y)
            {
                cols.push_back(m.get(static_cast<X>(x), static_cast<Y>(y)));
            }
            rows.push_back(std::move(cols));
        }

        return rows;
    }

    template <typename T, typename X, typename Y>
    void deserialize_matrix(const nlohmann::json& j, fixed_size_matrix<T, X, Y>& m)
    {
        if (!j.is_array() || j.size() != enum_max<X>())
        {
            throw std::runtime_error("Invalid matrix row count in model JSON");
        }

        for (size_t x = 0; x < enum_max<X>(); ++x)
        {
            const auto& row = j.at(x);
            if (!row.is_array() || row.size() != enum_max<Y>())
            {
                throw std::runtime_error("Invalid matrix column count in model JSON");
            }

            for (size_t y = 0; y < enum_max<Y>(); ++y)
            {
                m.set(static_cast<X>(x), static_cast<Y>(y), row.at(y).template get<T>());
            }
        }
    }



} // namespace rllm::json_helpers

namespace rllm::json_helpers
{
} // namespace rllm::json_helpers
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include <math_utils.hpp>

#include <RandomHelpers.hpp>
#include <tokenizer_map.hpp>
#include <Range.hpp>
#include <enum_iterator.hpp>
#include <flexible_cols_matrix.hpp>
#include <flexible_rows_matrix.hpp>
#include <flexible_rows_cols_matrix.hpp>
#include <fixed_size_matrix.hpp>
#include <fixed_size_vector.hpp>

namespace rllm
{
    //using rlmm_float_small = _Float16;
    using rlmm_float_small = float;
    using rlmm_float = float;
    static constexpr rlmm_float RLMM_ZERO = rlmm_float{0};
    static constexpr rlmm_float RLMM_ONE = rlmm_float{1};
    static constexpr rlmm_float RLMM_NEG_ONE = rlmm_float{-1};

    static constexpr float MIN_NEURON_INPUT = -0.01f;
    static constexpr float MAX_NEURON_INPUT = 1.0f;

    static constexpr auto RED = "\033[31m";
    static constexpr auto RESET = "\033[0m";

#define LOG_ONCE(...) \
    do \
    { \
        static int counter = 0; \
        if (counter < 3) \
        { \
            __VA_ARGS__; \
            ++counter; \
        } \
    } while (0)


    using Token = std::string;

    // Dimensionality of each token's learned embedding vector.
    // The first intermediate layer is tiled as: neuron[p * EmbeddingDimension::MAX + d] for position p, dimension d.
    enum class EmbeddingDimension : size_t
    {
        START = 0,
        MAX = 512
    };

    // position of a token in the input sequence. For example, in the input "the cat sat", the token "cat" has
    // position 1.
    enum class PositionIndex : size_t
    {
        START = 0,
        MAX = 128,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // Index of an attention head (0..NUM_HEADS-1).
    enum class HeadsIndex : size_t
    {
        START = 0,
        MAX = 8
    };

    // Per-head embedding dimension: EmbeddingDimension::MAX / HeadsIndex::MAX = 64.
    enum class HeadDimension : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(EmbeddingDimension::MAX) / static_cast<size_t>(HeadsIndex::MAX)
    };

    // Feed-forward hidden dimension: static_cast<int>(FFDimension::MAX) = 4  EmbeddingDimension::MAX.
    enum class FFDimension : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(EmbeddingDimension::MAX) * 4
    };

    static inline TokenID inc(TokenID id)
    {
        assert(id != TokenID::UNKNOWN_TOKEN_ID);
        assert(id < TokenID::MAX);
        return static_cast<TokenID>(static_cast<int32_t>(id) + 1);
    }

    static inline EmbeddingDimension inc(EmbeddingDimension id)
    {
        assert(id < EmbeddingDimension::MAX);
        return static_cast<EmbeddingDimension>(static_cast<size_t>(id) + 1);
    }

    static inline PositionIndex inc(PositionIndex id)
    {
        assert(id != PositionIndex::UNKNOWN_POSITION_INDEX);
        assert(id < PositionIndex::MAX);
        return static_cast<PositionIndex>(static_cast<int32_t>(id) + 1);
    }

    static inline PositionIndex dec(PositionIndex id)
    {
        assert(id != PositionIndex::UNKNOWN_POSITION_INDEX);
        assert(id < PositionIndex::MAX);
        assert(id > PositionIndex::START);
        return static_cast<PositionIndex>(static_cast<int32_t>(id) - 1);
    }

    static inline HeadsIndex inc(HeadsIndex id)
    {
        assert(id < HeadsIndex::MAX);
        return static_cast<HeadsIndex>(static_cast<size_t>(id) + 1);
    }

    static inline HeadDimension inc(HeadDimension id)
    {
        assert(id < HeadDimension::MAX);
        return static_cast<HeadDimension>(static_cast<size_t>(id) + 1);
    }

    static inline FFDimension inc(FFDimension id)
    {
        assert(id < FFDimension::MAX);
        return static_cast<FFDimension>(static_cast<size_t>(id) + 1);
    }

    // Index of an outgoing connection slot within a single neuron's connection list.
    enum class NeuronConnectionIndex : size_t
    {
        START = 0,
        MAX = 128
    };

    static inline NeuronConnectionIndex inc(NeuronConnectionIndex id)
    {
        assert(id < NeuronConnectionIndex::MAX);
        return static_cast<NeuronConnectionIndex>(static_cast<size_t>(id) + 1);
    }

    using InputLine = fixed_size_vector<TokenID, PositionIndex>;


    struct Score
    {
        fixed_size_vector<rlmm_float, TokenID> values;
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
    };

} // namespace rllm
#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(RLLM_ENABLE_OVERFLOW_CHECK_ADD)
#define OVERFLOW_CHECK_ADD(a, b) ::math::check_add_not_overflows((a), (b))
#else
#define OVERFLOW_CHECK_ADD(a, b) ((void)0)
#endif

namespace math
{
    template <typename A, typename B>
    constexpr auto max(A a, B b)
    {
        using C = std::common_type_t<A, B>;
        const C ac = static_cast<C>(a);
        const C bc = static_cast<C>(b);
        return (ac < bc) ? bc : ac;
    }

    template <typename A, typename B>
    constexpr auto min(A a, B b)
    {
        using C = std::common_type_t<A, B>;
        const C ac = static_cast<C>(a);
        const C bc = static_cast<C>(b);
        return (bc < ac) ? bc : ac;
    }

    template <typename V, typename L, typename H>
    constexpr auto clamp(V v, L lo, H hi)
    {
        using C = std::common_type_t<V, L, H>;
        const C vc = static_cast<C>(v);
        const C lc = static_cast<C>(lo);
        const C hc = static_cast<C>(hi);
        return min(max(vc, lc), hc);
    }

    template <typename A, typename B>
    constexpr void check_add_not_overflows(A a, B b)
    {
        static_assert(std::is_arithmetic_v<A> && std::is_arithmetic_v<B>);
        using C = std::common_type_t<A, B>;

        const C ac = static_cast<C>(a);
        const C bc = static_cast<C>(b);

        if constexpr (std::is_floating_point_v<C>)
        {
            const C sum = ac + bc;
            assert(std::isfinite(sum));
            assert(sum <= std::numeric_limits<C>::max());
            assert(sum >= std::numeric_limits<C>::lowest());
        }
        else if constexpr (std::is_signed_v<C>)
        {
            const C max_c = std::numeric_limits<C>::max();
            const C min_c = std::numeric_limits<C>::lowest();
            assert((bc <= 0 || ac <= max_c - bc) && (bc >= 0 || ac >= min_c - bc));
        }
        else
        {
            const C max_c = std::numeric_limits<C>::max();
            assert(ac <= max_c - bc);
        }
    }
}
#pragma once

#include <cstddef>

#include <math_utils.hpp>

#include <parallel.hpp>
#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>
#include <fixed_size_matrix.hpp>
#include <flexible_rows_matrix.hpp>
#include <LayerPrimitives.hpp>

namespace rllm
{
    // C[m,n]  = A[m,k] @ B[n,k]^T   (B stored row-major [n  k])
    // m comes from A.num_rows() at runtime.
    template<typename K_enum, typename N_enum, typename BType>
    static void matmul_ABt(
        const flexible_rows_matrix<rlmm_float, PositionIndex, K_enum>& A,
        const fixed_size_matrix<BType, N_enum, K_enum>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, N_enum>& C)
    {
        const PositionIndex m = A.num_rows();
        PARFOR_2D(i, j, enum_iterator2D<PositionIndex, N_enum>(m))
            float sum = 0.f;
#pragma omp simd reduction(+:sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(K_enum::MAX); ++l_idx)
            {
                const float term = A[i, static_cast<K_enum>(l_idx)] * B[j, static_cast<K_enum>(l_idx)];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = static_cast<rlmm_float>(sum);
        ENDFOR
    }

    // C[m,n]  = A[m,k] @ B[k,n]     (B stored row-major [k  n])
    // m comes from A.num_rows() at runtime.
    template<typename K_enum, typename N_enum, typename BType>
    static void matmul_AB(
        const flexible_rows_matrix<rlmm_float, PositionIndex, K_enum>& A,
        const fixed_size_matrix<BType, K_enum, N_enum>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, N_enum>& C)
    {
        const PositionIndex m = A.num_rows();
        PARFOR_2D(i, j, enum_iterator2D<PositionIndex, N_enum>(m))
            float sum = 0.f;
#pragma omp simd reduction(+:sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(K_enum::MAX); ++l_idx)
            {
                const float term = A[i, static_cast<K_enum>(l_idx)] * B[static_cast<K_enum>(l_idx), j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = static_cast<rlmm_float>(sum);
        ENDFOR
    }

    // C[m,n] += A^T[m,k] @ B[k,n]   (A provided row-major [k  m]; accumulates into C)
    // k comes from A.num_rows() at runtime.
    template<typename M_enum, typename N_enum>
    static void matmul_AtB_acc(
        const flexible_rows_matrix<rlmm_float, PositionIndex, M_enum>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, N_enum>& B,
        fixed_size_matrix<rlmm_float, M_enum, N_enum>& C)
    {
        const PositionIndex k = A.num_rows();
        PARFOR_2D(i, j, enum_iterator2D<M_enum, N_enum>())
            float sum = 0.f;
            const size_t k_count = static_cast<size_t>(k);
#pragma omp simd reduction(+:sum)
            for (size_t l_idx = 0; l_idx < k_count; ++l_idx)
            {
                const float term = A[static_cast<PositionIndex>(l_idx), i] * B[static_cast<PositionIndex>(l_idx), j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<rlmm_float>(sum);
        ENDFOR
    }

} // namespace rllm
#pragma once

#include <Corpus.hpp>
#include <InputLayer.hpp>
#include <TransformerBlock.hpp>
#include <LayerPrimitives.hpp>
#include <OutputLayer.hpp>
#include <Statistics.hpp>

#include <nlohmann/json_fwd.hpp>
#include <chrono>
#include <random>
#include <string>
#include <vector>

namespace rllm
{
    void set_nn_log_file(const std::string& filename);

    enum class TrainingMethod
    {
        TWO_TOK,
        THREE_TOK,
        INCREASINGLY_LONGER_SEQUENCES,
        RANDOM_LINE_RANDOM_LEN,
        WINDOW, // sliding window of N tokens over flat corpus ((N-1) inputs  predict next)
    };

    const char* training_method_to_string(TrainingMethod method);

    class NeuralNetwork
    {
      public:
        // Denominator for convergence threshold: fires_nothing_ce_loss / k.
        // Higher k = tighter threshold = more gradient steps per example.
        static constexpr float k_convergence_divisor = 16.0f;

        NeuralNetwork(size_t num_layers, Corpus& corpus, Statistics& stats)
            : m_corpus(corpus)
            , m_stats(stats)
            , m_input_layer()
            , m_output_layer(corpus)
            // Compute CE-based constants from the actual corpus size.
            , m_fires_nothing_ce_loss(std::log(static_cast<float>(TokenID::MAX)))
            , m_convergence_threshold(m_fires_nothing_ce_loss / k_convergence_divisor)
        {
            assert(static_cast<size_t>(TokenID::MAX) > 1);
            for (size_t i = 0; i < num_layers; ++i)
                m_transformer_blocks.emplace_back();
        }
        ~NeuralNetwork() = default;
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        const Corpus& get_corpus() const { return m_corpus; }
        Statistics&   get_statistics() const { return m_stats; }
        const OutputLayer& get_output_layer() const { return m_output_layer; }
        const InputLayer& get_input_layer() const { return m_input_layer; }
        size_t get_transformer_block_count() const { return m_transformer_blocks.size(); }

        void set_training_method(TrainingMethod m) { m_training_method = m; }
        void set_window_size(int n) { assert(n >= 2); m_window_size = n; }

        void propagate_backward(const Score& score);
        void propagate_forward(const InputLine& input);

        // Returns the top-K output tokens with the highest activation.
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k) const;

        void train(
            bool verbose,
            size_t num_epochs,
            const std::optional<std::string>& input_filename,
            const std::optional<std::chrono::seconds>& checkpointing_interval = std::nullopt
        );
        float compute_loss(TokenID expected_output_token) const;
        void set_random_weights_and_connections();

        // returns true on success, false on failure (e.g. file not found or parse error)
        bool load(const std::string& filename);
        void save(const std::string& filename) const;

        // Mean-pool the last transformer block's hidden state over the sequence dimension.
        // Equivalent to last_hidden_state.mean(dim=1) in PyTorch.
        // Must be called after propagate_forward().
        fixed_size_vector<rlmm_float, EmbeddingDimension> get_last_hidden_mean() const
        {
            fixed_size_vector<rlmm_float, EmbeddingDimension> result;
            const size_t seq_len = static_cast<size_t>(m_seq_len);
            if (seq_len == 0)
                return result;
            for (const auto d : enum_iterator<EmbeddingDimension>())
            {
                float sum = 0.0f;
                for (const auto pos : enum_iterator<PositionIndex>(m_seq_len))
                    sum += static_cast<float>(m_last_hidden[pos, d]);
                result[d] = static_cast<rlmm_float>(sum / static_cast<float>(seq_len));
            }
            return result;
        }

      private:
        Corpus&    m_corpus;
        Statistics& m_stats;
        InputLayer  m_input_layer;
        InputLine   m_last_input;   // saved in propagate_forward for use in propagate_backward
        std::vector<TransformerBlock> m_transformer_blocks;
        OutputLayer m_output_layer;

        // Hidden state at the final position after the last transformer block.
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> m_last_hidden;
        PositionIndex m_seq_len{PositionIndex::START};

        // Computed from the actual corpus size.
        const float m_fires_nothing_ce_loss;
        const float m_convergence_threshold;

        void dump_top_predictions();
        void do_training(const InputLine& train_output, bool verbose, size_t max_iterations);

        TrainingMethod m_training_method = TrainingMethod::TWO_TOK;
        int m_window_size = 2;

        void train_with_up_to_N(const InputLine& line_of_file, bool verbose, size_t max_iterations, int num_tokens);
        void train_with_increasingly_longer_sequences(const InputLine& line_of_file, bool verbose, size_t max_iterations);
        void train_with_random_len_from_start(
            const InputLine& line_of_file,
            bool verbose,
            size_t max_iterations,
            std::mt19937& rng
        );
        void train_with_window(
            int window_size,
            bool verbose,
            size_t num_epochs,
            const std::optional<std::chrono::seconds>& checkpointing_interval
        );
        void train_random_line_random_len_epoch(
            size_t epoch,
            const std::vector<InputLine>& training_lines,
            bool verbose,
            size_t num_epochs,
            const std::optional<std::chrono::seconds>& checkpointing_interval,
            std::chrono::steady_clock::time_point& last_checkpoint_at,
            std::mt19937& rng
        );

        void do_whole_corpus_window_based_training(
            bool verbose,
            size_t num_epochs,
            const std::optional<std::chrono::seconds>& checkpointing_interval
        );
        void do_line_based_training(
            bool verbose,
            size_t num_epochs,
            const std::optional<std::chrono::seconds>& checkpointing_interval
        );

        bool training_method_is_line_based() const
        {
            switch (m_training_method)
            {
            case TrainingMethod::TWO_TOK:
            case TrainingMethod::THREE_TOK:
            case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
            case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
                return true;
            case TrainingMethod::WINDOW:
                return false;
            }
            return false;
        }
    };

} // namespace rllm
#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <Corpus.hpp>

namespace rllm
{
    // OutputLayer holds the learned linear projection ("LM head") from the last
    // transformer block's hidden state at the final sequence position to
    // vocabulary logits, plus the scoring and serialisation logic.
    //
    // W_lm_head is [TokenID::MAX  EmbeddingDimension::MAX] (out  in).
    class OutputLayer
    {
      public:
        OutputLayer(const Corpus& corpus);
        ~OutputLayer() = default;
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer& operator=(const OutputLayer&) = delete;

        // Initialise W_lm_head with small random values.
        void set_random_weights();

        // Project h_last[D_MODEL] to vocabulary logits, storing them in m_inputs.
        void forward_from_hidden(const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last);

        // Backpropagate the output delta through W_lm_head.
        // Returns d_h_last[D_MODEL] =    L/   h_last and updates W_lm_head via SGD+momentum.
        fixed_size_vector<rlmm_float, EmbeddingDimension> backward_and_update(
            const fixed_size_vector<rlmm_float, TokenID>& delta,
            const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
            float learning_rate
        );

        void compute_score(Score& score, const TokenID expected_output_token);
        void compute_deltas(const Score& score, fixed_size_vector<rlmm_float, TokenID>& deltas) const;
        void rms_normalize_inputs();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

        // Vocabulary logits computed by forward_from_hidden().
        fixed_size_vector<rlmm_float, TokenID> m_inputs;

      private:
        const Corpus& m_corpus;

        // LM head weight matrix [vocab  D_MODEL] (out  in), row-major.
        fixed_size_matrix<rlmm_float_small, TokenID, EmbeddingDimension> W_lm_head;
        fixed_size_matrix<rlmm_float, TokenID, EmbeddingDimension> V_lm_head; // SGD momentum velocities
    };

} // namespace rllm
#pragma once

//  Backend selection
// Controlled via cmake: -DPARALLEL_BACKEND=openmp|fastfork|sequential
// cmake translates that into -DUSE_OPENMP or -DUSE_FASTFORK; no define  sequential.

//  OpenMP backend
#if defined(USE_OPENMP)

#include <omp.h>

namespace parallel {
    void init_parallel();
    inline int  get_max_threads()      { return omp_get_max_threads(); }
    inline void set_num_threads(int n) { omp_set_num_threads(n); }
    inline int  get_thread_num()       { return omp_get_thread_num(); }
}

#define PARFOR(v, ...)          _Pragma("omp parallel for schedule(static)") \
                                for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  _Pragma("omp parallel for schedule(static)") \
                                for (auto [v1, v2] : (__VA_ARGS__)) {
#define ENDFOR }

// Parallel sections: each PARSECTION body runs concurrently.
// Usage: PARSECTIONS_BEGIN body1; PARSECTION body2; PARSECTION body3; PARSECTIONS_END
#define PARSECTIONS_BEGIN  _Pragma("omp parallel sections") { _Pragma("omp section")
#define PARSECTION         _Pragma("omp section")
#define PARSECTIONS_END    }

// No per-worker stats available under OpenMP; emit a no-op.
#define PARALLEL_DUMP_STATS() ((void)0)

//  fastfork backend
#elif defined(USE_FASTFORK)

#include <algorithm>
#include <bit>
#include <fastfork/fastfork.hpp>

namespace parallel {
    void init_parallel();
    inline int  get_max_threads() { return fastfork::get_max_threads(); }
    inline void set_num_threads(int n) { fastfork::set_num_threads(n); }
    // fastfork does not expose per-thread IDs
    inline int get_thread_num() { return fastfork::get_thread_num(); }
}

// Each loop iteration is forked as an independent task.
// The extra inner scope `{ {` lets ENDFOR use `}}` to close both PARFOR
// and PARFOR_2D uniformly while still yielding one task per outer row for 2D.
#define PARFOR(v, ...) \
    { auto _ff_rng_ = (__VA_ARGS__); \
      fastfork::Context _ff_ctx_; \
      for (auto v : _ff_rng_) \
          fastfork::fork_task(_ff_ctx_, [&, v]() { {
// PARFOR_2D partitions the 2D space into dynamically-sized blocks so that
// the total task count  FF_PARFOR_2D_TASKS_PER_THREAD  n_threads for large
// iteration spaces, or  n_threads for small ones (< FF_PARFOR_2D_SMALL_THRESH
// elements total) where fork-overhead would dominate fine-grained tasks.
//
// Tile-size derivation (all at runtime so set_num_threads() is respected):
//   K       = 1 if total < threshold, else FF_PARFOR_2D_TASKS_PER_THREAD
//   tile_outer = max(1, outer_size / n_threads)
//   n_outer_blocks = ceil(outer_size / tile_outer)
//   tile_inner = bit_floor(inner_size * n_outer_blocks / (K * n_threads))
//
// Examples with K thresholds and 48 threads (large  K=4, small  K=1):
//     8  512  (small)  tile 1  64     8   8  =  64 tasks  (1.3 threads)
//    64  512  (small)  tile 1  512   64   1  =  64 tasks  (1.3 threads)
//    64  2048 (large)  tile 1  512   64   4  = 256 tasks  (5.3 threads)
//   128  2048 (large)  tile 2  512   64   4  = 256 tasks  (5.3 threads)
//
// Override thresholds with -DFF_PARFOR_2D_TASKS_PER_THREAD=N or
// -DFF_PARFOR_2D_SMALL_THRESH=N.
#ifndef FF_PARFOR_2D_TASKS_PER_THREAD
#  define FF_PARFOR_2D_TASKS_PER_THREAD 4
#endif
#ifndef FF_PARFOR_2D_SMALL_THRESH
#  define FF_PARFOR_2D_SMALL_THRESH 100000
#endif
#define PARFOR_2D(v1, v2, ...) \
    { auto _ff_rng_ = (__VA_ARGS__); \
      fastfork::Context _ff_ctx_; \
      const size_t _ff_os_ = _ff_rng_.outer_size(); \
      const size_t _ff_is_ = _ff_rng_.inner_size(); \
      const size_t _ff_nt_ = static_cast<size_t>(fastfork::get_max_threads()); \
      const size_t _ff_k_  = (_ff_os_ * _ff_is_ < size_t{FF_PARFOR_2D_SMALL_THRESH}) \
                             ? size_t{1} : size_t{FF_PARFOR_2D_TASKS_PER_THREAD}; \
      const size_t _ff_to_ = math::max(size_t{1}, _ff_os_ / _ff_nt_); \
      const size_t _ff_n_ob_ = (_ff_os_ + _ff_to_ - 1) / _ff_to_; \
      const size_t _ff_ti_ = std::bit_floor( \
          math::max(size_t{1}, \
                   _ff_is_ * _ff_n_ob_ / (_ff_k_ * _ff_nt_))); \
      for (size_t _ff_ob_ = 0; _ff_ob_ < _ff_os_; _ff_ob_ += _ff_to_) \
          for (size_t _ff_ib_ = 0; _ff_ib_ < _ff_is_; _ff_ib_ += _ff_ti_) \
              fastfork::fork_task(_ff_ctx_, [&, _ff_ob_, _ff_ib_, _ff_to_, _ff_ti_]() { \
                  const size_t _ff_oe_ = math::min(_ff_ob_ + _ff_to_, _ff_os_); \
                  const size_t _ff_ie_ = math::min(_ff_ib_ + _ff_ti_, _ff_is_); \
                  for (const auto [v1, v2] : _ff_rng_.block_range(_ff_ob_, _ff_oe_, _ff_ib_, _ff_ie_)) {
#define ENDFOR \
          }}); \
      }

#define PARSECTIONS_BEGIN  { fastfork::Context _ff_ctx_; fastfork::fork_task(_ff_ctx_, [&]() {
#define PARSECTION         }); fastfork::fork_task(_ff_ctx_, [&]() {
#define PARSECTIONS_END    }); }

// Dump per-worker fastfork statistics to stderr then reset the counters.
#define PARALLEL_DUMP_STATS() \
    do { \
        const auto _ff_stats_ = fastfork::get_worker_stats(); \
        const int  _ff_nw_    = fastfork::get_max_threads(); \
        fprintf(stderr, "  %4s  %12s  %12s  %12s  %12s\n", \
                "tid", "exec_own", "stolen", "idle_polls", "enqueued"); \
        for (int _i_ = 0; _i_ < _ff_nw_; ++_i_) \
            fprintf(stderr, "  %4d  %12llu  %12llu  %12llu  %12llu\n", \
                    _i_, \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_executed_own), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_stolen), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].idle_polls), \
                    static_cast<unsigned long long>(_ff_stats_[_i_].tasks_enqueued)); \
        fastfork::reset_worker_stats(); \
    } while (false)

//  Sequential fallback
#else

namespace parallel {
    inline int  get_max_threads()      { return 1; }
    inline void set_num_threads(int)   {}
    inline int  get_thread_num()       { return 0; }
    void init_parallel();
}

#define PARFOR(v, ...)          for (auto v : (__VA_ARGS__)) {
#define PARFOR_2D(v1, v2, ...)  for (auto [v1, v2] : (__VA_ARGS__)) {
#define ENDFOR }

#define PARSECTIONS_BEGIN  {
#define PARSECTION
#define PARSECTIONS_END    }

#define PARALLEL_DUMP_STATS() ((void)0)

#endif
#pragma once

#include <cstdlib>
#include <utility>

namespace rllm
{
    inline size_t random_int(size_t min, size_t max)
    {
        return min + rand() % (max - min + 1);
    }

    /** returns a random value in the range [min, max] */
    inline float get_random_value(float min, float max)
    {
        return min + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (max - min);
    }

    template<typename T>
    T get_random_enum_value(T max_value = T::MAX) {
        static_assert(RAND_MAX >= static_cast<int>(T::MAX), "RAND_MAX must be greater than or equal to the number of enum values");
        return static_cast<T>(rand() % static_cast<int>(max_value));
    }

    template<typename T>
    T get_random_enum_value_centered_around(T center, int range) {
        assert(RAND_MAX >= (2 * range + 1)); // sanity check to ensure RAND_MAX is sufficient
        int offset = (rand() % (2 * range + 1)) - range; // random value in [-range, range]
        int raw_value = static_cast<int>(center) + offset;
        int wrapped_value = (raw_value + static_cast<int>(T::MAX)) % static_cast<int>(T::MAX); // wrap around using modulo
        return static_cast<T>(wrapped_value);
    }

} // namespace rllm
#pragma once

namespace rllm
{
    template <typename T>
    struct Range
    {
        T lo = T{0};
        T hi = T{1};
    };
} // namespace rllm
#pragma once

#include <NeuralNetwork.hpp>

#include <chrono>
#include <string>

namespace rllm
{
    class RLLM
    {
      public:
        RLLM(const std::vector<std::string>& filters);
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(
            const std::optional<std::string>& input_filename,
            const std::string& output_filename,
            size_t num_layers,
            bool verbose,
            TrainingMethod method,
            std::optional<std::chrono::seconds> checkpointing_interval,
            int window_size,
            size_t num_epochs,
            const std::string& train_corpus_dir
        );
        void prompt_mode(const std::string& input_filename, const std::optional<std::string>& one_shot_prompt = std::nullopt);

        struct PromptOptions
        {
            bool highest_prio_only = true;
        };

      private:
        void process_line(const std::string& line, Corpus& corpus, NeuralNetwork& nn, PromptOptions& options);

        const std::vector<std::string> m_filters;
    };

} // namespace rllm#pragma once

#include <chrono>
#include <cstdint>
#include <print>
#include <atomic>
#include <string>

namespace rllm
{


    class Statistics
    {
      public:
        class TotalLearnRecorderScope
        {
          public:
            TotalLearnRecorderScope(Statistics& stats)
                : m_stats(stats)
            {
                start  = std::chrono::steady_clock::now();
            }

            ~TotalLearnRecorderScope()
            {
                const auto end = std::chrono::steady_clock::now();
                const auto sum = end - start;

                m_stats.total_learning_duration.fetch_add(std::chrono::duration_cast<std::chrono::milliseconds>(sum).count());
            }

          private:
            Statistics& m_stats;
            std::chrono::steady_clock::time_point start;
        };


        void record_learning_failure()
        {
            m_learning_failures++;
        }

        void record_learning_success()
        {
            m_learning_successes++;
        }

        void print_statistics() const
        {
            std::println("Total learning failures: {}", m_learning_failures);
            std::println("Total learning successes: {}", m_learning_successes);

            auto duration = total_learning_duration.load();
            std::println("Total learning process took {} ms", duration);
        }

        size_t num_learning_failures() const { return m_learning_failures; }
        size_t num_learning_successes() const { return m_learning_successes; }
        size_t total_learning_duration_ms() const { return total_learning_duration.load(); }

      private:
        size_t m_learning_failures = 0;
        size_t m_learning_successes = 0;
        std::atomic<size_t> total_learning_duration{0};
    };
} // namespace rllm#pragma once
#include <Corpus.hpp>
#include <format>

namespace std {
    template<>
    struct formatter<rllm::TokenID> {
        constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
        auto format(const rllm::TokenID& id, std::format_context& ctx) const {
            return std::format_to(ctx.out(), "{}", static_cast<std::underlying_type_t<rllm::TokenID>>(id));
        }
    };
}
#pragma once

#include <LayerPrimitives.hpp>
#include <matmul.hpp>
#include <parallel.hpp>
#include <nlohmann/json_fwd.hpp>
#include <memory>
#include <vector>

namespace rllm
{
    // Forward declaration for heap-allocated forward-pass activation workspace.
    struct ForwardWorkspace;
    // Forward declaration for heap-allocated backward-pass scratch workspace.
    struct BackwardWorkspace;
    // A single transformer decoder block:
    //   Pre-RMSNorm  causal multi-head self-attention  residual
    //   Pre-RMSNorm  SwiGLU feed-forward network        residual
    //
    // Hyperparameters (compile-time):
    //   D_MODEL   = EmbeddingDimension::MAX  = 512
    //   NUM_HEADS = static_cast<int>(HeadsIndex::MAX) = 8      HeadDimension::MAX = 64
    //   static_cast<int>(FFDimension::MAX)      = D_MODEL * 4              = 2048
    //
    // All weight matrices are stored [out  in] row-major on the heap.
    // Optimizer: SGD + momentum (   =0.9), gradient clip    1, vel clip    0.1, weight clamp    2.
    class TransformerBlock
    {
      public:
        TransformerBlock();            // defined in .cc after ForwardWorkspace is complete
        ~TransformerBlock();           // defined in .cc after ForwardWorkspace is complete
        TransformerBlock(TransformerBlock&&) noexcept;            // defined in .cc
        TransformerBlock& operator=(TransformerBlock&&) noexcept; // defined in .cc
        TransformerBlock(const TransformerBlock&) = delete;
        TransformerBlock& operator=(const TransformerBlock&) = delete;

        // Forward pass.  h[seq_len  D_MODEL] is modified in-place.
        // Caches intermediate activations for the backward pass.
        void forward(flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len);

        // Backward pass.  dout[seq_len  D_MODEL] = dL/dh_out.
        // Writes dL/dh_in into din (same shape) and updates all weights.
        void backward(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dout,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& din,
            float learning_rate
        );

        void randomize();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

        // Public RMS norm: used by NeuralNetwork to apply the final pre-LM-head norm.
        static void apply_rms_norm(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& y
        )
        {
            rms_norm(x, y);
        }

        // Test helper: expose causal softmax without exposing internals broadly.
        static void causal_softmax_for_test(
            flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& x,
            PositionIndex T
        )
        {
            causal_softmax(x, T);
        }

        // Test helper: expose softmax backward Jacobian application.
        static void softmax_backward_for_test(
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dp,
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& p,
            flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dscores,
            PositionIndex T
        )
        {
            softmax_backward(dp, p, dscores, T);
        }

      private:
        // Optimizer hyper-parameters
        static constexpr float MOMENTUM_BETA = 0.9f;
        static constexpr float GRAD_CLIP = 1.0f;
        static constexpr float VEL_CLIP = 0.1f;
        static constexpr float WEIGHT_CLAMP = 2.0f;

        // Attention weights [D_MODEL  D_MODEL] (out_dim  in_dim), row-major
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension> W_q, W_k, W_v, W_o;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> V_q, V_k, V_v, V_o;

        // SwiGLU FFN:
        //   gate, up:  [FFDimension::MAX     D_MODEL]  (out  in)
        //   down:      [D_MODEL  FFDimension::MAX   ]  (out  in)
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension> W_gate, W_up;
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension> W_down;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> V_gate, V_up;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension> V_down;

        // Activations cached during forward() for use in backward().
        // Heap-allocated to avoid blowing the stack (~21 MB of fixed-size arrays).
        std::unique_ptr<ForwardWorkspace> m_fwd_ws;
        // Scratch workspace for backward(); cached to avoid per-call heap allocation.
        std::unique_ptr<BackwardWorkspace> m_bwd_ws;

        // RMSNorm:  for each row t  y_t = x_t / rms(x_t)
        static void rms_norm(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& y
        );

        // RMSNorm backward: dx +=    L/   x  given dy =    L/   y and the original x
        static void rms_norm_backward(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dy,
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dx
        );

        // In-place causal softmax over the active [T  T] block of x.
        static void causal_softmax(flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& x,
          PositionIndex T);

        // Accumulates softmax backward into dscores (stride T).
        // dp is the per-head d_scores matrix; p is the cached per-head softmax matrix.
        static void softmax_backward(
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dp,
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& p,
            flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dscores,
            PositionIndex T
        );

        // SwiGLU backward: computes d_gate_pre and d_up_pre from d_ffn_act.
        static void swiglu_backward(
            PositionIndex seq,
            const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
            const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
            const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_ffn_act,
            flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_gate_pre,
            flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_up_pre
        );

        // SGD + momentum update: clips gradients, clips velocity, clamps weights.
        template <typename R_enum, typename C_enum, typename WType, typename VType, typename GType>
        static void sgd_update(
            fixed_size_matrix<WType, R_enum, C_enum>& W,
            fixed_size_matrix<VType, R_enum, C_enum>& vel,
            const fixed_size_matrix<GType, R_enum, C_enum>& grad,
            float lr
        )
        {
            PARFOR_2D(r, c, enum_iterator2D<R_enum, C_enum>())
                const float g = math::clamp(
                    grad[r, c],
                    -GRAD_CLIP,
                    GRAD_CLIP
                );
                vel[r, c] = math::clamp(
                    MOMENTUM_BETA * vel[r, c] + lr * g,
                    -VEL_CLIP,
                    VEL_CLIP
                );
                W[r, c] = math::clamp(
                    W[r, c] + vel[r, c],
                    -WEIGHT_CLAMP,
                    WEIGHT_CLAMP
                );
            ENDFOR
        }
    };

} // namespace rllm
