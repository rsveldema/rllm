#pragma once

#include <Corpus.hpp>
#include <InputLayer.hpp>
#include <TransformerBlock.hpp>
#include <LayerPrimitives.hpp>
#include <OutputLayer.hpp>
#include <Statistics.hpp>

#include <nlohmann/json_fwd.hpp>
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
        WINDOW, // sliding window of N tokens over flat corpus ((N-1) inputs → predict next)
    };

    const char* training_method_to_string(TrainingMethod method);

    class NeuralNetwork
    {
      public:
        NeuralNetwork(size_t num_layers, Corpus& corpus, Statistics& stats)
            : m_corpus(corpus)
            , m_stats(stats)
            , m_input_layer()
            , m_output_layer(corpus)
            // Compute CE-based constants from the actual corpus size.
            , m_fires_nothing_ce_loss(std::log(static_cast<float>(TokenID::MAX)))
            , m_convergence_threshold(m_fires_nothing_ce_loss / 4.0f)
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

        void set_training_method(TrainingMethod m) { m_training_method = m; }
        void set_window_size(int n) { assert(n >= 2); m_window_size = n; }

        void propagate_backward(const Score& score);
        void propagate_forward(const InputLine& input);

        // Returns the top-K output tokens with the highest activation.
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k) const;

        void train(bool verbose, size_t num_epochs,
            const std::optional<std::string>& input_filename, const std::optional<size_t>& checkpointing_interval = std::nullopt);
        float compute_loss(TokenID expected_output_token) const;
        void set_random_weights_and_connections();

        // returns true on success, false on failure (e.g. file not found or parse error)
        bool load(const std::string& filename);
        void save(const std::string& filename) const;

        void disable_checkpointing() { m_checkpointing_interval = std::nullopt; }
        void enable_checkpointing(size_t interval) { m_checkpointing_interval = interval; }

      private:
        Corpus&    m_corpus;
        Statistics& m_stats;
        InputLayer  m_input_layer;
        InputLine   m_last_input;   // saved in propagate_forward for use in propagate_backward
        std::vector<TransformerBlock> m_transformer_blocks;
        OutputLayer m_output_layer;

        // Hidden state at the final position after the last transformer block.
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_last_hidden;
        PositionIndex m_seq_len{PositionIndex::START};

        // Computed from the actual corpus size.
        const float m_fires_nothing_ce_loss;
        const float m_convergence_threshold;

        std::optional<size_t> m_checkpointing_interval = 5000;

        void dump_top_predictions();
        void do_training(const InputLine& train_output, bool verbose, size_t max_iterations);

        TrainingMethod m_training_method = TrainingMethod::TWO_TOK;
        int m_window_size = 2;

        void train_with_up_to_N(const InputLine& line_of_file, bool verbose, size_t max_iterations, int num_tokens);
        void train_with_increasingly_longer_sequences(const InputLine& line_of_file, bool verbose, size_t max_iterations);
        void train_with_window(int window_size, bool verbose, size_t num_epochs);

        void do_whole_corpus_window_based_training(bool verbose, size_t num_epochs);
        void do_line_based_training(bool verbose, size_t num_epochs);

        bool training_method_is_line_based() const
        {
            switch (m_training_method)
            {
            case TrainingMethod::TWO_TOK:
            case TrainingMethod::THREE_TOK:
            case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                return true;
            case TrainingMethod::WINDOW:
                return false;
            }
            return false;
        }
    };

} // namespace rllm
