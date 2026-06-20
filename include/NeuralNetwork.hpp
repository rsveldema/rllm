#pragma once

#include <Corpus.hpp>
#include <InputLayer.hpp>
#include <TransformerBlock.hpp>
#include <LayerPrimitives.hpp>
#include <OutputLayer.hpp>
#include <Statistics.hpp>
#include <string>

#include <fixed_size_obj_vector.hpp>

#include <nlohmann/json_fwd.hpp>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rllm
{
    void set_nn_log_file(const std::string& filename);
    struct NeuralNetworkForwardWorkspace;
    struct BackwardPropWorkspace;

    enum class TrainingMethod
    {
        TWO_TOK,
        THREE_TOK,
        INCREASINGLY_LONGER_SEQUENCES,
        RANDOM_LINE_RANDOM_LEN,
        RANDOM_LINE_FULL, // pick a random line, train on the full line (last token is target)
        WINDOW, // sliding window of N tokens over flat corpus ((N-1) inputs → predict next)
    };

    const char* training_method_to_string(TrainingMethod method);

    class NeuralNetwork
    {
      public:
        // Denominator for convergence threshold: fires_nothing_ce_loss / k.
        // Higher k = tighter threshold = more gradient steps per example.
        // NOTE: if set too high, we drive a specific example to high confidence but fail to learn from other examples, harming generalization.
        static constexpr float k_convergence_divisor = 2.0f;

        NeuralNetwork(size_t num_layers, Corpus& corpus, Statistics& stats);
        ~NeuralNetwork();
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        const Corpus& get_corpus() const { return m_corpus; }
        Statistics&   get_statistics() const { return m_stats; }
        const fixed_size_obj_vector<OutputLayer, MultiTokenPredictionIndex>& get_output_layers() const { return m_output_layers; }
        const OutputLayer& get_output_layer(MultiTokenPredictionIndex idx) const { return m_output_layers[idx]; }
        const InputLayer& get_input_layer() const { return m_input_layer; }
        size_t get_transformer_block_count() const { return m_transformer_blocks.size(); }

        void set_training_method(TrainingMethod m) { m_training_method = m; }
        void set_window_size(int n) { assert(n >= 2); m_window_size = n; }

        void propagate_forward();

        // Returns the top-K output tokens with the highest activation.
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k, MultiTokenPredictionIndex head) const;

        void train(
            bool verbose,
            size_t num_epochs,
            const std::optional<std::string>& input_filename,
            const std::optional<std::chrono::seconds>& checkpointing_interval = std::nullopt
        );

        void set_random_weights_and_connections();

        // returns true on success, false on failure (e.g. file not found or parse error)
        bool load(const std::string& filename);
        void checkpoint() const;
        void save(const std::string& filename) const;
        // Safetensors serialization
        void save_to_safetensors(const std::string& filename) const;
        bool load_from_safetensors(const std::string& filename);

        // Mean-pool the last transformer block's hidden state over the sequence dimension.
        // Equivalent to last_hidden_state.mean(dim=1) in PyTorch.
        // Must be called after propagate_forward().
        fixed_size_vector<float, EmbeddingDimension> get_last_hidden_mean() const
        {
            fixed_size_vector<float, EmbeddingDimension> result;
            const size_t seq_len = static_cast<size_t>(m_seq_len);
            if (seq_len == 0)
                return result;
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
            {
                float sum = 0.0f;
                for (const auto pos : enum_iterator1D<PositionIndex>(m_seq_len))
                    sum += static_cast<float>(m_last_hidden[pos, d]);
                result[d] = static_cast<float>(sum / static_cast<float>(seq_len));
            }
            return result;
        }

        /** set the input for the neural network.
         * Call this just before    propagate_forward() and then call propagate_backward_mtp() to train on this input.
         */
        InputLine& get_last_input() {
            return m_last_input;
        }
        

      private:
                static constexpr size_t VALIDATION_PERCENT = 20;
                static constexpr size_t EARLY_STOPPING_PATIENCE = 3;
                static constexpr float  VALIDATION_IMPROVEMENT_EPSILON = 1e-4f;

        Corpus&    m_corpus;
        Statistics& m_stats;
        InputLayer  m_input_layer;
        InputLine   m_last_input;   // saved in propagate_forward for use in propagate_backward
        std::vector<TransformerBlock> m_transformer_blocks;
        fixed_size_obj_vector<OutputLayer, MultiTokenPredictionIndex> m_output_layers;

        // Hidden state at the final position after the last transformer block.
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_last_hidden;
        PositionIndex m_seq_len{PositionIndex::START};
        std::unique_ptr<NeuralNetworkForwardWorkspace> m_forward_workspace;
        std::unique_ptr<BackwardPropWorkspace> m_backward_workspace;

        // Computed from the actual corpus size.
        const float m_fires_nothing_ce_loss;
        const float m_convergence_threshold;

        void dump_top_predictions();
        void trace_probes_for_example(const char* phase, size_t iter, float loss_value, const std::string& full_string);
        void do_training(const InputLine& train_output, bool verbose, size_t max_iterations);
        // Accumulates gradients from all valid MTP heads and backpropagates once.
        void propagate_backward_mtp(
            const fixed_size_obj_vector<Score, MultiTokenPredictionIndex>& scores,
            MultiTokenPredictionIndex num_valid
        );
        float evaluate_average_loss(const std::vector<InputLine>& evaluation_lines);

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
            case TrainingMethod::RANDOM_LINE_FULL:
                return true;
            case TrainingMethod::WINDOW:
                return false;
            }
            return false;
        }
    };

} // namespace rllm
