#pragma once

#include <Corpus.hpp>
#include <InputLayer.hpp>
#include <IntermediateLayer.hpp>
#include <LayerPrimitives.hpp>
#include <OutputLayer.hpp>
#include <Statistics.hpp>

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace rllm
{
    void set_nn_log_file(const std::string& filename);

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
            assert(static_cast<size_t>(TokenID::MAX) > 1); // need at least 2 token types for training to work
            for (size_t i = 0; i < num_layers; ++i)
            {
                m_intermediate_layers.emplace_back(m_corpus);
            }
        }
        ~NeuralNetwork() = default;
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        const Corpus& get_corpus() const
        {
            return m_corpus;
        }
        Statistics& get_statistics() const
        {
            return m_stats;
        }
        const OutputLayer& get_output_layer() const
        {
            return m_output_layer;
        }

        enum class TrainingMethod
        {
            TWO_TOK,
            THREE_TOK,
            INCREASINGLY_LONGER_SEQUENCES
        };

        void set_training_method(TrainingMethod m)
        {
            m_training_method = m;
        }

        void propagate_backward(const Score& score);
        void propagate_forward(const InputLine& input);

        // returns the top-K with the biggest activation in the output layer, as pairs of (token_id, activation_value)
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k) const;

        void train(bool verbose);
        // Compute mean squared error loss between output activations and expected output
        float compute_loss(TokenID expected_output_token) const;
        void set_random_weights_and_connections();

        void load(const std::string& filename);
        void save(const std::string& filename) const;

      private:
        Corpus& m_corpus;
        Statistics& m_stats;
        InputLayer m_input_layer;
        InputLine m_last_input; // stored during propagate_forward for use in propagate_backward
        std::vector<IntermediateLayer> m_intermediate_layers;
        OutputLayer m_output_layer;

        // Computed from the actual corpus size in set_random_weights_and_connections().
        const float m_fires_nothing_ce_loss; // should never see this value, overriden at runtime
        const float m_convergence_threshold; // should never see this value, overriden at runtime

        void dump_top_predictions();
        void do_training(const InputLine& train_output, bool verbose, size_t max_iterations);

        TrainingMethod m_training_method = TrainingMethod::TWO_TOK;

        void train_with_up_to_N(const InputLine& line_of_file, bool verbose, size_t max_iterations, int num_tokens);
        void
        train_with_increasingly_longer_sequences(const InputLine& line_of_file, bool verbose, size_t max_iterations);
    };

} // namespace rllm
