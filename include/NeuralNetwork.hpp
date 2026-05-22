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
    class NeuralNetwork
    {
      public:
        NeuralNetwork(size_t num_layers, Corpus& corpus, Statistics& stats)
        : m_corpus(corpus), m_stats(stats)
        {
            for (size_t i = 0; i < num_layers; ++i)
            {
                m_intermediate_layers.emplace_back(m_corpus);
            }
        }
        ~NeuralNetwork() = default;
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        const Corpus& get_corpus() const { return m_corpus; }
        Statistics& get_statistics() const { return m_stats; }
        const OutputLayer& get_output_layer() const { return m_output_layer; }

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
        std::vector<IntermediateLayer> m_intermediate_layers;
        OutputLayer m_output_layer;

        void dump_top_predictions();
        void do_training(const InputLine& train_output, bool verbose);
    };

} // namespace rllm
