#pragma once

#include <Corpus.hpp>
#include <InputLayer.hpp>
#include <IntermediateLayer.hpp>
#include <LayerPrimitives.hpp>
#include <OutputLayer.hpp>

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace rllm
{
    class NeuralNetwork
    {
      public:
        NeuralNetwork(size_t num_layers, Corpus& corpus)
        : m_corpus(corpus)
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

        void compute_score(Score& score, const TokenID expected_output_token);
        void propagate_backward(const Score& score);
        void propagate_forward();

        void set_input_layer(const InputLine& input);

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
        InputLayer m_input_layer;
        std::vector<IntermediateLayer> m_intermediate_layers;
        OutputLayer m_output_layer;

        void dump_top_predictions();
        void dump_weights_and_triggers_for_token(TokenID token_id);
        void dump_path_weights_and_triggers(TokenID token_id) const;
        void dump_neurons_whose_weights_were_increasing() const;
        bool output_is_reachable_from_inputs(TokenID token_id) const;
        bool has_active_path_to_token(TokenID token_id) const;
        void do_training(const InputLine& train_output, bool verbose);
    };

} // namespace rllm
