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
        NeuralNetwork(size_t num_layers)
            : m_intermediate_layers(num_layers)
        {}
        ~NeuralNetwork() = default;
        NeuralNetwork(const NeuralNetwork&) = delete;
        NeuralNetwork& operator=(const NeuralNetwork&) = delete;

        void compute_score(Score& score, const TokenID expected_output_token);
        void propagate_backward(const Score& score);
        void propagate_forward();

        void set_input_layer(const InputLine& input);

        // returns the top-K with the biggest activation in the output layer, as pairs of (token_id, activation_value)
        std::vector<OutputToken> get_best_output_token_ids(size_t top_k, Corpus& corpus) const;


        void train(Corpus& corpus);
        void set_random_weights_and_connections(Corpus& corpus);

        void load(const std::string& filename);
        void save(const std::string& filename) const;

      private:
        InputLayer m_input_layer;
        std::vector<IntermediateLayer> m_intermediate_layers;
        OutputLayer m_output_layer;

        void dump_top_predictions(Corpus& corpus);
        void dump_weights_and_triggers_for_token(TokenID token_id);
        void dump_path_weights_and_triggers(TokenID token_id) const;
        bool output_is_reachable_from_inputs(TokenID token_id) const;
        bool has_active_path_to_token(TokenID token_id) const;
    };

    class RLLM
    {
      public:
        RLLM();
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(const std::string& filename, size_t num_layers);
        void prompt_mode(const std::string& filename);
    };

} // namespace rllm