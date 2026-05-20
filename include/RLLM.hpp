#pragma once

#include <Corpus.hpp>
#include <InputLayer.hpp>
#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace rllm
{
    class OutputLayer
    {
      public:
        OutputLayer()
        {}
        ~OutputLayer() = default;
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer& operator=(const OutputLayer&) = delete;

        void set_random_weights_and_connections_for_output_layer(Corpus& corpus);

        void update_output_weights(const template_token_vector<float, TokenID>& delta, float learning_rate);

        void compute_score(Score& score, const TokenID expected_output_token);

       void compute_deltas(const Score& score, template_token_vector<float, TokenID> &deltas) const;
        void load(const nlohmann::json& j);
        nlohmann::json save() const;


      public:
        template_token_vector<float, TokenID> m_inputs;
        template_token_vector<float, TokenID> m_trigger_values;
        template_token_vector<float, TokenID> m_weights;
    };

    class IntermediateLayer
    {
      public:
        IntermediateLayer()
        {}
        ~IntermediateLayer() = default;
        IntermediateLayer(const IntermediateLayer&) = delete;
        IntermediateLayer& operator=(const IntermediateLayer&) = delete;
        IntermediateLayer(IntermediateLayer&&) = delete;
        IntermediateLayer& operator=(IntermediateLayer&&) = delete;

        void propagate_forward(IntermediateLayer& next_layer);
        void propagate_backward(
            const template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& delta,
            template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
            float learning_rate
        );

        void propagate_backward(
            const  template_token_vector<float, TokenID>& delta,
            template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
            float learning_rate
        );

        void set_random_weights_and_connections();
        void set_random_weights_and_connections_to_output_layer(Corpus& corpus);
        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        // accumulated input for each neuron in the layer
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> m_inputs;
        // if m_inputs[i] >= m_trigger_values[i], then neuron 'i' fires.
        // this value is learned during training and can be thought of as the "bias"
        // for the neuron.
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> m_trigger_values;
        // the weight of the connection from neuron 'i' in this layer
        // to neuron 'm_connections[i]' in the next layer
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex> m_weights;
        // neuron 'n' is connected to neuron 'm_connections[i,j]' in the next layer
        template_token_matrix<std::pair<IntermediateLayerIndex, PositionIndex>, IntermediateLayerIndex, PositionIndex>
            m_connections;
    };


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
    };

    class RLLM
    {
      public:
        RLLM();
        ~RLLM() = default;
        RLLM(const RLLM&) = delete;
        RLLM& operator=(const RLLM&) = delete;

        void train_mode(const std::string& filename);
        void prompt_mode(const std::string& filename);
    };

} // namespace rllm