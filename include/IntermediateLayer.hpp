#pragma once

#include <LayerPrimitives.hpp>
#include <Corpus.hpp>

#include <nlohmann/json_fwd.hpp>


namespace rllm
{
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
            const template_token_vector<float, TokenID>& delta,
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

        friend class NeuralNetwork;
    };

} // namespace rllm
