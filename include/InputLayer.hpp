#pragma once

#include <LayerPrimitives.hpp>
#include <Corpus.hpp>
#include <IntermediateLayer.hpp>

#include <nlohmann/json_fwd.hpp>

namespace rllm
{
    class InputLayer
    {
      public:
        InputLayer()
        {}
        ~InputLayer() = default;
        InputLayer(const InputLayer&) = delete;
        InputLayer& operator=(const InputLayer&) = delete;

        void set_input_layer(const InputLine& input);

        void set_input(const TokenID token_id, const PositionIndex pos_index, float value)
        {
            m_inputs.set(token_id, pos_index, value);
        }

        void propagate_forward(IntermediateLayer& next_layer) const;

        void set_random_weights_and_connections();
        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        template_token_matrix<float, TokenID, PositionIndex> m_inputs;

        // neuron 'i' is connected to neuron 'm_connections[i,j]' in the next layer
        template_token_matrix<std::pair<IntermediateLayerIndex, PositionIndex>, TokenID, PositionIndex>
            m_connections;

        friend class NeuralNetwork;
    };

} // namespace rllm
