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

        // Maps (token, position) to a single IntermediateLayerIndex via a mixing hash
        // so that different (tok, pos) pairs spread uniformly across the layer.
        static IntermediateLayerIndex hash_input(TokenID tok, PositionIndex pos)
        {
            const size_t t = static_cast<size_t>(tok);
            const size_t p = static_cast<size_t>(pos);
            const size_t h = (t * 2654435761ULL ^ p * 40503ULL)
                             % static_cast<size_t>(IntermediateLayerIndex::MAX);
            return static_cast<IntermediateLayerIndex>(h);
        }

        float get_input_value(TokenID tok, PositionIndex pos) const
        {
            if (pos >= m_input.size())
                return 0.0f;
            return m_input[pos] == tok ? 1.0f : 0.0f;
        }

        void propagate_forward(IntermediateLayer& next_layer) const;

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        InputLine m_input;

        friend class NeuralNetwork;
    };

} // namespace rllm
