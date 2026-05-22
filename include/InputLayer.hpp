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

        void propagate_forward(const InputLine& input, IntermediateLayer& next_layer) const;

        void load(const nlohmann::json& j);
        nlohmann::json save() const;
    };

} // namespace rllm
