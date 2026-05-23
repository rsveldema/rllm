#pragma once

#include <Corpus.hpp>
#include <IntermediateLayer.hpp>
#include <LayerPrimitives.hpp>

#include <array>
#include <nlohmann/json_fwd.hpp>

namespace rllm
{
    class InputLayer
    {
      public:
        // Each input position gets its own contiguous slice of the first intermediate layer.
        // EmbeddingDimension::MAX and IntermediateLayerIndex::MAX are defined in LayerPrimitives.hpp.
        // Token at position p maps to neurons [p*EmbeddingDimension::MAX, p*EmbeddingDimension::MAX + EmbeddingDimension::MAX).

        InputLayer()
        {
            reset_embeddings();
        }
        ~InputLayer() = default;
        InputLayer(const InputLayer&) = delete;
        InputLayer& operator=(const InputLayer&) = delete;

        void propagate_forward(const InputLine& input, IntermediateLayer& next_layer) const;

        // Update embeddings using the delta from the first intermediate layer.
        // delta[i] = ∂L/∂(first_intermediate_layer.m_inputs[i]) after the full backward pass.
        void propagate_backward(
            const InputLine& input,
            const template_token_vector<float, IntermediateLayerIndex>& delta,
            float learning_rate
        );

        void set_random_embeddings();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        // m_embeddings[token_id][d] — learned embedding value for dimension d of token token_id.
        template_token_vector<template_token_vector<float, EmbeddingDimension>, TokenID> m_embeddings;

        void reset_embeddings();
    };

} // namespace rllm
