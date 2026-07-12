#pragma once

#include <TextTrainer.hpp>

namespace rllm
{
    struct TextTrainerForwardWorkspace
    {
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> h;
        fixed_size_vector<float, EmbeddingDimension> h_last;
        std::vector<ForwardWorkspace> transformer_workspaces;

        TextTrainerForwardWorkspace(PositionIndex seq_len, size_t num_transformer_blocks)
            : h(seq_len)
        {
            transformer_workspaces.reserve(num_transformer_blocks);
            for (size_t i = 0; i < num_transformer_blocks; ++i)
                transformer_workspaces.emplace_back(seq_len);
            h_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(VulkanQueue& queue, PositionIndex seq_len)
        {
            h.set_rows(seq_len);
            for (auto& workspace : transformer_workspaces)
                workspace.reset(queue, seq_len);
        }
    };

    struct BackwardPropWorkspace
    {
        fixed_size_vector<float, TokenID> output_layer_delta;
        fixed_size_vector<float, EmbeddingDimension> h_last_vec;
        fixed_size_vector<float, EmbeddingDimension> dh_last;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> dh;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> din;
        BackwardWorkspace transformer_block;

        explicit BackwardPropWorkspace(PositionIndex seq_len)
            : dh(seq_len), din(seq_len), transformer_block(seq_len)
        {
            output_layer_delta.set_size(TokenID::MAX);
            h_last_vec.set_size(EmbeddingDimension::MAX);
            dh_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(VulkanQueue& queue, PositionIndex seq_len)
        {
            dh.set_rows(seq_len);
            din.set_rows(seq_len);
            transformer_block.reset(queue, seq_len);
            output_layer_delta.zero(queue);
            h_last_vec.zero(queue);
            dh_last.zero(queue);
        }
    };

    struct GradientAccumulationWorkspace
    {
        fixed_size_obj_vector<OutputLayerGradientAccumulator, MultiTokenPredictionIndex> output_layers;
        std::vector<TransformerGradientAccumulator> transformer_blocks;
        EmbeddingGradientAccumulator embeddings;

        explicit GradientAccumulationWorkspace(size_t num_transformer_blocks)
            : transformer_blocks(num_transformer_blocks)
        {
            output_layers.set_size(MultiTokenPredictionIndex::MAX);
        }

        void reset(VulkanQueue& queue)
        {
            for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
                output_layers[oi].reset(queue);
            for (auto& block : transformer_blocks)
                block.reset(queue);
            embeddings.reset();
        }
    };
}
