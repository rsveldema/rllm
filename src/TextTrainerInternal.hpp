#pragma once

#include <TextTrainer.hpp>

namespace rllm
{
    void scatter_dh_last_to_row(
        const fixed_size_vector<float, EmbeddingDimension>& dh_last,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        PositionIndex last_pos);
    void check_hidden_nan_finding_mode(VulkanQueue& queue,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
        PositionIndex rows, const char* phase);
    void check_hidden_vector_nan_finding_mode(VulkanQueue& queue,
        const fixed_size_vector<float, EmbeddingDimension>& values, const char* phase);
    void check_token_vector_nan_finding_mode(VulkanQueue& queue,
        const fixed_size_vector<float, TokenID>& values, float bound,
        const char* name, const char* phase);

    struct TextTrainerForwardWorkspace
    {
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> h;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> loop_input;
        fixed_size_vector<float, EmbeddingDimension> loop_iteration_embedding;
        fixed_size_vector<float, EmbeddingDimension> h_last;
        std::vector<ForwardWorkspace> transformer_workspaces;

        TextTrainerForwardWorkspace(PositionIndex seq_len, size_t num_transformer_blocks)
            : h(seq_len), loop_input(seq_len)
        {
            loop_iteration_embedding.set_size(EmbeddingDimension::MAX);
            transformer_workspaces.reserve(num_transformer_blocks);
            for (size_t i = 0; i < num_transformer_blocks; ++i)
                transformer_workspaces.emplace_back(seq_len);
            h_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(VulkanQueue& queue, PositionIndex seq_len)
        {
            h.set_rows(seq_len);
            loop_input.set_rows(seq_len);
            for (auto& workspace : transformer_workspaces)
                workspace.reset(queue, seq_len);
        }
    };

    void prepare_loop_iteration_embedding(
        TextTrainerForwardWorkspace& workspace, size_t iteration, VulkanQueue& queue);

    struct BackwardPropWorkspace
    {
        fixed_size_vector<float, TokenID> output_layer_delta;
        fixed_size_vector<float, PositionIndex> output_string_table_index_delta;
        fixed_size_vector<float, EmbeddingDimension> h_last_vec;
        fixed_size_vector<float, EmbeddingDimension> dh_last;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> dh;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> din;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> loop_input_gradient;
        BackwardWorkspace transformer_block;

        explicit BackwardPropWorkspace(PositionIndex seq_len)
            : dh(seq_len), din(seq_len), loop_input_gradient(seq_len), transformer_block(seq_len)
        {
            output_layer_delta.set_size(TokenID::MAX);
            output_string_table_index_delta.set_size(PositionIndex::MAX);
            h_last_vec.set_size(EmbeddingDimension::MAX);
            dh_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(VulkanQueue& queue, PositionIndex seq_len)
        {
            dh.set_rows(seq_len);
            din.set_rows(seq_len);
            loop_input_gradient.set_rows(seq_len);
            loop_input_gradient.zero(queue);
            transformer_block.reset(queue, seq_len);
            output_layer_delta.zero(queue);
            output_string_table_index_delta.zero(queue);
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
            embeddings.reset(queue);
        }
    };
}
