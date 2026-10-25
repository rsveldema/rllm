#pragma once

#include <TextTrainer.hpp>
#include <backprop/Muon.hpp>

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

        ForwardWorkspace& transformer_workspace(size_t execution)
        {
            assert(!transformer_workspaces.empty());
            return transformer_workspaces.size() == 1
                ? transformer_workspaces.front()
                : transformer_workspaces[execution];
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

    struct DustWorkspace
    {
        struct ActivationCache
        {
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> h_norm_attn;
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> attn_concat;
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> h_norm_ff;
            flexible_rows_matrix<float, PositionIndex, FFDimension> ffn_act;

            explicit ActivationCache(PositionIndex rows)
                : h_norm_attn(rows), attn_concat(rows), h_norm_ff(rows), ffn_act(rows)
            {}

            void reset(PositionIndex rows)
            {
                h_norm_attn.set_rows(rows);
                attn_concat.set_rows(rows);
                h_norm_ff.set_rows(rows);
                ffn_act.set_rows(rows);
            }

            size_t storage_size_bytes() const
            {
                return h_norm_attn.storage_size_bytes() + attn_concat.storage_size_bytes() +
                    h_norm_ff.storage_size_bytes() + ffn_act.storage_size_bytes();
            }
        };

        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> hidden_error;
        flexible_rows_matrix<float, PositionIndex, FFDimension> ff_error;
        std::vector<ActivationCache> activation_caches;
        dust::ForwardPerturbation perturbation;

        DustWorkspace(PositionIndex rows, size_t execution_count)
            : hidden_error(rows), ff_error(rows), perturbation(rows, 0, 0.0f, 0.0f)
        {
            activation_caches.reserve(execution_count);
            for (size_t execution = 0; execution < execution_count; ++execution)
                activation_caches.emplace_back(rows);
        }

        void reset(PositionIndex rows)
        {
            hidden_error.set_rows(rows);
            ff_error.set_rows(rows);
            for (auto& cache : activation_caches)
                cache.reset(rows);
        }

        void allocate_device_buffers(VulkanQueue& queue)
        {
            hidden_error.zero(queue);
            ff_error.zero(queue);
            perturbation.hidden_noise.zero(queue);
            perturbation.ff_noise.zero(queue);
            for (auto& cache : activation_caches)
            {
                cache.h_norm_attn.zero(queue);
                cache.attn_concat.zero(queue);
                cache.h_norm_ff.zero(queue);
                cache.ffn_act.zero(queue);
            }
        }

        size_t storage_size_bytes() const
        {
            size_t bytes = hidden_error.storage_size_bytes() + ff_error.storage_size_bytes();
            for (const auto& cache : activation_caches)
                bytes += cache.storage_size_bytes();
            return bytes;
        }
    };

    struct GradientAccumulationWorkspace
    {
        fixed_size_obj_vector<OutputLayerGradientAccumulator, MultiTokenPredictionIndex> output_layers;
        std::vector<TransformerGradientAccumulator> transformer_blocks;
        EmbeddingGradientAccumulator embeddings;
        std::unique_ptr<backprop::MuonWorkspace> muon;

        explicit GradientAccumulationWorkspace(size_t num_transformer_blocks, bool use_muon)
            : transformer_blocks(num_transformer_blocks)
        {
            output_layers.set_size(MultiTokenPredictionIndex::MAX);
            if (use_muon)
                muon = std::make_unique<backprop::MuonWorkspace>();
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
