#include <backprop/Muon.hpp>

#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>
#include <rllm_vulkan_runtime.hpp>

namespace rllm::backprop
{
    namespace
    {
        constexpr int NS_STEPS = 5;

        void initialize_square(
            // OFFLOAD_PARAMETERS(momentum, update, norm, diagnostics)
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& momentum,
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
            fixed_size_vector<float, TempStorage>& norm,
            const fixed_size_vector<float, TempStorage>& diagnostics
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            norm.zero(queue);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (momentum, update, norm, diagnostics))
            const float grad = (update[row, col] * diagnostics[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE]);
            const float next_momentum = ((0.95f * momentum[row, col]) + (0.05f * grad));
            momentum[row, col] = next_momentum;
            const float value = ((0.05f * grad) + (0.95f * next_momentum));
            update[row, col] = value;
            atomicAdd(norm[TempStorage::START], (value * value));
            ENDFOR
        }

        void initialize_tall(
            // OFFLOAD_PARAMETERS(momentum, update, norm, diagnostics)
            fixed_size_matrix<float, FFDimension, EmbeddingDimension>& momentum,
            fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
            fixed_size_vector<float, TempStorage>& norm,
            const fixed_size_vector<float, TempStorage>& diagnostics
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            norm.zero(queue);
            const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (momentum, update, norm, diagnostics))
            const float grad = (update[row, col] * diagnostics[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE]);
            const float next_momentum = ((0.95f * momentum[row, col]) + (0.05f * grad));
            momentum[row, col] = next_momentum;
            const float value = ((0.05f * grad) + (0.95f * next_momentum));
            update[row, col] = value;
            atomicAdd(norm[TempStorage::START], (value * value));
            ENDFOR
        }

        void initialize_wide(
            // OFFLOAD_PARAMETERS(momentum, update, norm, diagnostics)
            fixed_size_matrix<float, EmbeddingDimension, FFDimension>& momentum,
            fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
            fixed_size_vector<float, TempStorage>& norm,
            const fixed_size_vector<float, TempStorage>& diagnostics
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            norm.zero(queue);
            const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (momentum, update, norm, diagnostics))
            const float grad = (update[row, col] * diagnostics[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE]);
            const float next_momentum = ((0.95f * momentum[row, col]) + (0.05f * grad));
            momentum[row, col] = next_momentum;
            const float value = ((0.05f * grad) + (0.95f * next_momentum));
            update[row, col] = value;
            atomicAdd(norm[TempStorage::START], (value * value));
            ENDFOR
        }

        void normalize_square(
            // OFFLOAD_PARAMETERS(update, norm)
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
            const fixed_size_vector<float, TempStorage>& norm
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, norm))
            update[row, col] /= max(sqrt(norm[TempStorage::START]), 1e-7f);
            ENDFOR
        }

        void normalize_tall(
            // OFFLOAD_PARAMETERS(update, norm)
            fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
            const fixed_size_vector<float, TempStorage>& norm
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, norm))
            update[row, col] /= max(sqrt(norm[TempStorage::START]), 1e-7f);
            ENDFOR
        }

        void normalize_wide(
            // OFFLOAD_PARAMETERS(update, norm)
            fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
            const fixed_size_vector<float, TempStorage>& norm
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, norm))
            update[row, col] /= max(sqrt(norm[TempStorage::START]), 1e-7f);
            ENDFOR
        }

        void gram_rows_square(
            // OFFLOAD_PARAMETERS(update, gram_matrix)
            const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gram_matrix
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, gram_matrix))
            float value = 0.0f;
            for (int inner = 0; inner < static_cast<int>(EmbeddingDimension::MAX); ++inner)
                value += (update[row, inner] * update[col, inner]);
            gram_matrix[row, col] = value;
            ENDFOR
        }

        void gram_rows_wide(
            // OFFLOAD_PARAMETERS(update, gram_matrix)
            const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gram_matrix
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, gram_matrix))
            float value = 0.0f;
            for (int inner = 0; inner < static_cast<int>(FFDimension::MAX); ++inner)
                value += (update[row, inner] * update[col, inner]);
            gram_matrix[row, col] = value;
            ENDFOR
        }

        void gram_cols_tall(
            // OFFLOAD_PARAMETERS(update, gram_matrix)
            const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gram_matrix
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, gram_matrix))
            float value = 0.0f;
            for (int inner = 0; inner < static_cast<int>(FFDimension::MAX); ++inner)
                value += (update[inner, row] * update[inner, col]);
            gram_matrix[row, col] = value;
            ENDFOR
        }

        void polynomial(
            // OFFLOAD_PARAMETERS(gram, result)
            const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gram,
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& result
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (gram, result))
            float square = 0.0f;
            for (int inner = 0; inner < static_cast<int>(EmbeddingDimension::MAX); ++inner)
                square += (gram[row, inner] * gram[inner, col]);
            result[row, col] = ((-4.7750f * gram[row, col]) + (2.0315f * square));
            ENDFOR
        }

        void iterate_square(
            // OFFLOAD_PARAMETERS(update, polynomial_value, next)
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
            const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& polynomial_value,
            fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& next
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, polynomial_value, next))
            float product = 0.0f;
            for (int inner = 0; inner < static_cast<int>(EmbeddingDimension::MAX); ++inner)
                product += (polynomial_value[row, inner] * update[inner, col]);
            next[row, col] = ((3.4445f * update[row, col]) + product);
            ENDFOR
            update.device_buffer().copy_from(queue, next.device_buffer(), update.storage_size_bytes());
        }

        void iterate_wide(
            // OFFLOAD_PARAMETERS(update, polynomial_value, next)
            fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
            const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& polynomial_value,
            fixed_size_matrix<float, EmbeddingDimension, FFDimension>& next
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, polynomial_value, next))
            float product = 0.0f;
            for (int inner = 0; inner < static_cast<int>(EmbeddingDimension::MAX); ++inner)
                product += (polynomial_value[row, inner] * update[inner, col]);
            next[row, col] = ((3.4445f * update[row, col]) + product);
            ENDFOR
            update.device_buffer().copy_from(queue, next.device_buffer(), update.storage_size_bytes());
        }

        void iterate_tall(
            // OFFLOAD_PARAMETERS(update, polynomial_value, next)
            fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
            const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& polynomial_value,
            fixed_size_matrix<float, FFDimension, EmbeddingDimension>& next
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (update, polynomial_value, next))
            float product = 0.0f;
            for (int inner = 0; inner < static_cast<int>(EmbeddingDimension::MAX); ++inner)
                product += (update[row, inner] * polynomial_value[inner, col]);
            next[row, col] = ((3.4445f * update[row, col]) + product);
            ENDFOR
            update.device_buffer().copy_from(queue, next.device_buffer(), update.storage_size_bytes());
        }

        void apply_square(
            // OFFLOAD_PARAMETERS(weight, update, learning_rate)
            fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& weight,
            const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
            float learning_rate
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (weight, update, learning_rate))
            const float old_weight = weight[row, col];
            weight[row, col] = (((1.0f - (learning_rate * 0.01f)) * old_weight) -
                (learning_rate * update[row, col]));
            ENDFOR
        }

        void apply_tall(
            // OFFLOAD_PARAMETERS(weight, update, learning_rate)
            fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& weight,
            const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
            float learning_rate
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (weight, update, learning_rate))
            const float old_weight = weight[row, col];
            weight[row, col] = (((1.0f - (learning_rate * 0.01f)) * old_weight) -
                ((learning_rate * 2.0f) * update[row, col]));
            ENDFOR
        }

        void apply_wide(
            // OFFLOAD_PARAMETERS(weight, update, learning_rate)
            fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& weight,
            const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
            float learning_rate
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
            OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (weight, update, learning_rate))
            const float old_weight = weight[row, col];
            weight[row, col] = (((1.0f - (learning_rate * 0.01f)) * old_weight) -
                (learning_rate * update[row, col]));
            ENDFOR
        }
    }

    void muon_update(fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& weight,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& momentum,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
        MuonWorkspace& workspace, float learning_rate,
        fixed_size_vector<float, TempStorage>& diagnostics)
    {
        initialize_square(momentum, update, workspace.norm, diagnostics);
        normalize_square(update, workspace.norm);
        for (int step = 0; step < NS_STEPS; ++step)
        {
            gram_rows_square(update, workspace.gram);
            polynomial(workspace.gram, workspace.gram_update);
            iterate_square(update, workspace.gram_update, workspace.next_square);
        }
        apply_square(weight, update, learning_rate);
    }

    void muon_update(fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& weight,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& momentum,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
        MuonWorkspace& workspace, float learning_rate,
        fixed_size_vector<float, TempStorage>& diagnostics)
    {
        initialize_tall(momentum, update, workspace.norm, diagnostics);
        normalize_tall(update, workspace.norm);
        for (int step = 0; step < NS_STEPS; ++step)
        {
            gram_cols_tall(update, workspace.gram);
            polynomial(workspace.gram, workspace.gram_update);
            iterate_tall(update, workspace.gram_update, workspace.next_tall);
        }
        apply_tall(weight, update, learning_rate);
    }

    void muon_update(fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& weight,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& momentum,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
        MuonWorkspace& workspace, float learning_rate,
        fixed_size_vector<float, TempStorage>& diagnostics)
    {
        initialize_wide(momentum, update, workspace.norm, diagnostics);
        normalize_wide(update, workspace.norm);
        for (int step = 0; step < NS_STEPS; ++step)
        {
            gram_rows_wide(update, workspace.gram);
            polynomial(workspace.gram, workspace.gram_update);
            iterate_wide(update, workspace.gram_update, workspace.next_wide);
        }
        apply_wide(weight, update, learning_rate);
    }
}
