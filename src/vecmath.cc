#include <TransformerBlock.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>
#include <vecmath.hpp>


namespace parallel
{

#if defined(RLLM_ENABLE_STATISTICS)
    thread_local DispatchParams g_vulkan_dispatch_params{};

    void set_vulkan_dispatch_params(std::string_view site, std::string_view param)
    {
        g_vulkan_dispatch_params.site = site;
        g_vulkan_dispatch_params.parameter = param;
    }


    DispatchParams get_dispatch_params() { return g_vulkan_dispatch_params; }

    void clear_vulkan_dispatch_params()
    {
        g_vulkan_dispatch_params = DispatchParams{};
    }
#endif
}

namespace rllm
{
    void fill(
        // OFFLOAD_PARAMETERS(dst, value, length)
        fixed_size_vector<int, PositionIndex>& dst,
        int value,
        PositionIndex length
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(length), (dst, value))
        dst[static_cast<size_t>(i)] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value, length)
        fixed_size_vector<float, PositionIndex>& dst,
        float value,
        PositionIndex length
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(length), (dst, value))
        dst[static_cast<size_t>(i)] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dst,
        float value
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(dst.num_rows());
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (dst, value))
        dst[r, c] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value)
        flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dst,
        float value
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, PositionIndex>(dst.num_rows(), dst.num_cols());
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (dst, value))
        dst[r, c] = value;
        ENDFOR
    }

    void copy_hidden_row_to_vector(
        // OFFLOAD_PARAMETERS(src, row, dst)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& src,
        PositionIndex row,
        fixed_size_vector<float, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, d, enum_iterator1D<EmbeddingDimension>(), (src, row, dst))
        dst[d] = src[row, d];
        ENDFOR
    }

    void sgd_update_Wqkvo_x_Vqkvo_dWqkvo__4_matrix(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(W1, vel1, grad1, W2, vel2, grad2, W3, vel3, grad3, W4, vel4, grad4, lr)
        fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& W1,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& vel1,
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& grad1,
        fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& W2,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& vel2,
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& grad2,
        fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& W3,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& vel3,
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& grad3,
        fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& W4,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& vel4,
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& grad4,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (W1, vel1, grad1, W2, vel2, grad2, W3, vel3, grad3, W4, vel4, grad4, lr))
        const float raw_g1 = grad1[r, c];
        const float g1 = math::clamp(raw_g1, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v1 = ((TransformerBlock::MOMENTUM_BETA * vel1[r, c]) + (lr * g1));
        vel1[r, c] = math::clamp(raw_v1, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w1 = (W1[r, c] + vel1[r, c]);
        W1[r, c] = math::clamp(raw_w1, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float raw_g2 = grad2[r, c];
        const float g2 = math::clamp(raw_g2, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v2 = ((TransformerBlock::MOMENTUM_BETA * vel2[r, c]) + (lr * g2));
        vel2[r, c] = math::clamp(raw_v2, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w2 = (W2[r, c] + vel2[r, c]);
        W2[r, c] = math::clamp(raw_w2, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float raw_g3 = grad3[r, c];
        const float g3 = math::clamp(raw_g3, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v3 = ((TransformerBlock::MOMENTUM_BETA * vel3[r, c]) + (lr * g3));
        vel3[r, c] = math::clamp(raw_v3, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w3 = (W3[r, c] + vel3[r, c]);
        W3[r, c] = math::clamp(raw_w3, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float raw_g4 = grad4[r, c];
        const float g4 = math::clamp(raw_g4, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v4 = ((TransformerBlock::MOMENTUM_BETA * vel4[r, c]) + (lr * g4));
        vel4[r, c] = math::clamp(raw_v4, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w4 = (W4[r, c] + vel4[r, c]);
        W4[r, c] = math::clamp(raw_w4, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void sgd_update_Wgateup_x_Vgateup_dWgateup__2_matrix(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(W1, vel1, grad1, W2, vel2, grad2, lr)
        fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& W1,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& vel1,
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& grad1,
        fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& W2,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& vel2,
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& grad2,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (W1, vel1, grad1, W2, vel2, grad2, lr))
        const float raw_g1 = grad1[r, c];
        const float g1 = math::clamp(raw_g1, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v1 = ((TransformerBlock::MOMENTUM_BETA * vel1[r, c]) + (lr * g1));
        vel1[r, c] = math::clamp(raw_v1, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w1 = (W1[r, c] + vel1[r, c]);
        W1[r, c] = math::clamp(raw_w1, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float raw_g2 = grad2[r, c];
        const float g2 = math::clamp(raw_g2, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v2 = ((TransformerBlock::MOMENTUM_BETA * vel2[r, c]) + (lr * g2));
        vel2[r, c] = math::clamp(raw_v2, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w2 = (W2[r, c] + vel2[r, c]);
        W2[r, c] = math::clamp(raw_w2, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void sgd_update_Wdown_x_Vdown_dWdown(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(W, vel, grad, lr)
        fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& W,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& vel,
        const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& grad,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (W, vel, grad, lr))
        const float raw_g = grad[r, c];
        const float g = math::clamp(raw_g, -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        const float raw_v = ((TransformerBlock::MOMENTUM_BETA * vel[r, c]) + (lr * g));
        vel[r, c] = math::clamp(raw_v, -TransformerBlock::VEL_CLIP, TransformerBlock::VEL_CLIP);
        const float raw_w = (W[r, c] + vel[r, c]);
        W[r, c] = math::clamp(raw_w, -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    /**
     * @brief C1 = A × B1^T, C2 = A × B2^T, and C3 = A × B3^T for Q/K/V-style projections.
     */
    void matmul_ABt_3_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2,B3,C3)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C1,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C2,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B3,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C3
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B1, C1, B2, C2, B3, C3))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float a = A[i, k];
                const float term1 = (a * B1[j, k]);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (a * B2[j, k]);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = (a * B3[j, k]);
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C1[i, j] = static_cast<float>(sum1);
            C2[i, j] = static_cast<float>(sum2);
            C3[i, j] = static_cast<float>(sum3);
        ENDFOR
    }

    /**
     * @brief C1 = A × B1^T and C2 = A × B2^T, where A is [m × k] and B1, B2 are [n × k].
     */
    void matmul_ABt_2_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C1,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C2
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B1, C1, B2, C2))
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float a = A[i, k];
                const float term1 = (a * B1[j, k]);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (a * B2[j, k]);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;
            }
            C1[i, j] = static_cast<float>(sum1);
            C2[i, j] = static_cast<float>(sum2);
        ENDFOR
    }

    void matmul_ABt(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[j, k]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_ABt(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(FFDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[j, k]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB_add(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(FFDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            const float raw = (C[i, j] + static_cast<float>(sum));
            C[i, j] = math::clamp(raw, -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB_add_3_matrix_muls(
        // OFFLOAD_PARAMETERS(A1, B1, A2, B2, A3, B3, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A1,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B1,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A2,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A3,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B3,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A1.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A1, B1, A2, B2, A3, B3, C))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);

                const float term1 = (A1[i, k] * B1[k, j]);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (A2[i, k] * B2[k, j]);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = (A3[i, k] * B3[k, j]);
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            const float raw = (C[i, j] + static_cast<float>((sum1 + (sum2 + sum3))));
            C[i, j] = math::clamp(raw, -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C, k_count))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[k, i] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<float>(sum);
        ENDFOR
    }

    void matmul_AtB_acc_2_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, B, C1, C2, k_count)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A1,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& C2,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A1, A2, B, C1, C2, k_count))
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float b = B[k, j];
                const float term1 = (A1[k, i] * b);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (A2[k, i] * b);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;
            }
            C1[i, j] += static_cast<float>(sum1);
            C2[i, j] += static_cast<float>(sum2);
        ENDFOR
    }

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C, k_count))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[k, i] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<float>(sum);
        ENDFOR
    }

    void matmul_AtB_acc_3_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, A3, B, C1, C2, C3, k_count)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A1,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A3,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C2,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C3,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A1, A2, A3, B, C1, C2, C3, k_count))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float b = B[k, j];
                const float term1 = (A1[k, i] * b);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (A2[k, i] * b);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = (A3[k, i] * b);
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C1[i, j] += static_cast<float>(sum1);
            C2[i, j] += static_cast<float>(sum2);
            C3[i, j] += static_cast<float>(sum3);
        ENDFOR
    }

    void element_wise_sum(
        // OFFLOAD_PARAMETERS(lhs, rhs, dst)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& lhs,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& rhs,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(lhs.num_rows());
        OFFLOAD_PARFOR_2D_PARAM(queue, t, d, grid, (lhs, rhs, dst))
        dst[t, d] = math::clamp((lhs[t, d] + rhs[t, d]), -10000.0f, 10000.0f);
        ENDFOR
    }

    void swiglu_forward(
        // OFFLOAD_PARAMETERS(gate_pre, up_pre, ffn_act)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& up_pre,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& ffn_act,
        PositionIndex seq_len
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(seq_len);
        OFFLOAD_PARFOR_2D_PARAM(queue, t, f, grid, (gate_pre, up_pre, ffn_act))
        const float g = gate_pre[t, f];
        const float silu = (g / (1.0f + std::exp(-g)));
        ffn_act[t, f] = math::clamp((silu * up_pre[t, f]), -10000.0f, 10000.0f);
        ENDFOR
    }
}
