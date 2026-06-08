#include <TransformerBlock.hpp>
#include <enum_iterator.hpp>
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
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator<PositionIndex>(length), (dst, value))
        dst[static_cast<size_t>(i)] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value, length)
        fixed_size_vector<rlmm_float, PositionIndex>& dst,
        rlmm_float value,
        PositionIndex length
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(i, enum_iterator<PositionIndex>(length), (dst, value))
        dst[static_cast<size_t>(i)] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value)
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst,
        rlmm_float value
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(dst.num_rows());
        OFFLOAD_PARFOR_2D_PARAM(r, c, grid, (dst, value))
        dst[r, c] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value)
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dst,
        rlmm_float value
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, PositionIndex>(dst.num_rows(), dst.num_cols());
        OFFLOAD_PARFOR_2D_PARAM(r, c, grid, (dst, value))
        dst[r, c] = value;
        ENDFOR
    }

    void copy_hidden_row_to_vector(
        // OFFLOAD_PARAMETERS(src, row, dst)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& src,
        PositionIndex row,
        fixed_size_vector<rlmm_float, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(d, enum_iterator<EmbeddingDimension>(), (src, row, dst))
        dst[d] = src[row, d];
        ENDFOR
    }

    void sgd_update_Wqkvo_x_Vqkvo_dWqkvo__4_matrix(
        // OFFLOAD_PARAMETERS(W1, vel1, grad1, W2, vel2, grad2, W3, vel3, grad3, W4, vel4, grad4, lr)
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W1,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel1,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad1,
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W2,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel2,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad2,
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W3,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel3,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad3,
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W4,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel4,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad4,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(r, c, grid, (W1, vel1, grad1, W2, vel2, grad2, W3, vel3, grad3, W4, vel4, grad4, lr))
        const float g1 = math::clamp(grad1[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel1[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel1[r, c] + lr * g1,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W1[r, c] = math::clamp(W1[r, c] + vel1[r, c], -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float g2 = math::clamp(grad2[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel2[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel2[r, c] + lr * g2,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W2[r, c] = math::clamp(W2[r, c] + vel2[r, c], -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float g3 = math::clamp(grad3[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel3[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel3[r, c] + lr * g3,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W3[r, c] = math::clamp(W3[r, c] + vel3[r, c], -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float g4 = math::clamp(grad4[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel4[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel4[r, c] + lr * g4,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W4[r, c] = math::clamp(W4[r, c] + vel4[r, c], -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void sgd_update_Wgateup_x_Vgateup_dWgateup__2_matrix(
        // OFFLOAD_PARAMETERS(W1, vel1, grad1, W2, vel2, grad2, lr)
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& W1,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& vel1,
        const fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& grad1,
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& W2,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& vel2,
        const fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& grad2,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(r, c, grid, (W1, vel1, grad1, W2, vel2, grad2, lr))
        const float g1 = math::clamp(grad1[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel1[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel1[r, c] + lr * g1,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W1[r, c] = math::clamp(W1[r, c] + vel1[r, c], -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);

        const float g2 = math::clamp(grad2[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel2[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel2[r, c] + lr * g2,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W2[r, c] = math::clamp(W2[r, c] + vel2[r, c], -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void sgd_update_Wdown_x_Vdown_dWdown(
        // OFFLOAD_PARAMETERS(W, vel, grad, lr)
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& W,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& vel,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& grad,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(r, c, grid, (W, vel, grad, lr))
        const float g = math::clamp(grad[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        vel[r, c] = math::clamp(
            TransformerBlock::MOMENTUM_BETA * vel[r, c] + lr * g,
            -TransformerBlock::VEL_CLIP,
            TransformerBlock::VEL_CLIP
        );
        W[r, c] = math::clamp(
            W[r, c] + vel[r, c],
            -TransformerBlock::WEIGHT_CLAMP,
            TransformerBlock::WEIGHT_CLAMP
        );
        ENDFOR
    }

    /**
     * @brief C1 = A × B1^T, C2 = A × B2^T, and C3 = A × B3^T for Q/K/V-style projections.
     */
    void matmul_ABt_3_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2,B3,C3)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C1,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C2,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B3,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C3
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B1, C1, B2, C2, B3, C3))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float a = A[i, k];
                const float term1 = a * B1[j, k];
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = a * B2[j, k];
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = a * B3[j, k];
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C1[i, j] = static_cast<rlmm_float>(sum1);
            C2[i, j] = static_cast<rlmm_float>(sum2);
            C3[i, j] = static_cast<rlmm_float>(sum3);
        ENDFOR
    }

    /**
     * @brief C1 = A × B1^T and C2 = A × B2^T, where A is [m × k] and B1, B2 are [n × k].
     */
    void matmul_ABt_2_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C1,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C2
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B1, C1, B2, C2))
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float a = A[i, k];
                const float term1 = a * B1[j, k];
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = a * B2[j, k];
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;
            }
            C1[i, j] = static_cast<rlmm_float>(sum1);
            C2[i, j] = static_cast<rlmm_float>(sum2);
        ENDFOR
    }

    void matmul_ABt(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[i, k] * B[j, k];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_ABt(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(FFDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[i, k] * B[j, k];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_AB(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[i, k] * B[k, j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_AB(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[i, k] * B[k, j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_AB_add(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(FFDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[i, k] * B[k, j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_AB_add_3_matrix_muls(
        // OFFLOAD_PARAMETERS(A1, B1, A2, B2, A3, B3, C)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A1,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B1,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A2,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B2,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A3,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B3,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        const PositionIndex m = A1.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A1, B1, A2, B2, A3, B3, C))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);

                const float term1 = A1[i, k] * B1[k, j];
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = A2[i, k] * B2[k, j];
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = A3[i, k] * B3[k, j];
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C[i, j] += static_cast<rlmm_float>(sum1 + sum2 + sum3);
        ENDFOR
    }

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& B,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C, k_count))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[k, i] * B[k, j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_AtB_acc_2_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, B, C1, C2, k_count)
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A1,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A2,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& C2,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A1, A2, B, C1, C2, k_count))
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float b = B[k, j];
                const float term1 = A1[k, i] * b;
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = A2[k, i] * b;
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;
            }
            C1[i, j] += static_cast<rlmm_float>(sum1);
            C2[i, j] += static_cast<rlmm_float>(sum2);
        ENDFOR
    }

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C, k_count))
            float sum = 0.f;
            RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = A[k, i] * B[k, j];
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<rlmm_float>(sum);
        ENDFOR
    }

    void matmul_AtB_acc_3_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, A3, B, C1, C2, C3, k_count)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A1,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A2,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A3,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& C2,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& C3,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A1, A2, A3, B, C1, C2, C3, k_count))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float b = B[k, j];
                const float term1 = A1[k, i] * b;
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = A2[k, i] * b;
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = A3[k, i] * b;
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C1[i, j] += static_cast<rlmm_float>(sum1);
            C2[i, j] += static_cast<rlmm_float>(sum2);
            C3[i, j] += static_cast<rlmm_float>(sum3);
        ENDFOR
    }

    void element_wise_sum(
        // OFFLOAD_PARAMETERS(lhs, rhs, dst)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& lhs,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& rhs,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(lhs.num_rows());
        OFFLOAD_PARFOR_2D_PARAM(t, d, grid, (lhs, rhs, dst))
        dst[t, d] = lhs[t, d] + rhs[t, d];
        ENDFOR
    }

    void swiglu_forward(
        // OFFLOAD_PARAMETERS(gate_pre, up_pre, ffn_act)
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& ffn_act,
        PositionIndex seq_len
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(seq_len);
        OFFLOAD_PARFOR_2D_PARAM(t, f, grid, (gate_pre, up_pre, ffn_act))
        const float g = gate_pre[t, f];
        const float silu = g / (1.0f + std::exp(-g));
        ffn_act[t, f] = silu * up_pre[t, f];
        ENDFOR
    }
}
