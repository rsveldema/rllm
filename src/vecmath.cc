#include <TransformerBlock.hpp>
#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>
#include <vecmath.hpp>

namespace rllm
{
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

    void sgd_update_Wqkvo_x_Vqkvo_dWqkvo(
        // OFFLOAD_PARAMETERS(W, vel, grad, lr)
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
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

    void sgd_update_Wgateup_x_Vgateup_dWgateup(
        // OFFLOAD_PARAMETERS(W, vel, grad, lr)
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& vel,
        const fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& grad,
        float lr
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
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
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B,
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

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
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

    void attention_scores_for_head(
        // OFFLOAD_PARAMETERS(scores_mat, Q, K, h_start, h_end, head_scale, seq_len)
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& scores_mat,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& Q,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& K,
        int h_start,
        int h_end,
        float head_scale,
        PositionIndex seq_len
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, PositionIndex>(seq_len, seq_len);
        OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (scores_mat, Q, K, h_start, h_end, head_scale, seq_len))
        float dot = 0.f;
        RLLM_OMP_SIMD_REDUCTION_PLUS(dot)
        for (int d = h_start; d < h_end; ++d)
            dot += Q[i, d] * K[j, d];
        scores_mat[i, j] = dot * head_scale;
        ENDFOR
    }

    void attention_values_for_head(
        // OFFLOAD_PARAMETERS(attn_concat, scores_mat, V, h_start, h_end, seq_len)
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& attn_concat,
        const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& scores_mat,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& V,
        int h_start,
        int h_end,
        PositionIndex seq_len
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, seq_len, (attn_concat, scores_mat, V, h_start, h_end))
        const float w = scores_mat[i, j];
        for (int d = h_start; d < h_end; ++d)
            attn_concat[i, d] += w * V[j, d];
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

    void element_wise_add(
        // OFFLOAD_PARAMETERS(src, dst)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& src,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(src.num_rows(), src.num_cols());
        OFFLOAD_PARFOR_2D_PARAM(t, d, grid, (src, dst))
        dst[t, d] += src[t, d];
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