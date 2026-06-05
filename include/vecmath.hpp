#pragma once

#include <LayerPrimitives.hpp>

namespace rllm
{
    void fill(fixed_size_vector<int, PositionIndex>& dst, int value, PositionIndex length);
    void fill(fixed_size_vector<rlmm_float, PositionIndex>& dst, rlmm_float value, PositionIndex length);

    void fill(flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst, rlmm_float value);
    void fill(flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dst, rlmm_float value);

    void copy_hidden_row_to_vector(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& src,
        PositionIndex row,
        fixed_size_vector<rlmm_float, EmbeddingDimension>& dst
    );

    void sgd_update_Wqkvo_x_Vqkvo_dWqkvo(
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad,
        float lr
    );

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
    );

    void sgd_update_Wgateup_x_Vgateup_dWgateup(
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& vel,
        const fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& grad,
        float lr
    );

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
    );

    void sgd_update_Wdown_x_Vdown_dWdown(
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& W,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& vel,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& grad,
        float lr
    );
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
    );

    void matmul_ABt_2_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2)
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C1,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C2
        // END_OFFLOAD_PARAMETERS
    );

    void matmul_ABt(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_ABt(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C
    );

    void matmul_ABt(
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_AB(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& C
    );

    void matmul_AB(
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_AB(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_AB_add(
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_AB_add(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& C
    );

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
    );

    void matmul_AtB_acc(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& B,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& C,
        PositionIndex k_count
    );

    void matmul_AtB_acc(
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& C,
        PositionIndex k_count
    );

    void matmul_AtB_acc_2_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, B, C1, C2, k_count)
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A1,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& A2,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& C2,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    );

    void matmul_AtB_acc(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& C,
        PositionIndex k_count
    );

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
    );

    void element_wise_sum(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& lhs,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& rhs,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst
    );

    void element_wise_add(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& src,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst
    );

    void swiglu_forward(
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& ffn_act,
        PositionIndex seq_len
    );
}
