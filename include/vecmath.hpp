#pragma once

#include <LayerPrimitives.hpp>

namespace rllm
{
    void fill(fixed_size_vector<int, PositionIndex>& dst, int value, PositionIndex length);
    void fill(fixed_size_vector<float, PositionIndex>& dst, float value, PositionIndex length);

    void fill(flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dst, float value);
    void fill(flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dst, float value);

    void copy_hidden_row_to_vector(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& src,
        PositionIndex row,
        fixed_size_vector<float, EmbeddingDimension>& dst
    );

    void adamw_update(fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& weight, fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& first, fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& second, const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient, float learning_rate, float bias_correction1, float bias_correction2);
    void adamw_update(fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& weight, fixed_size_matrix<float, FFDimension, EmbeddingDimension>& first, fixed_size_matrix<float, FFDimension, EmbeddingDimension>& second, const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient, float learning_rate, float bias_correction1, float bias_correction2);
    void adamw_update(fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& weight, fixed_size_matrix<float, EmbeddingDimension, FFDimension>& first, fixed_size_matrix<float, EmbeddingDimension, FFDimension>& second, const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient, float learning_rate, float bias_correction1, float bias_correction2);
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
    );

    void matmul_ABt_2_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C1,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C2
        // END_OFFLOAD_PARAMETERS
    );

    void matmul_ABt(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_ABt(
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_AB(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C
    );

    void matmul_AB(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
    );

    void matmul_AB_add(
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
    );

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
    );

    void matmul_AtB_acc(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& C,
        PositionIndex k_count
    );

    void matmul_AtB_acc_2_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, B, C1, C2, k_count)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A1,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& C2,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    );

    void matmul_AtB_acc(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C,
        PositionIndex k_count
    );

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
    );

    void element_wise_sum(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& lhs,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& rhs,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dst
    );

    void swiglu_forward(
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& up_pre,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& ffn_act,
        PositionIndex seq_len
    );
}
