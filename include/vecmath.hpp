#pragma once

#include <LayerPrimitives.hpp>

namespace rllm
{
    void sgd_update_Wqkvo_x_Vqkvo_dWqkvo(
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& vel,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& grad,
        float lr
    );

    void sgd_update_Wgateup_x_Vgateup_dWgateup(
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension>& W,
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& vel,
        const fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension>& grad,
        float lr
    );

    void sgd_update_Wdown_x_Vdown_dWdown(
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension>& W,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& vel,
        const fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension>& grad,
        float lr
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

    void matmul_AtB_acc(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension>& C,
        PositionIndex k_count
    );

    void attention_scores_for_head(
        flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& scores_mat,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& Q,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& K,
        int h_start,
        int h_end,
        float head_scale,
        PositionIndex seq_len
    );

    void attention_values_for_head(
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& attn_concat,
        const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& scores_mat,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& V,
        int h_start,
        int h_end,
        PositionIndex seq_len
    );

    void element_wise_sum(
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& lhs,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& rhs,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dst
    );

    void swiglu_forward(
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& ffn_act,
        PositionIndex seq_len
    );
}