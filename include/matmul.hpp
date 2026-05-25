#pragma once

#include <cstddef>

#include <enum_iterator.hpp>
#include <enum_iterator2D.hpp>
#include <fixed_size_matrix.hpp>
#include <flexible_rows_matrix.hpp>
#include <LayerPrimitives.hpp>

namespace rllm
{
    // C[m,n]  = A[m,k] @ B[n,k]^T   (B stored row-major [n × k])
    // m comes from A.num_rows() at runtime.
    template<typename K_enum, typename N_enum>
    static void matmul_ABt(
        const flexible_rows_matrix<float, PositionIndex, K_enum>& A,
        const fixed_size_matrix<float, N_enum, K_enum>& B,
        flexible_rows_matrix<float, PositionIndex, N_enum>& C)
    {
        const PositionIndex m = A.num_rows();
#pragma omp parallel for schedule(static)
        for (const auto [i, j] : enum_iterator2D<PositionIndex, N_enum>(m))
        {
            float sum = 0.f;
#pragma omp simd reduction(+:sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(K_enum::MAX); ++l_idx)
                sum += A[i, static_cast<K_enum>(l_idx)] * B[j, static_cast<K_enum>(l_idx)];
            C[i, j] = sum;
        }
    }

    // C[m,n]  = A[m,k] @ B[k,n]     (B stored row-major [k × n])
    // m comes from A.num_rows() at runtime.
    template<typename K_enum, typename N_enum>
    static void matmul_AB(
        const flexible_rows_matrix<float, PositionIndex, K_enum>& A,
        const fixed_size_matrix<float, K_enum, N_enum>& B,
        flexible_rows_matrix<float, PositionIndex, N_enum>& C)
    {
        const PositionIndex m = A.num_rows();
#pragma omp parallel for schedule(static)
        for (const auto [i, j] : enum_iterator2D<PositionIndex, N_enum>(m))
        {
            float sum = 0.f;
#pragma omp simd reduction(+:sum)
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(K_enum::MAX); ++l_idx)
                sum += A[i, static_cast<K_enum>(l_idx)] * B[static_cast<K_enum>(l_idx), j];
            C[i, j] = sum;
        }
    }

    // C[m,n] += A^T[m,k] @ B[k,n]   (A provided row-major [k × m]; accumulates into C)
    // k comes from A.num_rows() at runtime.
    template<typename M_enum, typename N_enum>
    static void matmul_AtB_acc(
        const flexible_rows_matrix<float, PositionIndex, M_enum>& A,
        const flexible_rows_matrix<float, PositionIndex, N_enum>& B,
        fixed_size_matrix<float, M_enum, N_enum>& C)
    {
        const PositionIndex k = A.num_rows();
#pragma omp parallel for schedule(static)
        for (const auto [i, j] : enum_iterator2D<M_enum, N_enum>())
        {
            float sum = 0.f;
            const size_t k_count = static_cast<size_t>(k);
#pragma omp simd reduction(+:sum)
            for (size_t l_idx = 0; l_idx < k_count; ++l_idx)
                sum += A[static_cast<PositionIndex>(l_idx), i] * B[static_cast<PositionIndex>(l_idx), j];
            C[i, j] += sum;
        }
    }

} // namespace rllm
