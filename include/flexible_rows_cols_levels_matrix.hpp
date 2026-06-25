#pragma once
// Compatibility wrapper — include the gpu_* and cpu_* files directly for new code.
#include <gpu/gpu_flex_rows_cols_levels_matrix.hpp>
#include <cpu/cpu_flex_rows_cols_levels_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename L, typename X, typename Y>
    using flexible_rows_cols_levels_matrix = gpu_flex_rows_cols_levels_matrix<ElementType, L, X, Y>;
} // namespace rllm
