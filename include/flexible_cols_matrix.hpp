#pragma once
// Compatibility wrapper — include the gpu_* and cpu_* files directly for new code.
#include <gpu/gpu_flex_cols_matrix.hpp>
#include <cpu/cpu_flex_cols_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    using flexible_cols_matrix = gpu_flex_cols_matrix<ElementType, X, Y>;
} // namespace rllm
