#pragma once
// Compatibility wrapper — include the gpu_* and cpu_* files directly for new code.
#include <gpu/gpu_flex_size_matrix.hpp>
#include <cpu/cpu_flex_size_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    using flexible_size_matrix = gpu_flex_size_matrix<ElementType, X, Y>;
} // namespace rllm
