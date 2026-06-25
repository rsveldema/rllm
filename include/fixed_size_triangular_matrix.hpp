#pragma once
// Compatibility wrapper — include the gpu_* and cpu_* files directly for new code.
#include <gpu/gpu_fixed_triangular_matrix.hpp>
#include <cpu/cpu_fixed_triangular_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    using fixed_size_triangular_matrix = gpu_fixed_triangular_matrix<ElementType, X, Y>;
} // namespace rllm
