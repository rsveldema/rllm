#pragma once
// Compatibility wrapper — include the gpu_* and cpu_* files directly for new code.
#include <gpu/gpu_fixed_matrix.hpp>
#include <cpu/cpu_fixed_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    using fixed_size_matrix = gpu_fixed_matrix<ElementType, X, Y>;
} // namespace rllm
