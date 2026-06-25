#pragma once
// Compatibility wrapper — include the gpu_* and cpu_* files directly for new code.
#include <gpu/gpu_fixed_vector.hpp>
#include <cpu/cpu_fixed_vector.hpp>

namespace rllm
{
    template <typename T, typename LengthType>
    using fixed_size_vector = gpu_fixed_vector<T, LengthType>;
} // namespace rllm
