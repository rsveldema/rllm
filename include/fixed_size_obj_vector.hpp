#pragma once
// Compatibility wrapper — include cpu_fixed_obj_vector.hpp directly for new code.
#include <cpu/cpu_fixed_obj_vector.hpp>

namespace rllm
{
    template <typename T, typename LengthType>
    using fixed_size_obj_vector = cpu_fixed_obj_vector<T, LengthType>;
} // namespace rllm
