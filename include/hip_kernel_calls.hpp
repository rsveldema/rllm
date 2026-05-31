#pragma once

#include <utility>

namespace rllm::hip
{
    template <typename Range, typename KernelFn>
    inline void launch_kernel_1d([[maybe_unused]] const char* kernel_id, Range&& range, KernelFn&& fn)
    {
        // Placeholder dispatch path until real HIP kernels are wired in.
        for (auto idx : range)
            fn(idx);
    }

    template <typename Range2D, typename KernelFn>
    inline void launch_kernel_2d([[maybe_unused]] const char* kernel_id, Range2D&& range, KernelFn&& fn)
    {
        // Placeholder dispatch path until real HIP kernels are wired in.
        for (const auto [i, j] : range)
            fn(i, j);
    }
} // namespace rllm::hip
