#pragma once

#include <cstdlib>
#include <utility>

#include <logging.hpp>

namespace rllm::hip
{
    template <typename Range, typename KernelFn>
    inline void launch_kernel_1d(const char* kernel_id, [[maybe_unused]] Range&& range, [[maybe_unused]] KernelFn&& fn)
    {
        LOG_ERROR("HIP kernel '{}' has no backend dispatch implementation yet.", kernel_id);
        std::abort();
    }

    template <typename Range2D, typename KernelFn>
    inline void launch_kernel_2d(const char* kernel_id, [[maybe_unused]] Range2D&& range, [[maybe_unused]] KernelFn&& fn)
    {
        LOG_ERROR("HIP kernel '{}' has no backend dispatch implementation yet.", kernel_id);
        std::abort();
    }
} // namespace rllm::hip
