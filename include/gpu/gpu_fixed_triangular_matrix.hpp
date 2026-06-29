#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <RandomHelpers.hpp>
#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>
#include <rllm_vulkan_runtime.hpp>
#include <cpu/cpu_fixed_triangular_matrix.hpp>

#ifndef PREFER_SPEED_OVER_MEMORY
#define PREFER_SPEED_OVER_MEMORY 0
#endif

#if PREFER_SPEED_OVER_MEMORY

#include <gpu/gpu_fixed_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    using gpu_fixed_triangular_matrix = gpu_fixed_matrix<ElementType, X, Y>;
} // namespace rllm

#else

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    class gpu_fixed_triangular_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);
        static_assert(ROWS == COLS, "gpu_fixed_triangular_matrix requires a square matrix");
        static constexpr size_t NUM_ELTS = ROWS * (ROWS + 1) / 2;

        gpu_fixed_triangular_matrix()
            : offloadable_data<ElementType>(NUM_ELTS)
        {}

        gpu_fixed_triangular_matrix(const gpu_fixed_triangular_matrix&) = delete;
        gpu_fixed_triangular_matrix& operator=(const gpu_fixed_triangular_matrix&) = delete;
        gpu_fixed_triangular_matrix(gpu_fixed_triangular_matrix&&) = default;
        gpu_fixed_triangular_matrix& operator=(gpu_fixed_triangular_matrix&&) = default;

        ~gpu_fixed_triangular_matrix() = default;
        /** H2D: direct Vulkan copy from cpu_fixed_triangular_matrix's pinned buffer to device. */
        void copy_from_cpu(VulkanQueue& queue, const cpu_fixed_triangular_matrix<ElementType, X, Y>& src)
        {
            this->m_data.copy_to_offload_buffer(queue, const_cast<VBaseHostBuffer&>(src.vk_host_buffer()));
        }

        /** D2H: direct Vulkan copy from device into cpu_fixed_triangular_matrix's pinned buffer. */
        void copy_to_cpu(VulkanQueue& queue, cpu_fixed_triangular_matrix<ElementType, X, Y>& dst) const
        {
            this->m_data.copy_from_offload_buffer(queue, dst.vk_host_buffer());
        }
        void fill_rand(VulkanQueue& queue, ElementType lo, ElementType hi)
        {
            cpu_fixed_triangular_matrix<ElementType, X, Y> tmp;
            auto* ptr = tmp.data();
            for (size_t i = 0; i < NUM_ELTS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
            copy_from_cpu(queue, tmp);
        }

        size_t storage_size_bytes() const { return NUM_ELTS * sizeof(ElementType); }

      private:
        size_t k(const X x, const Y y) const
        {
            const size_t sx = static_cast<size_t>(x), sy = static_cast<size_t>(y);
            assert(sx < ROWS); assert(sy < COLS); assert(sy <= sx);
            const size_t ki = sx * (sx + 1) / 2 + sy;
            assert(ki < NUM_ELTS);
            return ki;
        }
    };
} // namespace rllm

#endif
