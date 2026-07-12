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
#include <cpu/cpu_fixed_matrix.hpp>

namespace rllm
{
    /** Fixed-size 2-D matrix backed by GPU-offloadable memory.
     *  The X and Y template parameters are enum types used for type-safe indexing.
     */
    template <typename ElementType, typename X, typename Y>
    class gpu_fixed_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_fixed_matrix()
            : offloadable_data<ElementType>(ROWS * COLS)
        {}

        gpu_fixed_matrix(const gpu_fixed_matrix&) = delete;
        gpu_fixed_matrix& operator=(const gpu_fixed_matrix&) = delete;
        gpu_fixed_matrix(gpu_fixed_matrix&&) = default;
        gpu_fixed_matrix& operator=(gpu_fixed_matrix&&) = default;

        ~gpu_fixed_matrix() = default;

        /** H2D: direct Vulkan copy from cpu_fixed_matrix's pinned host buffer to device. */
        void copy_from_cpu(VulkanQueue& queue, const cpu_fixed_matrix<ElementType, X, Y>& src)
        {
            this->m_data.copy_to_offload_buffer(queue, const_cast<VBaseHostBuffer&>(src.vk_host_buffer()));
        }

        /** D2H: direct Vulkan copy from device into cpu_fixed_matrix's pinned host buffer. */
        void copy_to_cpu(VulkanQueue& queue, cpu_fixed_matrix<ElementType, X, Y>& dst) const
        {
            this->m_data.copy_from_offload_buffer(queue, dst.vk_host_buffer());
        }

        /** Fill with random values using a cpu_fixed_matrix intermediate. */
        void fill_rand(VulkanQueue& queue, ElementType lo, ElementType hi)
        {
            cpu_fixed_matrix<ElementType, X, Y> tmp;
            auto* ptr = tmp.data();
            for (size_t i = 0; i < ROWS * COLS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
            copy_from_cpu(queue, tmp);
        }

        /** H2D partial: upload one row from the given cpu source. */
        void copy_row_to_offload_buffer(VulkanQueue& queue, X x, const cpu_fixed_matrix<ElementType, X, Y>& src)
        {
            assert(static_cast<size_t>(x) < ROWS);
            this->m_data.copy_range_to_offload_buffer(
                queue, const_cast<VBaseHostBuffer&>(src.vk_host_buffer()),
                static_cast<size_t>(x) * COLS, COLS);
        }

        /** Export one row from the cpu source (no device access needed). */
        template <typename Out>
        static void export_row(X x, const cpu_fixed_matrix<ElementType, X, Y>& src, Out& out)
        {
            assert(static_cast<size_t>(x) < ROWS);
            const auto* row = src.data() + static_cast<size_t>(x) * COLS;
            for (size_t i = 0; i < COLS; ++i)
                out[i] = row[i];
        }

        size_t storage_size_bytes() const
        {
            return ROWS * COLS * sizeof(ElementType);
        }


    };
} // namespace rllm
