#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <IMemorySpace.hpp>
#include <Range.hpp>
#include <offloadable_data.hpp>
#include <rllm_vulkan_runtime.hpp>
#include <cpu/cpu_flex_size_matrix.hpp>
#include <parallel.hpp>

namespace rllm
{
    /** Both dimensions are fully runtime-sized. GPU-offloadable. */
    template <typename ElementType, typename X, typename Y>
    class gpu_flex_size_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_flex_size_matrix()
            : offloadable_data<ElementType>(ROWS * COLS)
            , m_rows(X::START), m_cols(Y::START)
            , m_capacity_elements(ROWS * COLS)
        {}

        gpu_flex_size_matrix(X rows, Y cols)
            : offloadable_data<ElementType>(ROWS * COLS)
            , m_rows(rows), m_cols(cols)
            , m_capacity_elements(ROWS * COLS)
        {}

        gpu_flex_size_matrix(const gpu_flex_size_matrix&) = delete;
        gpu_flex_size_matrix& operator=(const gpu_flex_size_matrix&) = delete;
        gpu_flex_size_matrix(gpu_flex_size_matrix&&) = default;
        gpu_flex_size_matrix& operator=(gpu_flex_size_matrix&&) = default;

        ~gpu_flex_size_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(element_count_for_size(rows, cols));
            m_rows = rows;
            m_cols = cols;
        }

        /** H2D: direct Vulkan copy from cpu_flex_size_matrix's pinned buffer to device. */
        void copy_from_cpu(VulkanQueue& queue, const cpu_flex_size_matrix<ElementType, X, Y>& src)
        {
            ensure_capacity(element_count_for_size(src.num_rows(), src.num_cols()));
            m_rows = src.num_rows(); m_cols = src.num_cols();
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().write(queue,
                const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), bytes);
        }

        /** D2H: direct Vulkan copy from device into cpu_flex_size_matrix's pinned buffer. */
        void copy_to_cpu(VulkanQueue& queue, cpu_flex_size_matrix<ElementType, X, Y>& dst) const
        {
            dst.set_size(m_rows, m_cols);
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().read(queue,
                dst.vk_host_buffer(), bytes);
        }

        X num_rows() const { return m_rows; }
        Y num_cols() const { return m_cols; }
        size_t storage_size_bytes() const { return m_capacity_elements * sizeof(ElementType); }

      private:
        static size_t element_count_for_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            return std::max<size_t>(1, static_cast<size_t>(rows) * static_cast<size_t>(cols));
        }

        size_t flat_index(X x, Y y) const
        {
            return static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y);
        }

        void ensure_capacity(size_t requested_elements)
        {
            if (requested_elements <= m_capacity_elements)
                return;
            std::fprintf(stderr, "gpu_flex_size_matrix exceeded its startup device allocation\n");
            std::abort();
        }

        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };
} // namespace rllm
