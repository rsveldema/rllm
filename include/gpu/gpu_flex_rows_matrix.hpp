#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>
#include <cpu/cpu_flex_rows_matrix.hpp>
#include <rllm_vulkan_runtime.hpp>

namespace rllm
{
    /** Rows can vary at runtime, columns are fixed. GPU-offloadable. */
    template <typename ElementType, typename X, typename Y>
    class gpu_flex_rows_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_flex_rows_matrix()
            : offloadable_data<ElementType>(COLS)
            , m_rows(X::START)
            , m_capacity_rows(1)
        {}

        gpu_flex_rows_matrix(X rows)
            : offloadable_data<ElementType>(element_count_for_rows(rows))
            , m_rows(rows)
            , m_capacity_rows(std::max<size_t>(1, static_cast<size_t>(rows)))
        {}

        gpu_flex_rows_matrix(const gpu_flex_rows_matrix&) = delete;
        gpu_flex_rows_matrix& operator=(const gpu_flex_rows_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows);
                m_rows = other.m_rows;
                m_capacity_rows = other.m_capacity_rows;
                const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * COLS * sizeof(ElementType));
                this->m_data.device_buffer().copy_from(rllm::vulkan_runtime::get_queue(0),
                    const_cast<VBaseDeviceBuffer&>(other.m_data.device_buffer()), bytes);
            }
            return *this;
        }
        gpu_flex_rows_matrix(gpu_flex_rows_matrix&&) = default;
        gpu_flex_rows_matrix& operator=(gpu_flex_rows_matrix&&) = default;

        ~gpu_flex_rows_matrix() = default;

        void copy_from(VulkanQueue& queue, const gpu_flex_rows_matrix& other)
        {
            ensure_capacity(other.m_rows);
            m_rows = other.m_rows;
            m_capacity_rows = other.m_capacity_rows;
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * COLS * sizeof(ElementType));
            this->m_data.device_buffer().copy_from(queue,
                const_cast<VBaseDeviceBuffer&>(other.m_data.device_buffer()), bytes);
        }

        void set_rows(X rows)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            ensure_capacity(rows);
            m_rows = rows;
        }

        /** H2D: direct Vulkan copy from cpu_flex_rows_matrix's pinned buffer to device. */
        void copy_from_cpu(VulkanQueue& queue, const cpu_flex_rows_matrix<ElementType, X, Y>& src)
        {
            ensure_capacity(src.num_rows());
            m_rows = src.num_rows();
            m_capacity_rows = std::max<size_t>(1, static_cast<size_t>(m_rows));
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * COLS * sizeof(ElementType));
            this->m_data.device_buffer().write(queue,
                const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), bytes);
        }

        /** D2H: direct Vulkan copy from device into cpu_flex_rows_matrix's pinned buffer. */
        void copy_to_cpu(VulkanQueue& queue, cpu_flex_rows_matrix<ElementType, X, Y>& dst) const
        {
            dst.set_rows(m_rows);
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * COLS * sizeof(ElementType));
            this->m_data.device_buffer().read(queue,
                dst.vk_host_buffer(), bytes);
        }

        X num_rows() const { return m_rows; }
        constexpr Y num_cols() const { return Y::MAX; }
        size_t storage_size_bytes() const { return m_capacity_rows * COLS * sizeof(ElementType); }

      private:
        static size_t element_count_for_rows(X rows)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            return std::max<size_t>(1, static_cast<size_t>(rows)) * COLS;
        }

        void ensure_capacity(X rows)
        {
            const size_t requested_rows = std::max<size_t>(1, static_cast<size_t>(rows));
            if (requested_rows <= m_capacity_rows)
                return;
            this->m_data = DevicePointer<ElementType>(requested_rows * COLS);
            m_capacity_rows = requested_rows;
        }

        X m_rows;
        size_t m_capacity_rows;
    };
} // namespace rllm
