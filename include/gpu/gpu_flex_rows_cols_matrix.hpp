#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>
#include <rllm_vulkan_runtime.hpp>
#include <cpu/cpu_flex_rows_cols_matrix.hpp>

namespace rllm
{
    /** Both rows and columns can vary at runtime. GPU-offloadable. */
    template <typename ElementType, typename X, typename Y>
    class gpu_flex_rows_cols_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_flex_rows_cols_matrix()
            : offloadable_data<ElementType>(1)
            , m_rows(X::START), m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        gpu_flex_rows_cols_matrix(X rows, Y cols)
            : offloadable_data<ElementType>(element_count_for_size(rows, cols))
            , m_rows(rows), m_cols(cols)
            , m_capacity_elements(element_count_for_size(rows, cols))
        {}

        gpu_flex_rows_cols_matrix(const gpu_flex_rows_cols_matrix&) = delete;
        gpu_flex_rows_cols_matrix& operator=(const gpu_flex_rows_cols_matrix&) = delete;
        gpu_flex_rows_cols_matrix(gpu_flex_rows_cols_matrix&&) = default;
        gpu_flex_rows_cols_matrix& operator=(gpu_flex_rows_cols_matrix&&) = default;

        ~gpu_flex_rows_cols_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(rows, cols);
            m_rows = rows;
            m_cols = cols;
        }
        /** H2D: direct Vulkan copy from cpu_flex_rows_cols_matrix's pinned buffer to device. */
        void copy_from_cpu(const cpu_flex_rows_cols_matrix<ElementType, X, Y>& src)
        {
            ensure_capacity(src.num_rows(), src.num_cols());
            m_rows = src.num_rows(); m_cols = src.num_cols();
            m_capacity_elements = std::max<size_t>(1, static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols));
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().write(rllm::vulkan_runtime::get_queue(0),
                const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), bytes);
        }

        /** D2H: direct Vulkan copy from device into cpu_flex_rows_cols_matrix's pinned buffer. */
        void copy_to_cpu(cpu_flex_rows_cols_matrix<ElementType, X, Y>& dst) const
        {
            dst.set_size(m_rows, m_cols);
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().read(rllm::vulkan_runtime::get_queue(0),
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

        void ensure_capacity(X rows, Y cols)
        {
            const size_t requested_elements = element_count_for_size(rows, cols);
            if (requested_elements <= m_capacity_elements)
                return;
            this->m_data = DevicePointer<ElementType>(requested_elements);
            m_capacity_elements = requested_elements;
        }

        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };
} // namespace rllm
