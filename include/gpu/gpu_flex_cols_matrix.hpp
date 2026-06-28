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
#include <cpu/cpu_flex_cols_matrix.hpp>

namespace rllm
{
    /** Columns can vary at runtime, rows are fixed. GPU-offloadable. */
    template <typename ElementType, typename X, typename Y>
    class gpu_flex_cols_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_flex_cols_matrix()
            : offloadable_data<ElementType>(ROWS * COLS), m_cols(Y::MAX)
        {}

        gpu_flex_cols_matrix(Y cols)
            : offloadable_data<ElementType>(ROWS * COLS), m_cols(cols)
        {}
        gpu_flex_cols_matrix(const gpu_flex_cols_matrix&) = delete;
        gpu_flex_cols_matrix& operator=(const gpu_flex_cols_matrix&) = delete;
        gpu_flex_cols_matrix(gpu_flex_cols_matrix&&) = default;
        gpu_flex_cols_matrix& operator=(gpu_flex_cols_matrix&&) = default;

        ~gpu_flex_cols_matrix() = default;

        void set_cols(Y cols)
        {
            assert(static_cast<size_t>(cols) <= COLS);
            m_cols = cols;
        }
        /** H2D: direct Vulkan copy from cpu_flex_cols_matrix's pinned buffer to device. */
        void copy_from_cpu(const cpu_flex_cols_matrix<ElementType, X, Y>& src)
        {
            m_cols = src.num_cols();
            const auto bytes = static_cast<VkDeviceSize>(ROWS * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().write(rllm::vulkan_runtime::get_queue(0),
                const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), bytes);
        }

        /** D2H: direct Vulkan copy from device into cpu_flex_cols_matrix's pinned buffer. */
        void copy_to_cpu(cpu_flex_cols_matrix<ElementType, X, Y>& dst) const
        {
            dst.set_cols(m_cols);
            const auto bytes = static_cast<VkDeviceSize>(ROWS * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().read(rllm::vulkan_runtime::get_queue(0),
                dst.vk_host_buffer(), bytes);
        }

        const X num_rows() const { return ROWS; }
        Y num_cols() const { return m_cols; }
        size_t storage_size_bytes() const { return ROWS * COLS * sizeof(ElementType); }

      private:
        Y m_cols;
    };
} // namespace rllm
