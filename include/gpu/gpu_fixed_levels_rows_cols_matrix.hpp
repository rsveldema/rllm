#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>
#include <rllm_vulkan_runtime.hpp>
#include <cpu/cpu_fixed_levels_rows_cols_matrix.hpp>

namespace rllm
{
    /** Fixed number of levels, with runtime-sized rows and columns. GPU-offloadable. */
    template <typename ElementType, typename L, typename X, typename Y>
    class gpu_fixed_levels_rows_cols_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t LEVELS = static_cast<size_t>(L::MAX);
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_fixed_levels_rows_cols_matrix()
            : offloadable_data<ElementType>(1)
            , m_rows(X::START)
            , m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        gpu_fixed_levels_rows_cols_matrix(X rows, Y cols)
            : offloadable_data<ElementType>(element_count_for_size(rows, cols))
            , m_rows(rows)
            , m_cols(cols)
            , m_capacity_elements(element_count_for_size(rows, cols))
        {}

        gpu_fixed_levels_rows_cols_matrix(const gpu_fixed_levels_rows_cols_matrix&) = delete;
        gpu_fixed_levels_rows_cols_matrix& operator=(const gpu_fixed_levels_rows_cols_matrix&) = delete;
        gpu_fixed_levels_rows_cols_matrix(gpu_fixed_levels_rows_cols_matrix&&) = default;
        gpu_fixed_levels_rows_cols_matrix& operator=(gpu_fixed_levels_rows_cols_matrix&&) = default;

        ~gpu_fixed_levels_rows_cols_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(rows, cols);
            m_rows = rows;
            m_cols = cols;
        }
        /** H2D: direct Vulkan copy from cpu_fixed_levels_rows_cols_matrix's pinned buffer to device. */
        void copy_from_cpu(const cpu_fixed_levels_rows_cols_matrix<ElementType, L, X, Y>& src)
        {
            m_rows = src.num_rows();
            m_cols = src.num_cols();
            const auto bytes = static_cast<VkDeviceSize>(LEVELS * static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().write(rllm::vulkan_runtime::get_queue(0),
                const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), bytes);
        }

        /** D2H: direct Vulkan copy from device into cpu_fixed_levels_rows_cols_matrix's pinned buffer. */
        void copy_to_cpu(cpu_fixed_levels_rows_cols_matrix<ElementType, L, X, Y>& dst) const
        {
            dst.set_size(m_rows, m_cols);
            const auto bytes = static_cast<VkDeviceSize>(LEVELS * static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().read(rllm::vulkan_runtime::get_queue(0),
                dst.vk_host_buffer(), bytes);
        }

        L num_levels() const { return L::MAX; }
        X num_rows() const { return m_rows; }
        Y num_cols() const { return m_cols; }
        size_t storage_size_bytes() const { return m_capacity_elements * sizeof(ElementType); }

      private:
        template <typename Index>
        static size_t to_size(Index index)
        {
            if constexpr (std::is_enum_v<std::remove_cv_t<std::remove_reference_t<Index>>>)
                return static_cast<size_t>(index);
            else
                return static_cast<size_t>(index);
        }

        static size_t element_count_for_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            return std::max<size_t>(1, LEVELS * static_cast<size_t>(rows) * static_cast<size_t>(cols));
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        size_t flat_index(LevelIndex level, RowIndex row, ColIndex col) const
        {
            const size_t level_i = to_size(level);
            const size_t row_i = to_size(row);
            const size_t col_i = to_size(col);
            assert(level_i < LEVELS);
            assert(row_i < static_cast<size_t>(m_rows));
            assert(col_i < static_cast<size_t>(m_cols));
            return (level_i * static_cast<size_t>(m_rows) + row_i) * static_cast<size_t>(m_cols) + col_i;
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
