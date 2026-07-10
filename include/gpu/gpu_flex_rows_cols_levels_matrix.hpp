#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <utility>

#include <IMemorySpace.hpp>
#include <parallel.hpp>
#include <offloadable_data.hpp>
#include <rllm_vulkan_runtime.hpp>
#include <cpu/cpu_flex_rows_cols_levels_matrix.hpp>

namespace rllm
{
    /** Levels, rows, and columns can all vary at runtime. GPU-offloadable. */
    template <typename ElementType, typename L, typename X, typename Y>
    class gpu_flex_rows_cols_levels_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t LEVELS = static_cast<size_t>(L::MAX);
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        gpu_flex_rows_cols_levels_matrix()
            : offloadable_data<ElementType>(LEVELS * ROWS * COLS)
            , m_levels(L::START), m_rows(X::START), m_cols(Y::START)
            , m_capacity_elements(LEVELS * ROWS * COLS)
        {}

        gpu_flex_rows_cols_levels_matrix(L levels, X rows, Y cols)
            : offloadable_data<ElementType>(LEVELS * ROWS * COLS)
            , m_levels(levels), m_rows(rows), m_cols(cols)
            , m_capacity_elements(LEVELS * ROWS * COLS)
        {}

        gpu_flex_rows_cols_levels_matrix(const gpu_flex_rows_cols_levels_matrix&) = delete;
        gpu_flex_rows_cols_levels_matrix& operator=(const gpu_flex_rows_cols_levels_matrix&) = delete;
        gpu_flex_rows_cols_levels_matrix(gpu_flex_rows_cols_levels_matrix&&) = default;
        gpu_flex_rows_cols_levels_matrix& operator=(gpu_flex_rows_cols_levels_matrix&&) = default;

        ~gpu_flex_rows_cols_levels_matrix() = default;

        void set_size(L levels, X rows, Y cols)
        {
            assert(static_cast<size_t>(levels) <= LEVELS);
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(levels, rows, cols);
            m_levels = levels; m_rows = rows; m_cols = cols;
        }
        /** H2D: direct Vulkan copy from cpu_flex_rows_cols_levels_matrix's pinned buffer to device. */
        void copy_from_cpu(VulkanQueue& queue, const cpu_flex_rows_cols_levels_matrix<ElementType, L, X, Y>& src)
        {
            ensure_capacity(src.num_levels(), src.num_rows(), src.num_cols());
            m_levels = src.num_levels(); m_rows = src.num_rows(); m_cols = src.num_cols();
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_levels) * static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().write(queue,
                const_cast<VBaseHostBuffer&>(src.vk_host_buffer()), bytes);
        }

        /** D2H: direct Vulkan copy from device into cpu_flex_rows_cols_levels_matrix's pinned buffer. */
        void copy_to_cpu(VulkanQueue& queue, cpu_flex_rows_cols_levels_matrix<ElementType, L, X, Y>& dst) const
        {
            dst.set_size(m_levels, m_rows, m_cols);
            const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(m_levels) * static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols) * sizeof(ElementType));
            this->m_data.device_buffer().read(queue,
                dst.vk_host_buffer(), bytes);
        }

        L num_levels() const { return m_levels; }
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

        static size_t element_count_for_size(L levels, X rows, Y cols)
        {
            assert(static_cast<size_t>(levels) <= LEVELS);
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            return std::max<size_t>(1,
                static_cast<size_t>(levels) * static_cast<size_t>(rows) * static_cast<size_t>(cols));
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        size_t flat_index(LevelIndex level, RowIndex row, ColIndex col) const
        {
            const size_t level_i = to_size(level), row_i = to_size(row), col_i = to_size(col);
            assert(level_i < static_cast<size_t>(m_levels));
            assert(row_i < static_cast<size_t>(m_rows));
            assert(col_i < static_cast<size_t>(m_cols));
            return (level_i * static_cast<size_t>(m_rows) + row_i) * static_cast<size_t>(m_cols) + col_i;
        }

        void ensure_capacity(L levels, X rows, Y cols)
        {
            const size_t requested_elements = element_count_for_size(levels, rows, cols);
            if (requested_elements <= m_capacity_elements)
                return;
            std::fprintf(stderr, "gpu_flex_rows_cols_levels_matrix exceeded its startup device allocation\n");
            std::abort();
        }

        L m_levels;
        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };
} // namespace rllm
