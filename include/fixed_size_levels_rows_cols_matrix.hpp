#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <IMemorySpace.hpp>

namespace rllm
{
    /** Fixed number of levels, with runtime-sized rows and columns. */
    template <typename ElementType, typename L, typename X, typename Y>
    class fixed_size_levels_rows_cols_matrix
    {
      public:
        static constexpr size_t LEVELS = static_cast<size_t>(L::MAX);
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        fixed_size_levels_rows_cols_matrix()
            : m_data(1)
            , m_rows(X::START)
            , m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        fixed_size_levels_rows_cols_matrix(X rows, Y cols)
            : m_data(element_count_for_size(rows, cols))
            , m_rows(rows)
            , m_cols(cols)
            , m_capacity_elements(element_count_for_size(rows, cols))
        {}

        fixed_size_levels_rows_cols_matrix(const fixed_size_levels_rows_cols_matrix& other)
            : m_data(element_count_for_size(other.m_rows, other.m_cols))
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_rows, other.m_cols))
        {
            m_data = other.m_data;
        }

        fixed_size_levels_rows_cols_matrix& operator=(const fixed_size_levels_rows_cols_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows, other.m_cols);
                m_data = other.m_data;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = element_count_for_size(other.m_rows, other.m_cols);
            }
            return *this;
        }

        fixed_size_levels_rows_cols_matrix(fixed_size_levels_rows_cols_matrix&& other)
            : m_data(element_count_for_size(other.m_rows, other.m_cols))
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_rows, other.m_cols))
        {
            m_data = other.m_data;
        }

        fixed_size_levels_rows_cols_matrix& operator=(fixed_size_levels_rows_cols_matrix&& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows, other.m_cols);
                m_data = other.m_data;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = element_count_for_size(other.m_rows, other.m_cols);
            }
            return *this;
        }

        ~fixed_size_levels_rows_cols_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(rows, cols);
            m_rows = rows;
            m_cols = cols;
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        void set(LevelIndex level, RowIndex row, ColIndex col, ElementType value)
        {
            m_data.get()[flat_index(level, row, col)] = value;
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        const ElementType& get(LevelIndex level, RowIndex row, ColIndex col) const
        {
            return m_data.get()[flat_index(level, row, col)];
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        ElementType& operator[](LevelIndex level, RowIndex row, ColIndex col)
        {
            return m_data.get()[flat_index(level, row, col)];
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        const ElementType& operator[](LevelIndex level, RowIndex row, ColIndex col) const
        {
            return m_data.get()[flat_index(level, row, col)];
        }

        void zero()
        {
            m_data.zero();
        }

        L num_levels() const
        {
            return L::MAX;
        }

        X num_rows() const
        {
            return m_rows;
        }

        Y num_cols() const
        {
            return m_cols;
        }

        ElementType* data()
        {
            return m_data.staging_data();
        }

        const ElementType* data() const
        {
            return m_data.staging_data();
        }

        ElementType* raw_staging_data() const
        {
            return m_data.raw_staging_data();
        }

#if RLLM_DEVICE_POINTER_HAS_OFFLOAD
        void* raw_offload_data() const
        {
            return m_data.raw_offload_data();
        }
#endif

#if defined(USE_VULKAN_OFFLOAD)
        VulkanRuntimeBuffer& vulkan_runtime_buffer() const
        {
            return m_data.vulkan_runtime_buffer();
        }
#endif

        DeviceMemoryOwner device_memory_owner() const
        {
            return m_data.device_memory_owner();
        }

        void set_pending_flush(std::function<void()> flush_fn)
        {
            m_data.set_pending_flush(std::move(flush_fn));
        }

        void mark_device_latest()
        {
            m_data.mark_device_latest();
        }

        bool needs_offload_sync() const
        {
            return m_data.needs_offload_sync();
        }

        size_t storage_size_bytes() const
        {
            return m_capacity_elements * sizeof(ElementType);
        }

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
            m_data.resize(requested_elements);
            m_capacity_elements = requested_elements;
        }

        DevicePointer<ElementType> m_data;
        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };
} // namespace rllm
