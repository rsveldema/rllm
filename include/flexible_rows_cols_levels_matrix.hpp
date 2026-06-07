#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <IMemorySpace.hpp>
#include <parallel.hpp>
#include <offloadable_data.hpp>


namespace rllm
{
    /** the number of levels, rows, and columns can vary */
    template <typename ElementType, typename L, typename X, typename Y>
    class flexible_rows_cols_levels_matrix: public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t LEVELS = static_cast<size_t>(L::MAX);
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_rows_cols_levels_matrix()
            : offloadable_data<ElementType>(1)
            , m_levels(L::START)
            , m_rows(X::START)
            , m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        flexible_rows_cols_levels_matrix(L levels, X rows, Y cols)
            : offloadable_data<ElementType>(element_count_for_size(levels, rows, cols))
            , m_levels(levels)
            , m_rows(rows)
            , m_cols(cols)
            , m_capacity_elements(element_count_for_size(levels, rows, cols))
        {}

        flexible_rows_cols_levels_matrix(const flexible_rows_cols_levels_matrix& other)
            : offloadable_data<ElementType>(element_count_for_size(other.m_levels, other.m_rows, other.m_cols))
            , m_levels(other.m_levels)
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_levels, other.m_rows, other.m_cols))
        {
            this->m_data = other.m_data;
        }

        flexible_rows_cols_levels_matrix& operator=(const flexible_rows_cols_levels_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_levels, other.m_rows, other.m_cols);
                this->m_data = other.m_data;
                m_levels = other.m_levels;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = element_count_for_size(other.m_levels, other.m_rows, other.m_cols);
            }
            return *this;
        }

        flexible_rows_cols_levels_matrix(flexible_rows_cols_levels_matrix&& other)
            : offloadable_data<ElementType>(element_count_for_size(other.m_levels, other.m_rows, other.m_cols))
            , m_levels(other.m_levels)
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_levels, other.m_rows, other.m_cols))
        {
            this->m_data = other.m_data;
        }

        flexible_rows_cols_levels_matrix& operator=(flexible_rows_cols_levels_matrix&& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_levels, other.m_rows, other.m_cols);
                this->m_data = other.m_data;
                m_levels = other.m_levels;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = element_count_for_size(other.m_levels, other.m_rows, other.m_cols);
            }
            return *this;
        }

        ~flexible_rows_cols_levels_matrix() = default;

        void set_size(L levels, X rows, Y cols)
        {
            assert(static_cast<size_t>(levels) <= LEVELS);
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(levels, rows, cols);
            m_levels = levels;
            m_rows = rows;
            m_cols = cols;
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        void set(LevelIndex level, RowIndex row, ColIndex col, ElementType value)
        {
            this->m_data.get()[flat_index(level, row, col)] = value;
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        const ElementType& get(LevelIndex level, RowIndex row, ColIndex col) const
        {
            return this->m_data.get()[flat_index(level, row, col)];
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        ElementType& operator[](LevelIndex level, RowIndex row, ColIndex col)
        {
            return this->m_data.get()[flat_index(level, row, col)];
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        const ElementType& operator[](LevelIndex level, RowIndex row, ColIndex col) const
        {
            return this->m_data.get()[flat_index(level, row, col)];
        }

        L num_levels() const
        {
            return m_levels;
        }

        X num_rows() const
        {
            return m_rows;
        }

        Y num_cols() const
        {
            return m_cols;
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

        static size_t element_count_for_size(L levels, X rows, Y cols)
        {
            assert(static_cast<size_t>(levels) <= LEVELS);
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            return std::max<size_t>(
                1,
                static_cast<size_t>(levels) * static_cast<size_t>(rows) * static_cast<size_t>(cols)
            );
        }

        template <typename LevelIndex, typename RowIndex, typename ColIndex>
        size_t flat_index(LevelIndex level, RowIndex row, ColIndex col) const
        {
            const size_t level_i = to_size(level);
            const size_t row_i = to_size(row);
            const size_t col_i = to_size(col);
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
            this->m_data.resize(requested_elements);
            m_capacity_elements = requested_elements;
        }

        L m_levels;
        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };

} // namespace rllm
