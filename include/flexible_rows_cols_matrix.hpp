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

namespace rllm
{
    /** the number of columns AND rows can vary */
    template <typename ElementType, typename X, typename Y>
    class flexible_rows_cols_matrix: public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_rows_cols_matrix()
            : offloadable_data<ElementType>(1)
            , m_rows(X::START), m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        flexible_rows_cols_matrix( X rows, Y cols)
            : offloadable_data<ElementType>(element_count_for_size(rows, cols))
            , m_rows(rows), m_cols(cols)
            , m_capacity_elements(element_count_for_size(rows, cols))
        {}

        flexible_rows_cols_matrix(const flexible_rows_cols_matrix& other)
            : offloadable_data<ElementType>(element_count_for_size(other.m_rows, other.m_cols))
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_rows, other.m_cols))
        {
            this->m_data = other.m_data;
        }

        flexible_rows_cols_matrix& operator=(const flexible_rows_cols_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows, other.m_cols);
                this->m_data = other.m_data;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = element_count_for_size(other.m_rows, other.m_cols);
            }
            return *this;
        }

        flexible_rows_cols_matrix(flexible_rows_cols_matrix&& other)
            : offloadable_data<ElementType>(element_count_for_size(other.m_rows, other.m_cols))
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_rows, other.m_cols))
        {
            this->m_data = other.m_data;
        }

        flexible_rows_cols_matrix& operator=(flexible_rows_cols_matrix&& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows, other.m_cols);
                this->m_data = other.m_data;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = element_count_for_size(other.m_rows, other.m_cols);
            }
            return *this;
        }

        ~flexible_rows_cols_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(rows, cols);
            m_rows = rows;
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        size_t storage_size_bytes() const
        {
            return m_capacity_elements * sizeof(ElementType);
        }

        X num_rows() const { return m_rows; }
        Y num_cols() const { return m_cols; }

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
            this->m_data.resize(requested_elements);
            m_capacity_elements = requested_elements;
        }

        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };

} // namespace rllm
