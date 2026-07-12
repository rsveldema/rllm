#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>
#include <cpu/cpu_data.hpp>

namespace rllm
{
    /** Both rows and columns can vary at runtime. CPU-only. */
    template <typename ElementType, typename X, typename Y>
    class cpu_flex_rows_cols_matrix : public cpu_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        cpu_flex_rows_cols_matrix()
            : cpu_data<ElementType>(1)
            , m_rows(X::START), m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        cpu_flex_rows_cols_matrix(X rows, Y cols)
            : cpu_data<ElementType>(element_count_for_size(rows, cols))
            , m_rows(rows), m_cols(cols)
            , m_capacity_elements(element_count_for_size(rows, cols))
        {}

        cpu_flex_rows_cols_matrix(const cpu_flex_rows_cols_matrix& other)
            : cpu_data<ElementType>(element_count_for_size(other.m_rows, other.m_cols))
            , m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_capacity_elements(element_count_for_size(other.m_rows, other.m_cols))
        {
            std::copy(other.data(), other.data() + m_capacity_elements, this->data());
        }

        cpu_flex_rows_cols_matrix& operator=(const cpu_flex_rows_cols_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows, other.m_cols);
                const size_t count = element_count_for_size(other.m_rows, other.m_cols);
                std::copy(other.data(), other.data() + count, this->data());
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = count;
            }
            return *this;
        }

        cpu_flex_rows_cols_matrix(cpu_flex_rows_cols_matrix&&) noexcept = default;
        cpu_flex_rows_cols_matrix& operator=(cpu_flex_rows_cols_matrix&&) noexcept = default;
        ~cpu_flex_rows_cols_matrix() = default;

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
            this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
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
            this->resize(requested_elements);
            m_capacity_elements = requested_elements;
        }

        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };
} // namespace rllm
