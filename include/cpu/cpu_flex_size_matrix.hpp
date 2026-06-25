#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>

#include <Range.hpp>
#include <cpu/cpu_data.hpp>

namespace rllm
{
    /** Both dimensions are fully runtime-sized. CPU-only. */
    template <typename ElementType, typename X, typename Y>
    class cpu_flex_size_matrix : public cpu_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        cpu_flex_size_matrix()
            : cpu_data<ElementType>(1)
            , m_rows(X::START), m_cols(Y::START)
            , m_capacity_elements(1)
        {}

        cpu_flex_size_matrix(X rows, Y cols)
            : cpu_data<ElementType>(element_count_for_size(rows, cols))
            , m_rows(rows), m_cols(cols)
            , m_capacity_elements(element_count_for_size(rows, cols))
        {}

        cpu_flex_size_matrix(const cpu_flex_size_matrix& other)
            : cpu_data<ElementType>(other.m_capacity_elements)
            , m_rows(other.m_rows), m_cols(other.m_cols)
            , m_capacity_elements(other.m_capacity_elements)
        {
            std::copy(other.data(), other.data() + m_capacity_elements, this->data());
        }

        cpu_flex_size_matrix& operator=(const cpu_flex_size_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_capacity_elements);
                std::copy(other.data(), other.data() + other.m_capacity_elements, this->data());
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_capacity_elements = other.m_capacity_elements;
            }
            return *this;
        }

        cpu_flex_size_matrix(cpu_flex_size_matrix&&) noexcept = default;
        cpu_flex_size_matrix& operator=(cpu_flex_size_matrix&&) noexcept = default;
        ~cpu_flex_size_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            ensure_capacity(element_count_for_size(rows, cols));
            m_rows = rows;
            m_cols = cols;
        }

        inline void set(X x, Y y, ElementType value) { (*this)[x, y] = value; }
        inline void set(const std::pair<const X, const Y>& indices, ElementType value) { set(indices.first, indices.second, value); }
        inline const ElementType& get(X x, Y y) const { return (*this)[x, y]; }
        inline const ElementType& get(const std::pair<const X, const Y>& indices) const { return get(indices.first, indices.second); }

        inline ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[flat_index(x, y)];
        }

        template <std::integral XIndex>
        inline ElementType& operator[](XIndex x, Y y) { return (*this)[static_cast<X>(x), y]; }
        template <std::integral YIndex>
        inline ElementType& operator[](X x, YIndex y) { return (*this)[x, static_cast<Y>(y)]; }
        template <std::integral XIndex, std::integral YIndex>
        inline ElementType& operator[](XIndex x, YIndex y) { return (*this)[static_cast<X>(x), static_cast<Y>(y)]; }

        inline const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[flat_index(x, y)];
        }

        template <std::integral XIndex>
        inline const ElementType& operator[](XIndex x, Y y) const { return (*this)[static_cast<X>(x), y]; }
        template <std::integral YIndex>
        inline const ElementType& operator[](X x, YIndex y) const { return (*this)[x, static_cast<Y>(y)]; }
        template <std::integral XIndex, std::integral YIndex>
        inline const ElementType& operator[](XIndex x, YIndex y) const { return (*this)[static_cast<X>(x), static_cast<Y>(y)]; }

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
            this->resize(requested_elements);
            m_capacity_elements = requested_elements;
        }

        X m_rows;
        Y m_cols;
        size_t m_capacity_elements;
    };
} // namespace rllm
