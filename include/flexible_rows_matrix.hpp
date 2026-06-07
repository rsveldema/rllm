#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <concepts>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>


namespace rllm
{
    /** the number of rows can vary, the number of columns are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_rows_matrix: public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_rows_matrix()
            : offloadable_data<ElementType>(COLS)
            , m_rows(X::START)
            , m_capacity_rows(1)
        {}

        flexible_rows_matrix(X rows)
            : offloadable_data<ElementType>(element_count_for_rows(rows))
            , m_rows(rows)
            , m_capacity_rows(std::max<size_t>(1, static_cast<size_t>(rows)))
        {}

        flexible_rows_matrix(const flexible_rows_matrix& other)
            : offloadable_data<ElementType>(element_count_for_rows(other.m_rows))
            , m_rows(other.m_rows)
            , m_capacity_rows(std::max<size_t>(1, static_cast<size_t>(other.m_rows)))
        {
            this->m_data = other.m_data;
        }

        flexible_rows_matrix& operator=(const flexible_rows_matrix& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows);
                this->m_data = other.m_data;
                m_rows = other.m_rows;
                m_capacity_rows = std::max<size_t>(1, static_cast<size_t>(other.m_rows));
            }
            return *this;
        }

        flexible_rows_matrix(flexible_rows_matrix&& other)
            : offloadable_data<ElementType>(element_count_for_rows(other.m_rows))
            , m_rows(other.m_rows)
            , m_capacity_rows(std::max<size_t>(1, static_cast<size_t>(other.m_rows)))
        {
            this->m_data = other.m_data;
        }

        flexible_rows_matrix& operator=(flexible_rows_matrix&& other)
        {
            if (this != &other)
            {
                ensure_capacity(other.m_rows);
                this->m_data = other.m_data;
                m_rows = other.m_rows;
                m_capacity_rows = std::max<size_t>(1, static_cast<size_t>(other.m_rows));
            }
            return *this;
        }

        ~flexible_rows_matrix() = default;

        void set_rows(X rows)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            ensure_capacity(rows);
            m_rows = rows;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            this->m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return this->m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return this->m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        template <std::integral YIndex>
        ElementType& operator[](X x, YIndex y)
        {
            return (*this)[x, static_cast<Y>(y)];
        }

        template <std::integral XIndex>
        ElementType& operator[](XIndex x, Y y)
        {
            return (*this)[static_cast<X>(x), y];
        }

        template <std::integral XIndex, std::integral YIndex>
        ElementType& operator[](XIndex x, YIndex y)
        {
            return (*this)[static_cast<X>(x), static_cast<Y>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return this->m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        template <std::integral YIndex>
        const ElementType& operator[](X x, YIndex y) const
        {
            return (*this)[x, static_cast<Y>(y)];
        }

        template <std::integral XIndex>
        const ElementType& operator[](XIndex x, Y y) const
        {
            return (*this)[static_cast<X>(x), y];
        }

        template <std::integral XIndex, std::integral YIndex>
        const ElementType& operator[](XIndex x, YIndex y) const
        {
            return (*this)[static_cast<X>(x), static_cast<Y>(y)];
        }

        X num_rows() const
        {
            return m_rows;
        }

        constexpr Y num_cols() const
        {
            return Y::MAX;
        }


        size_t storage_size_bytes() const
        {
            return m_capacity_rows * COLS * sizeof(ElementType);
        }

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
            this->m_data.resize(requested_rows * COLS);
            m_capacity_rows = requested_rows;
        }

        X m_rows;
        size_t m_capacity_rows;
    };

} // namespace rllm
