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
    /** the number of columns can vary, the number of rows are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_cols_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_cols_matrix()
            : offloadable_data<ElementType>(ROWS * COLS), m_cols(Y::MAX)
        {}

        flexible_cols_matrix(Y cols)
            : offloadable_data<ElementType>(ROWS * COLS), m_cols(cols)
        {}

        flexible_cols_matrix(const flexible_cols_matrix& other)
            :  offloadable_data<ElementType>(ROWS * COLS), m_cols(other.m_cols)
        {
            this->m_data = other.m_data;
        }

        flexible_cols_matrix& operator=(const flexible_cols_matrix& other)
        {
            if (this != &other)
            {
                this->m_data = other.m_data;
                m_cols = other.m_cols;
            }
            return *this;
        }

        flexible_cols_matrix(flexible_cols_matrix&& other)
            : offloadable_data<ElementType>(ROWS * COLS), m_cols(other.m_cols)
        {
            this->m_data = other.m_data;
        }

        flexible_cols_matrix& operator=(flexible_cols_matrix&& other)
        {
            if (this != &other)
            {
                this->m_data = other.m_data;
                m_cols = other.m_cols;
            }
            return *this;
        }

        ~flexible_cols_matrix() = default;

        void set_cols(Y cols)
        {
            assert(static_cast<size_t>(cols) <= COLS);
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->m_data.get()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const X num_rows() const
        {
            return ROWS;
        }

        Y num_cols() const
        {
            return m_cols;
        }


        size_t storage_size_bytes() const
        {
            return ROWS * COLS * sizeof(ElementType);
        }

      private:
        Y m_cols;
    };

} // namespace rllm
