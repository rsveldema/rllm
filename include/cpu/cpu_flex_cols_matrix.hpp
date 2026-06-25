#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>
#include <cpu/cpu_data.hpp>

namespace rllm
{
    /** Columns can vary at runtime, rows are fixed. CPU-only. */
    template <typename ElementType, typename X, typename Y>
    class cpu_flex_cols_matrix : public cpu_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        cpu_flex_cols_matrix()
            : cpu_data<ElementType>(ROWS * COLS), m_cols(Y::MAX)
        {}

        cpu_flex_cols_matrix(Y cols)
            : cpu_data<ElementType>(ROWS * COLS), m_cols(cols)
        {}

        cpu_flex_cols_matrix(const cpu_flex_cols_matrix& other)
            : cpu_data<ElementType>(ROWS * COLS), m_cols(other.m_cols)
        {
            std::copy(other.data(), other.data() + ROWS * COLS, this->data());
        }

        cpu_flex_cols_matrix& operator=(const cpu_flex_cols_matrix& other)
        {
            if (this != &other)
            {
                std::copy(other.data(), other.data() + ROWS * COLS, this->data());
                m_cols = other.m_cols;
            }
            return *this;
        }

        cpu_flex_cols_matrix(cpu_flex_cols_matrix&&) noexcept = default;
        cpu_flex_cols_matrix& operator=(cpu_flex_cols_matrix&&) noexcept = default;
        ~cpu_flex_cols_matrix() = default;

        void set_cols(Y cols)
        {
            assert(static_cast<size_t>(cols) <= COLS);
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return this->data()[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const X num_rows() const { return ROWS; }
        Y num_cols() const { return m_cols; }
        size_t storage_size_bytes() const { return ROWS * COLS * sizeof(ElementType); }

      private:
        Y m_cols;
    };
} // namespace rllm
