#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>

namespace rllm
{
    /** the number of columns can vary, the number of rows are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_cols_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_cols_matrix()
            : m_cols(Y::MAX)
        {
            m_data.fill(ElementType{});
        }

        flexible_cols_matrix(Y cols)
            : m_cols(cols)
        {
            m_data.fill(ElementType{});
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
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        // Adds each element of other (must have the same runtime dimensions) into this matrix.
        void element_wise_add(const flexible_cols_matrix& other)
        {
            assert(ROWS == other.ROWS && m_cols == other.m_cols);
            const size_t n = static_cast<size_t>(ROWS) * static_cast<size_t>(m_cols);
#pragma omp simd
            for (size_t i = 0; i < n; ++i)
                m_data[i] += other.m_data[i];
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            auto& cell = m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] += delta;
        }

        const X num_rows() const
        {
            return ROWS;
        }

        Y num_cols() const
        {
            return m_cols;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
        Y m_cols;
    };

} // namespace rllm
