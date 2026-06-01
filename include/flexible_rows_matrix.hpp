#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <concepts>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>

namespace rllm
{
    /** the number of rows can vary, the number of columns are fixed */
    template <typename ElementType, typename X, typename Y>
    class flexible_rows_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_rows_matrix()
            : m_data(ROWS * COLS)
            , m_rows(X::MAX)
        {
            fill(ElementType{});
        }

        flexible_rows_matrix(X rows)
            : m_data(ROWS * COLS)
            , m_rows(rows)
        {
            fill(ElementType{});
        }

        flexible_rows_matrix(const flexible_rows_matrix& other)
            : m_data(ROWS * COLS)
            , m_rows(other.m_rows)
        {
            m_data = other.m_data;
        }

        flexible_rows_matrix& operator=(const flexible_rows_matrix& other)
        {
            if (this != &other)
            {
                m_data = other.m_data;
                m_rows = other.m_rows;
            }
            return *this;
        }

        flexible_rows_matrix(flexible_rows_matrix&& other)
            : m_data(ROWS * COLS)
            , m_rows(other.m_rows)
        {
            m_data = other.m_data;
        }

        flexible_rows_matrix& operator=(flexible_rows_matrix&& other)
        {
            if (this != &other)
            {
                m_data = other.m_data;
                m_rows = other.m_rows;
            }
            return *this;
        }

        ~flexible_rows_matrix() = default;

        void set_rows(X rows)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            m_rows = rows;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
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
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
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

        void fill(ElementType value)
        {
            std::fill_n(m_data.get(), ROWS * COLS, value);
        }

        void zero()
        {
            m_data.zero();
        }

        // Adds each element of other (must have the same runtime dimensions) into this matrix.
        void element_wise_add(const flexible_rows_matrix& other)
        {
            assert(m_rows == other.m_rows);
            const size_t n = static_cast<size_t>(m_rows) * COLS;
            RLLM_OMP_SIMD
            for (size_t i = 0; i < n; ++i)
                m_data.get()[i] += other.m_data.get()[i];
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < COLS);
            m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] += delta;
        }

        X num_rows() const
        {
            return m_rows;
        }

        constexpr Y num_cols() const
        {
            return Y::MAX;
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

        void set_pending_flush(std::function<void()> flush_fn)
        {
            m_data.set_pending_flush(std::move(flush_fn));
        }

        bool needs_offload_sync() const
        {
            return m_data.needs_offload_sync();
        }

        size_t storage_size_bytes() const
        {
            return ROWS * COLS * sizeof(ElementType);
        }

      private:
        DevicePointer<ElementType> m_data;
        X m_rows;
    };

} // namespace rllm
