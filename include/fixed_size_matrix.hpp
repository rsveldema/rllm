#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <Range.hpp>
#include <RandomHelpers.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    class fixed_size_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        fixed_size_matrix()
        {
            m_data.fill(ElementType{});
        }
        ~fixed_size_matrix() = default;

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        void fill_rand(ElementType lo, ElementType hi)
        {
            for (auto& v : m_data)
                v = get_random_value(lo, hi);
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
            cell = std::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] += delta;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
    };
} // namespace rllm
