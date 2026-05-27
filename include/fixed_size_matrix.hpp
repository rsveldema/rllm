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
    /** We assume that the ElementType supports default construction, copy assignment, and arithmetic operations.
     * Should be sth like float/float16/int8, not a complex struct.  The X and Y template parameters are the enum 
     * types for the row and column indices, respectively, and are only used to determine the matrix dimensions 
     * and provide type safety for indexing.
     */
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

        inline void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        inline ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline void fill(ElementType value)
        {
            m_data.fill(value);
        }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            for (auto& v : m_data)
                v = static_cast<ElementType>(get_random_value(lo, hi));
        }

        inline void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        inline void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        inline void add_no_clamp(const X x, const Y y, ElementType delta)
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
