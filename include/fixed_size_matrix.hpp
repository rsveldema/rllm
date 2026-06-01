#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
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
            : m_data(ROWS * COLS)
        {
            fill(ElementType{});
        }

        fixed_size_matrix(const fixed_size_matrix& other)
            : m_data(ROWS * COLS)
        {
            assign_from(other);
        }

        fixed_size_matrix& operator=(const fixed_size_matrix& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        fixed_size_matrix(fixed_size_matrix&& other)
            : m_data(ROWS * COLS)
        {
            assign_from(other);
        }

        fixed_size_matrix& operator=(fixed_size_matrix&& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        ~fixed_size_matrix() = default;

        inline void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        inline ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline void fill(ElementType value)
        {
            std::fill_n(m_data.get(), ROWS * COLS, value);
        }

        inline void zero()
        {
            m_data.zero();
        }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            auto* ptr = m_data.get();
            for (size_t i = 0; i < ROWS * COLS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
        }

        inline void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
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
            m_data.get()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] += delta;
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
                void assign_from(const fixed_size_matrix& other)
                {
                    m_data = other.m_data;
                }

        DevicePointer<ElementType> m_data;
    };
} // namespace rllm
