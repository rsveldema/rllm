#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include <parallel.hpp>
#include <Range.hpp>
#include <RandomHelpers.hpp>
#include <IMemorySpace.hpp>
#include <offloadable_data.hpp>

#include "fixed_size_matrix.hpp"

namespace rllm
{
#ifndef PREFER_SPEED_OVER_MEMORY
#define PREFER_SPEED_OVER_MEMORY 0
#endif

#if PREFER_SPEED_OVER_MEMORY

template <typename ElementType, typename X, typename Y>
using fixed_size_triangular_matrix = fixed_size_matrix<ElementType, X, Y>;

#else
    /** We assume that the ElementType supports default construction, copy assignment, and arithmetic operations.
     * Should be sth like float/float16/int8, not a complex struct.  The X and Y template parameters are the enum
     * types for the row and column indices, respectively, and are only used to determine the matrix dimensions
     * and provide type safety for indexing.
     */
    template <typename ElementType, typename X, typename Y>
    class fixed_size_triangular_matrix : public offloadable_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);
        static_assert(ROWS == COLS, "fixed_size_triangular_matrix currently requires a square matrix");
        static constexpr size_t NUM_ELTS = ROWS * (ROWS + 1) / 2;

        fixed_size_triangular_matrix()
            : offloadable_data<ElementType>(NUM_ELTS)
        {}

        fixed_size_triangular_matrix(const fixed_size_triangular_matrix& other)
            : offloadable_data<ElementType>(NUM_ELTS)
        {
            assign_from(other);
        }

        fixed_size_triangular_matrix& operator=(const fixed_size_triangular_matrix& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        fixed_size_triangular_matrix(fixed_size_triangular_matrix&& other)
            : offloadable_data<ElementType>(NUM_ELTS)
        {
            assign_from(other);
        }

        fixed_size_triangular_matrix& operator=(fixed_size_triangular_matrix&& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        ~fixed_size_triangular_matrix() = default;

        inline void set(const X x, const Y y, ElementType value)
        {
            const size_t sx = static_cast<size_t>(x);
            const size_t sy = static_cast<size_t>(y);
            assert(sx < ROWS);
            assert(sy < COLS);
            assert(sy <= sx);
            const size_t k = sx * (sx + 1) / 2 + sy;
            assert(k < NUM_ELTS);
            this->m_data.get()[k] = value;
        }

        inline ElementType& get(const X x, const Y y)
        {
            const size_t sx = static_cast<size_t>(x);
            const size_t sy = static_cast<size_t>(y);
            assert(sx < ROWS);
            assert(sy < COLS);
            assert(sy <= sx);
            const size_t k = sx * (sx + 1) / 2 + sy;
            assert(k < NUM_ELTS);
            return this->m_data.get()[k];
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            const size_t sx = static_cast<size_t>(x);
            const size_t sy = static_cast<size_t>(y);
            assert(sx < ROWS);
            assert(sy < COLS);
            assert(sy <= sx);
            const size_t k = sx * (sx + 1) / 2 + sy;
            assert(k < NUM_ELTS);
            return this->m_data.get()[k];
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        inline const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        inline ElementType& operator[](X x, Y y)
        {
            return get(x, y);
        }

        template <std::integral XIndex>
        inline ElementType& operator[](XIndex x, Y y)
        {
            return get(static_cast<X>(x), y);
        }

        template <std::integral YIndex>
        inline ElementType& operator[](X x, YIndex y)
        {
            return get(x, static_cast<Y>(y));
        }

        template <std::integral XIndex, std::integral YIndex>
        inline ElementType& operator[](XIndex x, YIndex y)
        {
            return get(static_cast<X>(x), static_cast<Y>(y));
        }

        inline const ElementType& operator[](X x, Y y) const
        {
            return get(x, y);
        }

        template <std::integral XIndex>
        inline const ElementType& operator[](XIndex x, Y y) const
        {
            return get(static_cast<X>(x), y);
        }

        template <std::integral YIndex>
        inline const ElementType& operator[](X x, YIndex y) const
        {
            return get(x, static_cast<Y>(y));
        }

        template <std::integral XIndex, std::integral YIndex>
        inline const ElementType& operator[](XIndex x, YIndex y) const
        {
            return get(static_cast<X>(x), static_cast<Y>(y));
        }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            auto* ptr = this->m_data.get();
            for (size_t i = 0; i < NUM_ELTS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
        }


        size_t storage_size_bytes() const
        {
            return NUM_ELTS * sizeof(ElementType);
        }

      private:
        void assign_from(const fixed_size_triangular_matrix& other)
        {
            this->m_data = other.m_data;
        }
    };

#endif
} // namespace rllm
