#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>

#include <RandomHelpers.hpp>
#include <cpu/cpu_data.hpp>

#ifndef PREFER_SPEED_OVER_MEMORY
#define PREFER_SPEED_OVER_MEMORY 0
#endif

#if PREFER_SPEED_OVER_MEMORY

#include <cpu/cpu_fixed_matrix.hpp>

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    using cpu_fixed_triangular_matrix = cpu_fixed_matrix<ElementType, X, Y>;
} // namespace rllm

#else

namespace rllm
{
    template <typename ElementType, typename X, typename Y>
    class cpu_fixed_triangular_matrix : public cpu_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);
        static_assert(ROWS == COLS, "cpu_fixed_triangular_matrix requires a square matrix");
        static constexpr size_t NUM_ELTS = ROWS * (ROWS + 1) / 2;

        cpu_fixed_triangular_matrix()
            : cpu_data<ElementType>(NUM_ELTS)
        {}

        cpu_fixed_triangular_matrix(const cpu_fixed_triangular_matrix& other)
            : cpu_data<ElementType>(NUM_ELTS)
        {
            std::copy(other.data(), other.data() + NUM_ELTS, this->data());
        }

        cpu_fixed_triangular_matrix& operator=(const cpu_fixed_triangular_matrix& other)
        {
            if (this != &other)
                std::copy(other.data(), other.data() + NUM_ELTS, this->data());
            return *this;
        }

        cpu_fixed_triangular_matrix(cpu_fixed_triangular_matrix&&) noexcept = default;
        cpu_fixed_triangular_matrix& operator=(cpu_fixed_triangular_matrix&&) noexcept = default;
        ~cpu_fixed_triangular_matrix() = default;

        inline void set(const X x, const Y y, ElementType value)
        {
            this->data()[k(x, y)] = value;
        }

        inline ElementType& get(const X x, const Y y)
        {
            return this->data()[k(x, y)];
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            return this->data()[k(x, y)];
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value) { set(indices.first, indices.second, value); }
        inline const ElementType& get(const std::pair<const X, const Y>& indices) const { return get(indices.first, indices.second); }

        inline ElementType& operator[](X x, Y y) { return get(x, y); }
        template <std::integral XI> inline ElementType& operator[](XI x, Y y) { return get(static_cast<X>(x), y); }
        template <std::integral YI> inline ElementType& operator[](X x, YI y) { return get(x, static_cast<Y>(y)); }
        template <std::integral XI, std::integral YI> inline ElementType& operator[](XI x, YI y) { return get(static_cast<X>(x), static_cast<Y>(y)); }

        inline const ElementType& operator[](X x, Y y) const { return get(x, y); }
        template <std::integral XI> inline const ElementType& operator[](XI x, Y y) const { return get(static_cast<X>(x), y); }
        template <std::integral YI> inline const ElementType& operator[](X x, YI y) const { return get(x, static_cast<Y>(y)); }
        template <std::integral XI, std::integral YI> inline const ElementType& operator[](XI x, YI y) const { return get(static_cast<X>(x), static_cast<Y>(y)); }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            auto* ptr = this->data();
            for (size_t i = 0; i < NUM_ELTS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
        }

        size_t storage_size_bytes() const { return NUM_ELTS * sizeof(ElementType); }

      private:
        size_t k(const X x, const Y y) const
        {
            const size_t sx = static_cast<size_t>(x), sy = static_cast<size_t>(y);
            assert(sx < ROWS); assert(sy < COLS); assert(sy <= sx);
            const size_t ki = sx * (sx + 1) / 2 + sy;
            assert(ki < NUM_ELTS);
            return ki;
        }
    };
} // namespace rllm

#endif
