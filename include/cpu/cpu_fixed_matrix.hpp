#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>

#include <RandomHelpers.hpp>
#include <cpu/cpu_data.hpp>

namespace rllm
{
    /** Fixed-size 2-D matrix backed by plain host memory (CPU-only). */
    template <typename ElementType, typename X, typename Y>
    class cpu_fixed_matrix : public cpu_data<ElementType>
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        cpu_fixed_matrix()
            : cpu_data<ElementType>(ROWS * COLS)
        {}

        cpu_fixed_matrix(const cpu_fixed_matrix& other)
            : cpu_data<ElementType>(ROWS * COLS)
        {
            assign_from(other);
        }

        cpu_fixed_matrix& operator=(const cpu_fixed_matrix& other)
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        cpu_fixed_matrix(cpu_fixed_matrix&& other) noexcept
            : cpu_data<ElementType>(ROWS * COLS)
        {
            assign_from(other);
        }

        cpu_fixed_matrix& operator=(cpu_fixed_matrix&& other) noexcept
        {
            if (this != &other)
                assign_from(other);
            return *this;
        }

        ~cpu_fixed_matrix() = default;

        inline void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            this->data()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        inline void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        inline const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return this->data()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        inline const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        inline ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return this->data()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        template <std::integral XIndex>
        inline ElementType& operator[](XIndex x, Y y) { return (*this)[static_cast<X>(x), y]; }
        template <std::integral YIndex>
        inline ElementType& operator[](X x, YIndex y) { return (*this)[x, static_cast<Y>(y)]; }
        template <std::integral XIndex, std::integral YIndex>
        inline ElementType& operator[](XIndex x, YIndex y) { return (*this)[static_cast<X>(x), static_cast<Y>(y)]; }

        inline const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return this->data()[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        template <std::integral XIndex>
        inline const ElementType& operator[](XIndex x, Y y) const { return (*this)[static_cast<X>(x), y]; }
        template <std::integral YIndex>
        inline const ElementType& operator[](X x, YIndex y) const { return (*this)[x, static_cast<Y>(y)]; }
        template <std::integral XIndex, std::integral YIndex>
        inline const ElementType& operator[](XIndex x, YIndex y) const { return (*this)[static_cast<X>(x), static_cast<Y>(y)]; }

        inline void fill_rand(ElementType lo, ElementType hi)
        {
            auto* ptr = this->data();
            for (size_t i = 0; i < ROWS * COLS; ++i)
                ptr[i] = static_cast<ElementType>(get_random_value(lo, hi));
        }

        template <typename Out>
        void export_row(X x, Out& out) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            auto* src = this->data() + static_cast<size_t>(x) * COLS;
            for (size_t i = 0; i < COLS; ++i)
                out[i] = src[i];
        }

        size_t storage_size_bytes() const
        {
            return ROWS * COLS * sizeof(ElementType);
        }

      private:
        void assign_from(const cpu_fixed_matrix& other)
        {
            std::copy(other.data(), other.data() + ROWS * COLS, this->data());
        }
    };
} // namespace rllm
