#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>

#include <Range.hpp>
#include <enum_iterator.hpp>
#include <parallel.hpp>


namespace rllm
{
    template <typename T, typename LengthType>
    class fixed_size_obj_vector
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        fixed_size_obj_vector() = default;
        fixed_size_obj_vector(const fixed_size_obj_vector&) = default;
        fixed_size_obj_vector& operator=(const fixed_size_obj_vector&) = default;
        fixed_size_obj_vector(fixed_size_obj_vector&&) noexcept = default;
        fixed_size_obj_vector& operator=(fixed_size_obj_vector&&) noexcept = default;
        ~fixed_size_obj_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            len = new_size;
        }

        void push_back(const T& value)
        {
            assert(len < LengthType::MAX);
            m_data[static_cast<size_t>(len)] = value;
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        void push_back(T&& value)
        {
            assert(len < LengthType::MAX);
            m_data[static_cast<size_t>(len)] = std::move(value);
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        const T& back() const
        {
            assert(len > LengthType::START);
            return m_data[static_cast<size_t>(len) - 1];
        }

        void pop_back()
        {
            assert(len > LengthType::START);
            len = static_cast<LengthType>(static_cast<size_t>(len) - 1);
        }

        bool empty() const
        {
            return len == LengthType::START;
        }

        LengthType size() const
        {
            return len;
        }

        T* data()
        {
            return m_data.data();
        }

        const T* data() const
        {
            return m_data.data();
        }

        T& operator[](LengthType index)
        {
            return m_data[static_cast<size_t>(index)];
        }

        T& operator[](size_t index)
        {
            assert(index < static_cast<size_t>(len));
            return m_data[index];
        }

        const T& operator[](LengthType index) const
        {
            return m_data[static_cast<size_t>(index)];
        }

        const T& operator[](size_t index) const
        {
            assert(index < static_cast<size_t>(len));
            return m_data[index];
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        std::array<T, CAPACITY> m_data{};
        LengthType len = LengthType::START;
    };

} // namespace rllm