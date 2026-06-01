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
    class fixed_size_vector
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        fixed_size_vector()
            : m_data(make_storage())
        {
            if constexpr (std::is_trivially_default_constructible_v<T>)
                fill(T{});
            // else: T's own default constructor already initialises each element.
        }

        fixed_size_vector(const fixed_size_vector& other)
            : m_data(make_storage())
            , len(other.len)
        {
            m_data = other.m_data;
        }

        fixed_size_vector& operator=(const fixed_size_vector& other)
        {
            if (this != &other)
            {
                m_data = other.m_data;
                len = other.len;
            }
            return *this;
        }

        fixed_size_vector(fixed_size_vector&& other)
            : m_data(make_storage())
            , len(other.len)
        {
            m_data = other.m_data;
        }

        fixed_size_vector& operator=(fixed_size_vector&& other)
        {
            if (this != &other)
            {
                m_data = other.m_data;
                len = other.len;
            }
            return *this;
        }

        ~fixed_size_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            len = new_size;
        }

        float get_highest_value(const LengthType length) const
        {
            const auto* data = m_data.get();
            float max_value = std::numeric_limits<float>::lowest();
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto val = data[static_cast<size_t>(i)];
                if (val > max_value)
                {
                    max_value = val;
                }
            }
            return max_value;
        }

        /** Normalize the values using the softmax function.
         * Each element is <= 1 and >= 0, and the sum of all elements is 1.
         * This is useful for converting the output of the neural network into
         * a probability distribution over the tokens.
         */
        void normalize_using_softmax(const LengthType length)
        {
            auto* data = m_data.get();
            const auto max_value = get_highest_value(length);

            float sum_exp = 0.0f;
            for (const auto i : enum_iterator<LengthType>(length))
            {
                sum_exp += std::exp(data[static_cast<size_t>(i)] - max_value);
            }

            for (const auto i : enum_iterator<LengthType>(length))
            {
                data[static_cast<size_t>(i)] = std::exp(data[static_cast<size_t>(i)] - max_value) / sum_exp;
            }
        }

        fixed_size_vector sub_array(LengthType length) const
        {
            assert(length <= len);
            const auto* data = m_data.get();
            fixed_size_vector result;
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto tok = data[static_cast<size_t>(i)];
                result.push_back(tok);
            }
            result.len = length;
            return result;
        }

        void push_back(T value)
        {
            assert(len < LengthType::MAX);
            m_data.get()[static_cast<size_t>(len)] = value;
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        const T& back() const
        {
            assert(len > LengthType::START);
            return m_data.get()[static_cast<size_t>(len) - 1];
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
            return m_data.get();
        }

        const T* data() const
        {
            return m_data.get();
        }

        T* raw_staging_data() const
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
            return static_cast<size_t>(len) * sizeof(T);
        }

        T& operator[](LengthType index)
        {
            return m_data.get()[static_cast<size_t>(index)];
        }

        T& operator[](size_t index)
        {
            assert(index < static_cast<size_t>(len));
            return m_data.get()[index];
        }

        const T& operator[](LengthType index) const
        {
            return m_data.get()[static_cast<size_t>(index)];
        }

        const T& operator[](size_t index) const
        {
            assert(index < static_cast<size_t>(len));
            return m_data.get()[index];
        }

        void fill(T value)
        {
            std::fill_n(m_data.get(), CAPACITY, value);
        }

        void fill(T value, LengthType length)
        {
            assert(length <= LengthType::MAX);
            auto* data = m_data.get();
            for (const auto i : enum_iterator<LengthType>(length))
            {
                data[static_cast<size_t>(i)] = value;
            }
        }

        void zero(LengthType length)
        {
            assert(length <= LengthType::MAX);
            auto* data = m_data.get();
            for (const auto i : enum_iterator<LengthType>(length))
            {
                data[static_cast<size_t>(i)] = 0;
            }
        }

        void zero()
        {
            m_data.zero();
        }

        /** add a value to an element at index with clamping */
        void add_with_clamp(LengthType index, T delta, Range<T> range)
        {
            auto& cell = m_data.get()[static_cast<size_t>(index)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        /** add a value to an element at index without clamping */
        void add_no_clamp(LengthType index, T delta)
        {
            m_data.get()[static_cast<size_t>(index)] += delta;
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        static DevicePointer<T> make_storage()
        {
            if constexpr (std::is_trivially_copyable_v<T>)
                return DevicePointer<T>(CAPACITY);
            else
                return DevicePointer<T>(new T[CAPACITY], CAPACITY, true);
        }

        DevicePointer<T> m_data;
        LengthType len = LengthType::START;
    };
} // namespace rllm
