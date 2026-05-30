#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

#include <Range.hpp>
#include <enum_iterator.hpp>

namespace rllm
{
    template <typename T, typename LengthType>
    class fixed_size_vector
    {
      public:
        fixed_size_vector()
        {
            if constexpr (std::is_trivially_default_constructible_v<T>)
                m_data.fill(T{});
            // else: T's own default constructor already initialises each element.
        }

        ~fixed_size_vector() = default;

        float get_highest_value(const LengthType length) const
        {
            float max_value = std::numeric_limits<float>::lowest();
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto val = m_data[static_cast<size_t>(i)];
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
            const auto max_value = get_highest_value(length);

            float sum_exp = 0.0f;
            for (const auto i : enum_iterator<LengthType>(length))
            {
                sum_exp += std::exp(m_data[static_cast<size_t>(i)] - max_value);
            }

            for (const auto i : enum_iterator<LengthType>(length))
            {
                m_data[static_cast<size_t>(i)] = std::exp(m_data[static_cast<size_t>(i)] - max_value) / sum_exp;
            }
        }

        fixed_size_vector sub_array(LengthType length) const
        {
            assert(length <= len);
            fixed_size_vector result;
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto tok = m_data[static_cast<size_t>(i)];
                result.push_back(tok);
            }
            result.len = length;
            return result;
        }

        void push_back(T value)
        {
            assert(len < LengthType::MAX);
            m_data[static_cast<size_t>(len)] = value;
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

        T& operator[](LengthType index)
        {
            return m_data[static_cast<size_t>(index)];
        }

        const T& operator[](LengthType index) const
        {
            return m_data[static_cast<size_t>(index)];
        }

        void fill(T value)
        {
            m_data.fill(value);
        }

        void fill(T value, LengthType length)
        {
            assert(length <= LengthType::MAX);
            for (const auto i : enum_iterator<LengthType>(length))
            {
                m_data[static_cast<size_t>(i)] = value;
            }
        }

        /** add a value to an element at index with clamping */
        void add_with_clamp(LengthType index, T delta, Range<T> range)
        {
            auto& cell = m_data[static_cast<size_t>(index)];
            cell = math::clamp(cell + delta, range.lo, range.hi);
        }

        /** add a value to an element at index without clamping */
        void add_no_clamp(LengthType index, T delta)
        {
            m_data[static_cast<size_t>(index)] += delta;
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        using token_vector_data_t = std::array<T, static_cast<size_t>(LengthType::MAX)>;
        token_vector_data_t m_data;
        LengthType len = LengthType::START;
    };
} // namespace rllm
