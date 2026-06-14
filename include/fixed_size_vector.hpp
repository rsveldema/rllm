#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>

#include <Range.hpp>
#include <enum_iterator1D.hpp>
#include <parallel.hpp>
#include <IMemorySpace.hpp>

#include <offloadable_data.hpp>


namespace rllm
{
    template <typename T, typename LengthType>
    class fixed_size_vector : public offloadable_data<T>
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        fixed_size_vector()
            : offloadable_data<T>(CAPACITY)
        {}

        fixed_size_vector(const fixed_size_vector& other)
            : offloadable_data<T>(CAPACITY)
            , len(other.len)
        {
            this->m_data = other.m_data;
        }

        fixed_size_vector& operator=(const fixed_size_vector& other)
        {
            if (this != &other)
            {
                this->m_data = other.m_data;
                len = other.len;
            }
            return *this;
        }

        fixed_size_vector(fixed_size_vector&& other)
            : offloadable_data<T>(CAPACITY)
            , len(other.len)
        {
            this->m_data = other.m_data;
        }

        fixed_size_vector& operator=(fixed_size_vector&& other)
        {
            if (this != &other)
            {
                this->m_data = other.m_data;
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
            const auto* data = this->m_data.get();
            float max_value = std::numeric_limits<float>::lowest();
            for (const auto i : enum_iterator1D<LengthType>(length))
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
            auto* data = this->m_data.get();
            const auto max_value = get_highest_value(length);

            float sum_exp = 0.0f;
            for (const auto i : enum_iterator1D<LengthType>(length))
            {
                sum_exp += std::exp(data[static_cast<size_t>(i)] - max_value);
            }

            for (const auto i : enum_iterator1D<LengthType>(length))
            {
                data[static_cast<size_t>(i)] = std::exp(data[static_cast<size_t>(i)] - max_value) / sum_exp;
            }
        }

        void sub_array(fixed_size_vector& result, LengthType length) const
        {
            assert(length <= len);
            const auto* data = this->m_data.get();
            result.clear();
            for (const auto i : enum_iterator1D<LengthType>(length))
            {
                const auto tok = data[static_cast<size_t>(i)];
                result.push_back(tok);
            }
            result.len = length;
        }

        void push_back(T value)
        {
            assert(len < LengthType::MAX);
            this->m_data.get()[static_cast<size_t>(len)] = value;
            len = static_cast<LengthType>(static_cast<size_t>(len) + 1);
        }

        const T& back() const
        {
            assert(len > LengthType::START);
            return this->m_data.get()[static_cast<size_t>(len) - 1];
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

        size_t storage_size_bytes() const
        {
            return static_cast<size_t>(len) * sizeof(T);
        }

        T& operator[](LengthType index)
        {
            assert(index < LengthType::MAX);
            return this->m_data.get()[static_cast<size_t>(index)];
        }

        T& operator[](size_t index)
        {
            assert(index < static_cast<size_t>(len));
            return this->m_data.get()[index];
        }

        const T& operator[](LengthType index) const
        {
            assert(index < LengthType::MAX);
            return this->m_data.get()[static_cast<size_t>(index)];
        }

        const T& operator[](size_t index) const
        {
            assert(index < static_cast<size_t>(len));
            return this->m_data.get()[index];
        }

        T get_offload_synced(LengthType index) const
        {
            assert(index < LengthType::MAX);
            return this->m_data.get_offload_synced(static_cast<size_t>(index));
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        LengthType len = LengthType::START;
    };
} // namespace rllm
