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
#include <IMemorySpace.hpp>


namespace rllm
{

    template <typename T, typename LengthType>
    class fixed_size_vector
    {
      public:
        static constexpr size_t CAPACITY = static_cast<size_t>(LengthType::MAX);

        fixed_size_vector()
            : m_data(1)
            , m_capacity(1)
        {}

        fixed_size_vector(const fixed_size_vector& other)
            : m_data(std::max<size_t>(1, static_cast<size_t>(other.len)))
            , len(other.len)
            , m_capacity(other.m_data.size())
        {
            m_data = other.m_data;
            m_capacity = m_data.size();
        }

        fixed_size_vector& operator=(const fixed_size_vector& other)
        {
            if (this != &other)
            {
                ensure_capacity(static_cast<size_t>(other.len));
                m_data = other.m_data;
                len = other.len;
                m_capacity = m_data.size();
            }
            return *this;
        }

        fixed_size_vector(fixed_size_vector&& other)
            : m_data(std::max<size_t>(1, static_cast<size_t>(other.len)))
            , len(other.len)
            , m_capacity(other.m_data.size())
        {
            m_data = other.m_data;
            m_capacity = m_data.size();
        }

        fixed_size_vector& operator=(fixed_size_vector&& other)
        {
            if (this != &other)
            {
                ensure_capacity(static_cast<size_t>(other.len));
                m_data = other.m_data;
                len = other.len;
                m_capacity = m_data.size();
            }
            return *this;
        }

        ~fixed_size_vector() = default;

        void set_size(LengthType new_size)
        {
            assert(new_size <= LengthType::MAX);
            ensure_capacity(static_cast<size_t>(new_size));
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

        void sub_array(fixed_size_vector& result, LengthType length) const
        {
            assert(length <= len);
            const auto* data = m_data.get();
            result.clear();
            for (const auto i : enum_iterator<LengthType>(length))
            {
                const auto tok = data[static_cast<size_t>(i)];
                result.push_back(tok);
            }
            result.len = length;
        }

        void push_back(T value)
        {
            assert(len < LengthType::MAX);
            ensure_capacity(static_cast<size_t>(len) + 1);
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

        DeviceMemoryOwner device_memory_owner() const
        {
            return m_data.device_memory_owner();
        }

        void set_pending_flush(std::function<void()> flush_fn)
        {
            m_data.set_pending_flush(std::move(flush_fn));
        }

        void mark_device_latest()
        {
            m_data.mark_device_latest();
        }

        void copy_to_offload_buffer()
        {
            m_data.copy_to_offload_buffer();
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

        T get_offload_synced(LengthType index) const
        {
            return m_data.get_offload_synced(static_cast<size_t>(index));
        }

        void zero()
        {
            m_data.zero();
        }

        void clear()
        {
            len = LengthType::START;
        }

      private:
        void ensure_capacity(size_t requested)
        {
            requested = std::max<size_t>(1, requested);
            if (requested <= m_capacity)
                return;

            size_t new_capacity = m_capacity;
            while (new_capacity < requested)
                new_capacity = std::min(CAPACITY, new_capacity * 2);
            m_data.resize(new_capacity);
            m_capacity = new_capacity;
        }

        DevicePointer<T> m_data;
        LengthType len = LengthType::START;
        size_t m_capacity;
    };
} // namespace rllm
