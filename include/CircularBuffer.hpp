#pragma once

#include <array>
#include <cstddef>
#include <cassert>

namespace rllm
{
    template <typename T, size_t N>
    class CircularBuffer
    {
      public:
        void push_back(T value)
        {
            m_data[m_head] = value;
            m_head = (m_head + 1) % N;
            if (m_count < N)
                ++m_count;
        }

        size_t size() const { return m_count; }
        size_t capacity() const { return N; }
        bool empty() const { return m_count == 0; }

        T back() const
        {
            assert(m_count > 0);
            size_t tail = (m_head + N - m_count) % N;
            return m_data[(tail + m_count - 1) % N];
        }

        // Iterates elements from oldest to newest.
        struct iterator
        {
            const CircularBuffer* buf;
            size_t index; // logical index 0..count-1

            T operator*() const
            {
                size_t tail = (buf->m_head + N - buf->m_count) % N;
                return buf->m_data[(tail + index) % N];
            }
            iterator& operator++() { ++index; return *this; }
            bool operator!=(const iterator& other) const { return index != other.index; }
        };

        iterator begin() const { return {this, 0}; }
        iterator end()   const { return {this, m_count}; }

      private:
        std::array<T, N> m_data{};
        size_t m_head  = 0;
        size_t m_count = 0;
    };

} // namespace rllm
