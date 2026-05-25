#pragma once

#include <cstddef>

namespace rllm
{
    template <typename Enum1, typename Enum2>
    class enum_iterator2D
    {
      public:
        enum_iterator2D(Enum1 end1 = Enum1::MAX, Enum2 end2 = Enum2::MAX)
            : m_pos(0)
            , m_end((static_cast<size_t>(end1) - static_cast<size_t>(Enum1::START)) *
                    (static_cast<size_t>(end2) - static_cast<size_t>(Enum2::START)))
            , m_size2(static_cast<size_t>(end2) - static_cast<size_t>(Enum2::START))
        {}

        std::pair<Enum1, Enum2> operator*() const
        {
            return {
                static_cast<Enum1>(static_cast<size_t>(Enum1::START) + m_pos / m_size2),
                static_cast<Enum2>(static_cast<size_t>(Enum2::START) + m_pos % m_size2)
            };
        }

        enum_iterator2D& operator++()
        {
            ++m_pos;
            return *this;
        }

        bool operator!=(const enum_iterator2D& other) const
        {
            return m_pos != other.m_pos;
        }

        void operator+=(size_t offset)
        {
            m_pos = (m_pos + offset) % m_end;
        }

        int operator-(const enum_iterator2D& other) const
        {
            return static_cast<int>(m_pos) - static_cast<int>(other.m_pos);
        }

        enum_iterator2D begin() const { return {0,     m_end, m_size2}; }
        enum_iterator2D end()   const { return {m_end, m_end, m_size2}; }

      private:
        enum_iterator2D(size_t pos, size_t end, size_t size2)
            : m_pos(pos), m_end(end), m_size2(size2)
        {}

        size_t m_pos;    // flat index in [0, m_end)
        size_t m_end;    // total iterations = size1 * size2
        size_t m_size2;  // stride of the inner dimension
    };
} // namespace rllm
