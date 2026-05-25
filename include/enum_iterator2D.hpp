#pragma once

#include <cstddef>

namespace rllm
{
    template <typename Enum1, typename Enum2>
    class enum_iterator2D
    {
      public:
        enum_iterator2D(Enum1 end1 = Enum1::MAX, Enum2 end2 = Enum2::MAX)
            : m_current1(Enum1::START)
            , m_current2(Enum2::START)
            , m_end1(end1)
            , m_end2(end2)
        {}

        enum_iterator2D(Enum1 start1, Enum1 end1, Enum2 start2, Enum2 end2)
            : m_current1(start1)
            , m_current2(start2)
            , m_end1(end1)
            , m_end2(end2)
        {}

        std::pair<Enum1, Enum2> operator*() const
        {
            return {m_current1, m_current2};
        }
        enum_iterator2D& operator++()
        {
            m_current2 = inc(m_current2);
            if (m_current2 == m_end2)
            {
                m_current2 = Enum2::START;
                m_current1 = inc(m_current1);
            }
            return *this;
        }

        bool operator!=(const enum_iterator2D& other) const
        {
            return m_current1 != other.m_current1 || m_current2 != other.m_current2;
        }

        void operator+=(size_t offset)
        {
            const size_t total_size = static_cast<size_t>(m_end1) * static_cast<size_t>(m_end2);
            const size_t current_pos = static_cast<size_t>(m_current1) * static_cast<size_t>(m_end2) + static_cast<size_t>(m_current2);
            const size_t new_pos = (current_pos + offset) % total_size;
            m_current1 = static_cast<Enum1>(new_pos / static_cast<size_t>(m_end2));
            m_current2 = static_cast<Enum2>(new_pos % static_cast<size_t>(m_end2));
        }

        int operator-(const enum_iterator2D& other) const
        {
            const size_t total_size = static_cast<size_t>(m_end1) * static_cast<size_t>(m_end2);
            const size_t current_pos = static_cast<size_t>(m_current1) * static_cast<size_t>(m_end2) + static_cast<size_t>(m_current2);
            const size_t other_pos = static_cast<size_t>(other.m_current1) * static_cast<size_t>(m_end2) + static_cast<size_t>(other.m_current2);
            return static_cast<int>((current_pos + total_size - other_pos) % total_size);
        }

        enum_iterator2D begin() const
        {
            return enum_iterator2D{Enum1::START, m_end1, Enum2::START, m_end2};
        }

        enum_iterator2D end() const
        {
            return enum_iterator2D{m_end1, m_end1, m_end2, m_end2};
        }

      private:
        Enum1 m_current1;
        Enum2 m_current2;
        Enum1 m_end1;
        Enum2 m_end2;
    };
} // namespace rllm
