#pragma once

#include <cstddef>

namespace rllm
{
    template <typename Enum>
    class enum_iterator1D
    {
      public:
        enum_iterator1D(Enum end = Enum::MAX)
                        : m_start(Enum::START)
                        , m_current(Enum::START)
            , m_end(end)
        {}

        enum_iterator1D(Enum start, Enum end)
                        : m_start(start)
                        , m_current(start)
            , m_end(end)
        {}

        // Number of inner (v2) values per outer (v1) step.
        size_t inner_size() const noexcept { return static_cast<size_t>(m_end) - static_cast<size_t>(m_start); }

        Enum operator*() const
        {
            return m_current;
        }
        enum_iterator1D& operator++()
        {
            m_current = inc(m_current);
            return *this;
        }

        bool operator!=(const enum_iterator1D& other) const
        {
            return m_current != other.m_current;
        }

        void operator+=(size_t offset)
        {
            m_current = static_cast<Enum>((static_cast<size_t>(m_current) + offset) % static_cast<size_t>(m_end));
        }

        int operator-(const enum_iterator1D& other) const
        {
            return static_cast<int>(m_current) - static_cast<int>(other.m_current);
        }

        enum_iterator1D begin() const
        {
            return enum_iterator1D{m_start, m_end};
        }

        enum_iterator1D end() const
        {
            return enum_iterator1D{m_end, m_end};
        }

      private:
                Enum m_start;
        Enum m_current;
        Enum m_end;
    };
} // namespace rllm
