#pragma once

#include <cstddef>
#include <tuple>

namespace rllm
{
    template <typename Enum1, typename Enum2, typename Enum3>
    class enum_iterator3D
    {
      public:
        enum_iterator3D(Enum1 end1 = Enum1::MAX, Enum2 end2 = Enum2::MAX, Enum3 end3 = Enum3::MAX)
            : m_pos(0)
            , m_size1(static_cast<size_t>(end1) - static_cast<size_t>(Enum1::START))
            , m_size2(static_cast<size_t>(end2) - static_cast<size_t>(Enum2::START))
            , m_size3(static_cast<size_t>(end3) - static_cast<size_t>(Enum3::START))
        {}

        std::tuple<Enum1, Enum2, Enum3> operator*() const
        {
            const size_t plane = m_size2 * m_size3;
            return {
                static_cast<Enum1>(static_cast<size_t>(Enum1::START) + m_pos / plane),
                static_cast<Enum2>(static_cast<size_t>(Enum2::START) + (m_pos / m_size3) % m_size2),
                static_cast<Enum3>(static_cast<size_t>(Enum3::START) + m_pos % m_size3)
            };
        }

        enum_iterator3D& operator++()
        {
            ++m_pos;
            return *this;
        }

        bool operator!=(const enum_iterator3D& other) const
        {
            return m_pos != other.m_pos;
        }

        enum_iterator3D begin() const { return {0, m_size1, m_size2, m_size3}; }
        enum_iterator3D end() const { return {m_size1 * m_size2 * m_size3, m_size1, m_size2, m_size3}; }

        size_t outer_size() const noexcept { return m_size1; }
        size_t middle_size() const noexcept { return m_size2; }
        size_t inner_size() const noexcept { return m_size3; }

      private:
        enum_iterator3D(size_t pos, size_t size1, size_t size2, size_t size3)
            : m_pos(pos)
            , m_size1(size1)
            , m_size2(size2)
            , m_size3(size3)
        {}

        size_t m_pos;
        size_t m_size1;
        size_t m_size2;
        size_t m_size3;
    };
} // namespace rllm
