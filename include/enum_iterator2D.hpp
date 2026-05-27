#pragma once

#include <cstddef>

namespace rllm
{
    /** iterates from Enum1::START to Enum1::MAX and Enum2::START to Enum2::MAX 
     * Conventiently assumes that Enum1 and Enum2 have contiguous values starting from START and that MAX is one past the last valid value.
    */
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

        // Number of inner (v2) values per outer (v1) step.
        size_t inner_size() const noexcept { return m_size2; }
        // Number of distinct v1 (outer) rows.
        size_t outer_size() const noexcept { return m_end / m_size2; }

        // Lightweight range over one outer row — suitable for range-for.
        struct RowRange
        {
            enum_iterator2D m_begin, m_end;
            enum_iterator2D begin() const noexcept { return m_begin; }
            enum_iterator2D end()   const noexcept { return m_end;   }
        };

        RowRange row_range(size_t outer_idx) const noexcept
        {
            const size_t start = outer_idx * m_size2;
            return {iter_at(start), iter_at(start + m_size2)};
        }

        // Range over a rectangular block [outer_begin, outer_end) x [inner_begin, inner_end).
        // Iterates row-major: for each outer row, scans all inner columns in the block.
        struct BlockRange
        {
            struct iterator
            {
                size_t m_inner_begin, m_inner_end;
                size_t m_oi, m_ii; // current outer/inner indices (relative to Enum::START)

                std::pair<Enum1, Enum2> operator*() const noexcept
                {
                    return {
                        static_cast<Enum1>(static_cast<size_t>(Enum1::START) + m_oi),
                        static_cast<Enum2>(static_cast<size_t>(Enum2::START) + m_ii)
                    };
                }

                iterator& operator++() noexcept
                {
                    if (++m_ii >= m_inner_end)
                    {
                        m_ii = m_inner_begin;
                        ++m_oi;
                    }
                    return *this;
                }

                bool operator!=(const iterator& other) const noexcept
                {
                    return m_oi != other.m_oi || m_ii != other.m_ii;
                }
            };

            size_t m_outer_begin, m_outer_end;
            size_t m_inner_begin, m_inner_end;

            iterator begin() const noexcept
            {
                return {m_inner_begin, m_inner_end, m_outer_begin, m_inner_begin};
            }
            iterator end() const noexcept
            {
                return {m_inner_begin, m_inner_end, m_outer_end, m_inner_begin};
            }
        };

        BlockRange block_range(size_t outer_begin, size_t outer_end,
                               size_t inner_begin, size_t inner_end) const noexcept
        {
            return {outer_begin, outer_end, inner_begin, inner_end};
        }

      private:
        enum_iterator2D iter_at(size_t pos) const noexcept
        {
            return {pos, m_end, m_size2};
        }

        enum_iterator2D(size_t pos, size_t end, size_t size2)
            : m_pos(pos), m_end(end), m_size2(size2)
        {}

        size_t m_pos;    // flat index in [0, m_end)
        size_t m_end;    // total iterations = size1 * size2
        size_t m_size2;  // stride of the inner dimension
    };
} // namespace rllm
