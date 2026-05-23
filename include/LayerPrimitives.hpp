#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <print>

namespace rllm
{
    static constexpr float MIN_NEURON_INPUT = -0.01f;
    static constexpr float MAX_NEURON_INPUT = 1.0f;

    static constexpr auto RED   = "\033[31m";
    static constexpr auto RESET = "\033[0m";

#define LOG_ONCE(...) do { \
    static int counter = 0;\
    if (counter < 3) { \
        __VA_ARGS__; \
        ++counter; \
    } \
} while(0)


    template <typename Enum>
    class enum_iterator
    {
      public:
        enum_iterator(Enum end = Enum::MAX)
            : m_current(Enum::START)
            , m_end(end)
        {}

        enum_iterator(Enum start, Enum end)
            : m_current(start)
            , m_end(end)
        {}

        Enum operator*() const
        {
            return m_current;
        }
        enum_iterator& operator++()
        {
            m_current = inc(m_current);
            return *this;
        }

        bool operator!=(const enum_iterator& other) const
        {
            return m_current != other.m_current;
        }

        void operator += (size_t offset)
        {
            m_current = static_cast<Enum>((static_cast<size_t>(m_current) + offset) % static_cast<size_t>(m_end));
        }

        int operator -(const enum_iterator& other) const
        {
            return static_cast<int>(m_current) - static_cast<int>(other.m_current);
        }

        enum_iterator begin() const
        {
            return enum_iterator{Enum::START, m_end};
        }

        enum_iterator end() const
        {
            return enum_iterator{m_end, m_end};
        }

      private:
        Enum m_current;
        Enum m_end;
    };

    enum class TokenID : int32_t
    {
        UNKNOWN_TOKEN_ID = -1,
        START = 0,
        MAX = 1024 * 2
    };

    static inline TokenID inc(TokenID id)
    {
        assert(id != TokenID::UNKNOWN_TOKEN_ID);
        assert(id < TokenID::MAX);
        return static_cast<TokenID>(static_cast<int32_t>(id) + 1);
    }


    using Token = std::string;

    // position of a token in the input sequence. For example, in the input "the cat sat", the token "cat" has
    // position 1.
    enum class PositionIndex : size_t
    {
        START = 0,
        MAX = 128,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // position of a neuron in the intermediate layer. For example, in the intermediate layer, neuron 0 is connected
    // to token 0 in the input layer, neuron 1 is connected to token 1 in the input layer, and so on.
    enum class IntermediateLayerIndex : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(TokenID::MAX) * 16,
        UNKNOWN_INTERMEDIATE_LAYER_INDEX = static_cast<size_t>(-1)
    };

    static inline PositionIndex inc(PositionIndex id)
    {
        assert(id != PositionIndex::UNKNOWN_POSITION_INDEX);
        assert(id < PositionIndex::MAX);
        return static_cast<PositionIndex>(static_cast<int32_t>(id) + 1);
    }

    static inline IntermediateLayerIndex inc(IntermediateLayerIndex id)
    {
        assert(id != IntermediateLayerIndex::UNKNOWN_INTERMEDIATE_LAYER_INDEX);
        assert(id < IntermediateLayerIndex::MAX);
        return static_cast<IntermediateLayerIndex>(static_cast<int32_t>(id) + 1);
    }

    template <typename T>
    struct Range
    {
        T lo = T{0};
        T hi = T{1};
    };

    template <typename T, typename LengthType>
    class template_token_vector
    {
      public:
        template_token_vector()
        {
            m_data.fill(T{});
        }
        ~template_token_vector() = default;

        template_token_vector substr(LengthType length) const
        {
            assert(length <= len);
            template_token_vector result;
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

        /** add a value to an element at index with clamping */
        void add_with_clamp(LengthType index, T delta, Range<T> range)
        {
            auto& cell = m_data[static_cast<size_t>(index)];
            cell = std::clamp(cell + delta, range.lo, range.hi);
        }

        /** add a value to an element at index without clamping */
        void add_no_clamp(LengthType index, T delta)
        {
            m_data[static_cast<size_t>(index)] += delta;
        }

      private:
        using token_vector_data_t = std::array<T, static_cast<size_t>(LengthType::MAX)>;
        token_vector_data_t m_data;
        LengthType len = LengthType::START;
    };

    using InputLine = template_token_vector<TokenID, PositionIndex>;


    template <typename T, typename X, typename Y>
    class template_token_matrix
    {
      public:
        template_token_matrix()
        {
            for (auto& row : m_data)
            {
                row.fill(T{});
            }
        }
        ~template_token_matrix() = default;

        void set(const X x, const Y y, T value)
        {
            auto& inner_data = m_data[static_cast<size_t>(x)];
            inner_data[static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, T value)
        {
            set(indices.first, indices.second, value);
        }

        const T& get(const X x, const Y y) const
        {
            return m_data[static_cast<size_t>(x)][static_cast<size_t>(y)];
        }

        const T& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        void fill(T value)
        {
            for (auto& row : m_data)
            {
                row.fill(value);
            }
        }

        void add_with_clamp(const X x, const Y y, T delta, Range<T> range)
        {
            auto& cell = m_data[static_cast<size_t>(x)][static_cast<size_t>(y)];
            cell = std::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, T delta, Range<T> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, T delta)
        {
            m_data[static_cast<size_t>(x)][static_cast<size_t>(y)] += delta;
        }

      private:
        using inner_array_t = std::array<T, static_cast<size_t>(Y::MAX)>;
        using matrix_data_t = std::array<inner_array_t, static_cast<size_t>(X::MAX)>;
        matrix_data_t m_data;
    };

    struct Score
    {
        template_token_vector<float, TokenID> values;
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
    };

} // namespace rllm
