#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include <tokenizer_map.hpp>
#include <RandomHelpers.hpp>

namespace rllm
{
    static constexpr float MIN_NEURON_INPUT = -0.01f;
    static constexpr float MAX_NEURON_INPUT = 1.0f;

    static constexpr auto RED = "\033[31m";
    static constexpr auto RESET = "\033[0m";

#define LOG_ONCE(...) \
    do \
    { \
        static int counter = 0; \
        if (counter < 3) \
        { \
            __VA_ARGS__; \
            ++counter; \
        } \
    } while (0)


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

        void operator+=(size_t offset)
        {
            m_current = static_cast<Enum>((static_cast<size_t>(m_current) + offset) % static_cast<size_t>(m_end));
        }

        int operator-(const enum_iterator& other) const
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


    using Token = std::string;

    // Dimensionality of each token's learned embedding vector.
    // The first intermediate layer is tiled as: neuron[p * EmbeddingDimension::MAX + d] for position p, dimension d.
    enum class EmbeddingDimension : size_t
    {
        START = 0,
        MAX = 512
    };

    // position of a token in the input sequence. For example, in the input "the cat sat", the token "cat" has
    // position 1.
    enum class PositionIndex : size_t
    {
        START = 0,
        MAX = 128,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // Index of an attention head (0..NUM_HEADS-1).
    enum class HeadsIndex : size_t
    {
        START = 0,
        MAX   = 8
    };

    // Per-head embedding dimension: EmbeddingDimension::MAX / HeadsIndex::MAX = 64.
    enum class HeadDimension : size_t
    {
        START = 0,
        MAX   = static_cast<size_t>(EmbeddingDimension::MAX) / static_cast<size_t>(HeadsIndex::MAX)
    };

    // Feed-forward hidden dimension: static_cast<int>(FFDimension::MAX) = 4 × EmbeddingDimension::MAX.
    enum class FFDimension : size_t
    {
        START = 0,
        MAX   = static_cast<size_t>(EmbeddingDimension::MAX) * 4
    };

    // position of a neuron in the intermediate layer. For example, in the intermediate layer, neuron 0 is connected
    // to token 0 in the input layer, neuron 1 is connected to token 1 in the input layer, and so on.
    enum class IntermediateLayerIndex : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(EmbeddingDimension::MAX) * static_cast<size_t>(PositionIndex::MAX),
        UNKNOWN_INTERMEDIATE_LAYER_INDEX = static_cast<size_t>(-1)
    };

    static inline TokenID inc(TokenID id)
    {
        assert(id != TokenID::UNKNOWN_TOKEN_ID);
        assert(id < TokenID::MAX);
        return static_cast<TokenID>(static_cast<int32_t>(id) + 1);
    }

    static inline EmbeddingDimension inc(EmbeddingDimension id)
    {
        assert(id < EmbeddingDimension::MAX);
        return static_cast<EmbeddingDimension>(static_cast<size_t>(id) + 1);
    }

    static inline PositionIndex inc(PositionIndex id)
    {
        assert(id != PositionIndex::UNKNOWN_POSITION_INDEX);
        assert(id < PositionIndex::MAX);
        return static_cast<PositionIndex>(static_cast<int32_t>(id) + 1);
    }

    static inline PositionIndex dec(PositionIndex id)
    {
        assert(id != PositionIndex::UNKNOWN_POSITION_INDEX);
        assert(id < PositionIndex::MAX);
        assert(id > PositionIndex::START);
        return static_cast<PositionIndex>(static_cast<int32_t>(id) - 1);
    }

    static inline HeadsIndex inc(HeadsIndex id)
    {
        assert(id < HeadsIndex::MAX);
        return static_cast<HeadsIndex>(static_cast<size_t>(id) + 1);
    }

    static inline HeadDimension inc(HeadDimension id)
    {
        assert(id < HeadDimension::MAX);
        return static_cast<HeadDimension>(static_cast<size_t>(id) + 1);
    }

    static inline FFDimension inc(FFDimension id)
    {
        assert(id < FFDimension::MAX);
        return static_cast<FFDimension>(static_cast<size_t>(id) + 1);
    }

    static inline IntermediateLayerIndex inc(IntermediateLayerIndex id)
    {
        assert(id != IntermediateLayerIndex::UNKNOWN_INTERMEDIATE_LAYER_INDEX);
        assert(id < IntermediateLayerIndex::MAX);
        return static_cast<IntermediateLayerIndex>(static_cast<int32_t>(id) + 1);
    }

    // Index of an outgoing connection slot within a single neuron's connection list.
    enum class NeuronConnectionIndex : size_t
    {
        START = 0,
        MAX = 128
    };

    static inline NeuronConnectionIndex inc(NeuronConnectionIndex id)
    {
        assert(id < NeuronConnectionIndex::MAX);
        return static_cast<NeuronConnectionIndex>(static_cast<size_t>(id) + 1);
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

        void clear()
        {
            len = LengthType::START;
        }

      private:
        using token_vector_data_t = std::array<T, static_cast<size_t>(LengthType::MAX)>;
        token_vector_data_t m_data;
        LengthType len = LengthType::START;
    };

    using InputLine = template_token_vector<TokenID, PositionIndex>;


    template <typename ElementType, typename X, typename Y>
    class flexible_size_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        flexible_size_matrix()
        : m_rows(X::MAX), m_cols(Y::MAX)
        {
            m_data.fill(ElementType{});
        }

        flexible_size_matrix(X rows, Y cols)
        : m_rows(rows), m_cols(cols)
        {
            m_data.fill(ElementType{});
        }
        ~flexible_size_matrix() = default;

        void set_size(X rows, Y cols)
        {
            assert(static_cast<size_t>(rows) <= ROWS);
            assert(static_cast<size_t>(cols) <= COLS);
            m_rows = rows;
            m_cols = cols;
        }

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            return m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        // Adds each element of other (must have the same runtime dimensions) into this matrix.
        void element_wise_add(const flexible_size_matrix& other)
        {
            assert(m_rows == other.m_rows && m_cols == other.m_cols);
            const size_t n = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
#pragma omp simd
            for (size_t i = 0; i < n; ++i)
                m_data[i] += other.m_data[i];
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            auto& cell = m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)];
            cell = std::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < static_cast<size_t>(m_rows));
            assert(static_cast<size_t>(y) < static_cast<size_t>(m_cols));
            m_data[static_cast<size_t>(x) * static_cast<size_t>(m_cols) + static_cast<size_t>(y)] += delta;
        }

        // Flat pointer access for BLAS-style routines.  Layout: row-major [ROWS × COLS].
        ElementType*       data()       { return m_data.data(); }
        const ElementType* data() const { return m_data.data(); }

        X num_rows() const { return m_rows; }
        Y num_cols() const { return m_cols; }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
        X m_rows;
        Y m_cols;
    };

    template <typename ElementType, typename X, typename Y>
    class fixed_size_matrix
    {
      public:
        static constexpr size_t ROWS = static_cast<size_t>(X::MAX);
        static constexpr size_t COLS = static_cast<size_t>(Y::MAX);

        fixed_size_matrix()
        {
            m_data.fill(ElementType{});
        }
        ~fixed_size_matrix() = default;

        void set(const X x, const Y y, ElementType value)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] = value;
        }

        void set(const std::pair<const X, const Y>& indices, ElementType value)
        {
            set(indices.first, indices.second, value);
        }

        const ElementType& get(const X x, const Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& get(const std::pair<const X, const Y>& indices) const
        {
            return get(indices.first, indices.second);
        }

        ElementType& operator[](X x, Y y)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        const ElementType& operator[](X x, Y y) const
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            return m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
        }

        void fill(ElementType value)
        {
            m_data.fill(value);
        }

        void fill_rand(ElementType lo, ElementType hi)
        {
            for (auto& v : m_data)
                v = get_random_value(lo, hi);
        }

        void add_with_clamp(const X x, const Y y, ElementType delta, Range<ElementType> range)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            auto& cell = m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)];
            cell = std::clamp(cell + delta, range.lo, range.hi);
        }

        void add_with_clamp(const std::pair<const X, const Y>& indices, ElementType delta, Range<ElementType> range)
        {
            add_with_clamp(indices.first, indices.second, delta, range);
        }

        void add_no_clamp(const X x, const Y y, ElementType delta)
        {
            assert(static_cast<size_t>(x) < ROWS);
            assert(static_cast<size_t>(y) < COLS);
            m_data[static_cast<size_t>(x) * COLS + static_cast<size_t>(y)] += delta;
        }

      private:
        using flat_data_t = std::array<ElementType, ROWS * COLS>;
        flat_data_t m_data;
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
