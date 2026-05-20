#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

namespace rllm
{
    enum class TokenID : int32_t
    {
        UNKNOWN_TOKEN_ID = -1,
        START = 0,
        MAX = 4096
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
        MAX = 1024 * 8,
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
    

    template <typename T, typename LengthType>
    class template_token_vector
    {
      public:
        template_token_vector()
        {
            m_data.fill(T{});
        }
        ~template_token_vector() = default;

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
        template_token_matrix(const template_token_matrix&) = delete;

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
        float weight;
    };

} // namespace rllm
