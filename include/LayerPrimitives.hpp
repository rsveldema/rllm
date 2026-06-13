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

#include <math_utils.hpp>

#include <RandomHelpers.hpp>
#include <tokenizer_map.hpp>
#include <Range.hpp>
#include <enum_iterator.hpp>
#include <flexible_cols_matrix.hpp>
#include <flexible_rows_matrix.hpp>
#include <flexible_rows_cols_matrix.hpp>
#include <fixed_size_matrix.hpp>
#include <fixed_size_vector.hpp>


namespace rllm
{
    //using rlmm_float_small = _Float16;
    using rlmm_float_small = float;
    using rlmm_float = float;
    static constexpr rlmm_float RLMM_ZERO = rlmm_float{0};
    static constexpr rlmm_float RLMM_ONE = rlmm_float{1};
    static constexpr rlmm_float RLMM_NEG_ONE = rlmm_float{-1};

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


    using Token = std::string;

    // Dimensionality of each token's learned embedding vector.
    // The first intermediate layer is tiled as: neuron[p * EmbeddingDimension::MAX + d] for position p, dimension d.
    enum class EmbeddingDimension : size_t
    {
        START = 0,
        MAX = 1024
    };

    // position of a token in the input sequence. For example, in the input "the cat sat", the token "cat" has
    // position 1.
    enum class PositionIndex : size_t
    {
        START = 0,
        MAX = 1024 * 16,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // Index of an attention head (0..HeadsIndex::MAX-1).
    enum class HeadsIndex : size_t
    {
        START = 0,
        MAX = 8
    };

    // we are predicting N next tokens in parallel,
    // so we have N parallel sets of attention heads and N parallel output tokens.
    enum class MultiTokenPredictionIndex : size_t
    {
        START = 0,
        ONE   = 1,
        TWO   = 2,
        THREE = 3,
        FOUR  = 4,
        MAX   = 4
    };

    enum class RmsNormPartialSumIndex : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(PositionIndex::MAX) * static_cast<size_t>(EmbeddingDimension::MAX)
    };

    static inline MultiTokenPredictionIndex inc(MultiTokenPredictionIndex id)
    {
        assert(id < MultiTokenPredictionIndex::MAX);
        return static_cast<MultiTokenPredictionIndex>(static_cast<size_t>(id) + 1);
    }

    // Per-head embedding dimension: EmbeddingDimension::MAX / HeadsIndex::MAX = 64.
    enum class HeadDimension : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(EmbeddingDimension::MAX) / static_cast<size_t>(HeadsIndex::MAX)
    };

    // Feed-forward hidden dimension: static_cast<int>(FFDimension::MAX) = 4 × EmbeddingDimension::MAX.
    enum class FFDimension : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(EmbeddingDimension::MAX) * 4
    };

    enum class TempStorage : size_t
    {
        START = 0,
        ZERO = 0,
        ONE = 1,
        MAX = 2 // defines an array with up-to M temp variables in a kernel
    };

    static inline TempStorage inc(TempStorage id)
    {
        assert(id < TempStorage::MAX);
        return static_cast<TempStorage>(static_cast<size_t>(id) + 1);
    }

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

    struct ConflictingToken { TokenID tok; PositionIndex pos; };

    enum class ConflictIndex : size_t {
        START = 0,
        MAX = 256
    };

    static inline ConflictIndex inc(ConflictIndex id)
    {
        assert(id < ConflictIndex::MAX);
        return static_cast<ConflictIndex>(static_cast<size_t>(id) + 1);
    }

    class InputLine : public fixed_size_vector<TokenID, PositionIndex>
    {
      public:
        using Base = fixed_size_vector<TokenID, PositionIndex>;
        using Base::Base;

        InputLine() = default;
        InputLine(const Base& other)
            : Base(other)
        {}

        InputLine& operator=(const Base& other)
        {
            Base::operator=(other);
            return *this;
        }

        void sub_array(InputLine& result, PositionIndex length) const
        {
            Base::sub_array(result, length);
        }

        uint64_t hash() const
        {
            constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ull;
            constexpr uint64_t FNV_PRIME = 1099511628211ull;

            uint64_t hash = FNV_OFFSET_BASIS;
            for (const auto i : enum_iterator<PositionIndex>(size()))
            {
                uint64_t value = static_cast<uint64_t>(static_cast<int>(operator[](i))) + 1ull;
                for (int byte = 0; byte < 8; ++byte)
                {
                    hash ^= (value & 0xffull);
                    hash *= FNV_PRIME;
                    value >>= 8;
                }
            }
            return hash;
        }
    };

    class InputLineView
    {
        public:
            InputLineView(const InputLine& data, 
                PositionIndex start,    
                PositionIndex length)
                : m_data(data)
                , m_start(start)
                , m_length(length)
            {}
    
            const TokenID& operator[](PositionIndex index) const
            {
                assert(((int)index+(int)m_start) < (int)m_length);
                return m_data[static_cast<size_t>(m_start) + static_cast<size_t>(index)];
            }
    
            PositionIndex size() const
            {
                return m_length;
            }
    
        private:
            const InputLine& m_data;
            PositionIndex m_start;
            PositionIndex m_length;
    };


    struct Score
    {
        Score()
        {
            values.set_size(TokenID::MAX);
            temp_values.set_size(TempStorage::MAX);
        }

        void reset()
        {
            values.zero();
            temp_values.zero();
        }

        fixed_size_vector<rlmm_float, TokenID> values;
        fixed_size_vector<rlmm_float, TempStorage> temp_values; // for use in softmax computation, to avoid modifying the original logits
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
    };

} // namespace rllm
