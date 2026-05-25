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

#include <parallel.hpp>

#include <RandomHelpers.hpp>
#include <tokenizer_map.hpp>
#include <Range.hpp>
#include <enum_iterator.hpp>
#include <flexible_cols_matrix.hpp>
#include <flexible_rows_matrix.hpp>
#include <flexible_rows_cols_matrix.hpp>
#include <fixed_size_matrix.hpp>
#include <template_vector.hpp>

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
        MAX = 8
    };

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

    // Index of a parallel processor/thread (0..63).
    enum class ProcessorIndex : size_t
    {
        START = 0,
        MAX = 64
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

    using InputLine = template_vector<TokenID, PositionIndex>;


    struct Score
    {
        template_vector<float, TokenID> values;
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
    };

} // namespace rllm
