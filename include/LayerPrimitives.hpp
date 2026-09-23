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
#include <Range.hpp>
#include <enum_iterator1D.hpp>
#include <fixed_size_matrix.hpp>
#include <fixed_size_vector.hpp>
#include <flexible_cols_matrix.hpp>
#include <flexible_rows_cols_matrix.hpp>
#include <flexible_rows_matrix.hpp>
#include <tokenizer_map.hpp>


#include "rllm_type_aliases.hpp"


namespace rllm
{
#ifndef RLLM_MAX_POSITION
#define RLLM_MAX_POSITION 8192
#endif

#ifndef RLLM_MAX_ATTENTION_POSITION
#define RLLM_MAX_ATTENTION_POSITION 256
#endif
#ifndef RLLM_MAX_BATCH_SIZE
#define RLLM_MAX_BATCH_SIZE (RLLM_MAX_POSITION / RLLM_MAX_ATTENTION_POSITION)
#endif
#ifndef RLLM_MTP_HEAD_COUNT
#define RLLM_MTP_HEAD_COUNT 1
#endif

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

    enum class TokenStringCategory : uint8_t
    {
        None,
        Local,
        Parameter,
        Field,
        Loop,
        Global,
        ClassName,
        String,
        Integer,
        Float
    };

    static constexpr size_t NO_STRING_TABLE_INDEX = static_cast<size_t>(-1);

    static inline TokenStringCategory token_string_category(TokenID token)
    {
        const auto it = tokenizer_map.find(token);
        if (it == tokenizer_map.end() || it->second.str == nullptr)
            return TokenStringCategory::None;
        const std::string_view text = it->second.str;
        if (text == "<FIELD>")
            return TokenStringCategory::Field;
        if (text == "<STRING>")
            return TokenStringCategory::String;
        if (text == "<INTEGER>")
            return TokenStringCategory::Integer;
        if (text == "<FLOAT>")
            return TokenStringCategory::Float;
        if (text == "<LOCAL>")
            return TokenStringCategory::Local;
        if (text == "<PARAM>")
            return TokenStringCategory::Parameter;
        if (text == "<GLOBAL>")
            return TokenStringCategory::Global;
        if (text == "<CLASS_NAME>")
            return TokenStringCategory::ClassName;
        if (text == "<LOOP>")
            return TokenStringCategory::Loop;
        return TokenStringCategory::None;
    }

    static inline bool is_string_table_index_token(TokenID token)
    {
        return token == TokenID::STRING_TABLE_INDEX;
    }

    static inline bool is_identifier_category_token(TokenID token)
    {
        const auto category = token_string_category(token);
        return category != TokenStringCategory::None &&
            category != TokenStringCategory::String &&
            category != TokenStringCategory::Integer &&
            category != TokenStringCategory::Float;
    }

    // Dimensionality of each token's learned embedding vector.
    // The first intermediate layer is tiled across multiple attention heads,
    // so the embedding dimension must be divisible by the number of heads.
    enum class EmbeddingDimension : size_t
    {
        START = 0,
        MAX = 1024 * 2
    };

    enum class IdentifierHashBucket : size_t
    {
        START = 0,
        MAX = 512
    };

    enum class IdentifierNgramSlot : size_t
    {
        START = 0,
        MAX = 16
    };

    static inline IdentifierHashBucket inc(IdentifierHashBucket id)
    {
        assert(id < IdentifierHashBucket::MAX);
        return static_cast<IdentifierHashBucket>(static_cast<size_t>(id) + 1);
    }

    static inline IdentifierNgramSlot inc(IdentifierNgramSlot id)
    {
        assert(id < IdentifierNgramSlot::MAX);
        return static_cast<IdentifierNgramSlot>(static_cast<size_t>(id) + 1);
    }

    // position of a token in the input sequence. For example, in the input "the cat sat", the token "cat" has
    // position 1.
    enum class PositionIndex : size_t
    {
        START = 0,
        MAX = RLLM_MAX_POSITION,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // Local key position within one independently attended sequence. Packed
    // batches share PositionIndex rows but never allocate cross-window scores.
    enum class AttentionPositionIndex : size_t
    {
        START = 0,
        MAX = RLLM_MAX_ATTENTION_POSITION
    };

    // Index of an attention head (0..HeadsIndex::MAX-1).
    // this is used to partition the embedding dimension into multiple heads,
    // each of which has its own attention mechanism.
    enum class HeadsIndex : size_t
    {
        START = 0,
        MAX = 32
    };

    // Batch axis for batched forward/backward paths.
    enum class BatchIndex : size_t
    {
        START = 0,
        MAX = RLLM_MAX_BATCH_SIZE
    };

    // we are predicting N next tokens in parallel,
    // so we have N parallel sets of attention heads and N parallel output tokens.
    enum class MultiTokenPredictionIndex : size_t
    {
        START = 0,
#if RLLM_MTP_HEAD_COUNT > 1
        ONE = 1,
#endif
#if RLLM_MTP_HEAD_COUNT > 2
        TWO = 2,
#endif
        MAX = RLLM_MTP_HEAD_COUNT
    };

    static_assert(RLLM_MTP_HEAD_COUNT >= 1 && RLLM_MTP_HEAD_COUNT <= 3);

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

    /**
     * @brief the model expands each 512-element token embedding to a 2048-element intermediate representation.
     *
     * The SwiGLU feed-forward block processes data in this wider space, then projects it back to 512 dimensions.
     * 4× is a conventional Transformer design choice that increases capacity;
     * it isn’t a mathematical requirement and could be tuned, though changing it affects parameter count,
     * memory, and compute roughly proportionally.
     */
    static constexpr size_t EMBEDDING_TO_INTERMEDIATE_RATIO = 4;

    // Feed-forward hidden dimension: static_cast<int>(FFDimension::MAX) = 4 × EmbeddingDimension::MAX.
    enum class FFDimension : size_t
    {
        START = 0,
        MAX = static_cast<size_t>(EmbeddingDimension::MAX) * EMBEDDING_TO_INTERMEDIATE_RATIO
    };

    enum class TempStorage : size_t
    {
        START = 0,
        ZERO = 0,
        ONE = 1,
        OPTIMIZER_GRADIENT_MAX = 0,
        OPTIMIZER_CLIPPED_COUNT = 1,
        OPTIMIZER_ADAM_UPDATE_SQUARE_SUM = 2,
        OPTIMIZER_ADAM_UPDATE_MAX = 3,
        OPTIMIZER_WEIGHT_UPDATE_SQUARE_SUM = 4,
        OPTIMIZER_WEIGHT_UPDATE_MAX = 5,
        OPTIMIZER_PARAMETER_COUNT = 6,
        OPTIMIZER_GRADIENT_SQUARE_SUM = 7,
        OPTIMIZER_GLOBAL_CLIP_SCALE = 8,
        MAX = 9 // maximum scratch/diagnostic slots used by kernels
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

    static inline BatchIndex inc(BatchIndex id)
    {
        assert(id < BatchIndex::MAX);
        return static_cast<BatchIndex>(static_cast<size_t>(id) + 1);
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

    struct ConflictingToken
    {
        TokenID tok;
        PositionIndex pos;
    };

    enum class ConflictIndex : size_t
    {
        START = 0,
        MAX = 256
    };

    static inline ConflictIndex inc(ConflictIndex id)
    {
        assert(id < ConflictIndex::MAX);
        return static_cast<ConflictIndex>(static_cast<size_t>(id) + 1);
    }

    class CpuInputLine 
    {
        public:

        void sub_array(CpuInputLine& result, PositionIndex length) const
        {
            assert(static_cast<size_t>(length) <= m_cpu.size());
            result.m_cpu.assign(m_cpu.begin(), m_cpu.begin() + static_cast<size_t>(length));
            result.string_table_index.assign(
                string_table_index.begin(), string_table_index.begin() + static_cast<size_t>(length));
            result.string_table_value = string_table_value;
            result.integer_constant_value = integer_constant_value;
            result.float_constant_value = float_constant_value;
        }

        void push_back(TokenID t, std::string_view string_value = {})
        {
            assert(m_cpu.size() < static_cast<size_t>(PositionIndex::MAX));
            m_cpu.push_back(t);
            string_table_index.push_back(intern_string_value(t, string_value));
        }

        void push_back_string_table_index(size_t index)
        {
            assert(m_cpu.size() < static_cast<size_t>(PositionIndex::MAX));
            assert(index < string_table_value.size());
            m_cpu.push_back(TokenID::STRING_TABLE_INDEX);
            string_table_index.push_back(index);
        }

        void push_front(TokenID t, std::string_view string_value = {})
        {
            assert(m_cpu.size() < static_cast<size_t>(PositionIndex::MAX));
            m_cpu.insert(m_cpu.begin(), t);
            string_table_index.insert(string_table_index.begin(), intern_string_value(t, string_value));
        }

        void push_back_from(const CpuInputLine& other, PositionIndex pos)
        {
            assert(m_cpu.size() < static_cast<size_t>(PositionIndex::MAX));
            const size_t source_pos = static_cast<size_t>(pos);
            assert(source_pos < other.m_cpu.size());
            m_cpu.push_back(other.m_cpu[source_pos]);
            const size_t other_index = other.string_table_index[source_pos];
            if (other_index == NO_STRING_TABLE_INDEX)
            {
                string_table_index.push_back(NO_STRING_TABLE_INDEX);
                return;
            }
            string_table_index.push_back(intern_string_value(
                other.m_cpu[source_pos], other.get_value_for_token(other.m_cpu[source_pos], other_index)));
        }

        template <typename UniformRandomBitGenerator>
        void permute_string_table(UniformRandomBitGenerator& rng)
        {
            permute_value_table(string_table_value, TokenStringCategory::None, rng);
            permute_value_table(integer_constant_value, TokenStringCategory::Integer, rng);
            permute_value_table(float_constant_value, TokenStringCategory::Float, rng);
        }

        template <typename UniformRandomBitGenerator>
        void permute_value_table(std::vector<std::string>& values,
            TokenStringCategory selected_category, UniformRandomBitGenerator& rng)
        {
            const size_t value_count = values.size();
            if (value_count < 2)
                return;

            std::vector<size_t> old_to_new(value_count);
            std::vector<size_t> old_order(value_count);
            for (size_t i = 0; i < value_count; ++i)
                old_order[i] = i;
            std::shuffle(old_order.begin(), old_order.end(), rng);

            std::vector<std::string> permuted_values(value_count);
            for (size_t new_index = 0; new_index < value_count; ++new_index)
            {
                const size_t old_index = old_order[new_index];
                assert(old_index < value_count);
                old_to_new[old_index] = new_index;
                permuted_values[new_index] = std::move(values[old_index]);
            }
            for (size_t pos = 0; pos < string_table_index.size(); ++pos)
            {
                size_t& index = string_table_index[pos];
                if (index == NO_STRING_TABLE_INDEX)
                    continue;
                const auto category = token_string_category(m_cpu[pos]);
                const bool selected = selected_category == TokenStringCategory::None
                    ? category != TokenStringCategory::Integer && category != TokenStringCategory::Float
                    : category == selected_category;
                if (!selected)
                    continue;
                assert(index < old_to_new.size());
                index = old_to_new[index];
            }
            values = std::move(permuted_values);
        }

        const TokenID& back() const
        {
            return m_cpu.back();
        }

        void pop_back()
        {
            m_cpu.pop_back();
            string_table_index.pop_back();
        }

        const TokenID& get(PositionIndex pos) const
        {
            return m_cpu[static_cast<size_t>(pos)];
        }

        const TokenID& get(size_t pos) const
        {
            return m_cpu[pos];
        }
        
        const TokenID& operator[](PositionIndex pos) const
        {
            return m_cpu[static_cast<size_t>(pos)];
        }

        void clear()
        {
            m_cpu.clear();
            string_table_index.clear();
            string_table_value.clear();
            integer_constant_value.clear();
            float_constant_value.clear();
        }

        bool empty() const
        {
            return m_cpu.empty();
        }

        PositionIndex size() const
        {
            return static_cast<PositionIndex>(m_cpu.size());
        }

        uint64_t hash() const
        {
            constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ull;
            constexpr uint64_t FNV_PRIME = 1099511628211ull;

            uint64_t hash = FNV_OFFSET_BASIS;
            for (const auto token : m_cpu)
            {
                uint64_t value = static_cast<uint64_t>(static_cast<int>(token)) + 1ull;
                for (int byte = 0; byte < 8; ++byte)
                {
                    hash ^= (value & 0xffull);
                    hash *= FNV_PRIME;
                    value >>= 8;
                }
            }
            for (const size_t index : string_table_index)
            {
                uint64_t value = index + 1ull;
                for (int byte = 0; byte < 8; ++byte)
                {
                    hash ^= (value & 0xffull);
                    hash *= FNV_PRIME;
                    value >>= 8;
                }
            }
            for (const auto& value : string_table_value)
                for (const unsigned char ch : value)
                {
                    hash ^= static_cast<uint64_t>(ch);
                    hash *= FNV_PRIME;
                }
            for (const auto& value : integer_constant_value)
                for (const unsigned char ch : value)
                {
                    hash ^= static_cast<uint64_t>(ch);
                    hash *= FNV_PRIME;
                }
            for (const auto& value : float_constant_value)
                for (const unsigned char ch : value)
                {
                    hash ^= static_cast<uint64_t>(ch);
                    hash *= FNV_PRIME;
                }
            return hash;
        }

        size_t get_string_table_index(PositionIndex pos) const
        {
            const size_t index = static_cast<size_t>(pos);
            assert(index < string_table_index.size());
            return string_table_index[index];
        }

        std::string_view get_string_table_value(size_t index) const
        {
            assert(index < string_table_value.size());
            return string_table_value[index];
        }

        const std::vector<std::string>& value_table_for_token(TokenID token) const
        {
            const auto category = token_string_category(token);
            if (category == TokenStringCategory::Integer)
                return integer_constant_value;
            if (category == TokenStringCategory::Float)
                return float_constant_value;
            return string_table_value;
        }

        std::string_view get_value_for_token(TokenID token, size_t index) const
        {
            const auto& values = value_table_for_token(token);
            assert(index < values.size());
            return values[index];
        }

        size_t value_count_for_token(TokenID token) const
        {
            return value_table_for_token(token).size();
        }

        std::vector<uint8_t> first_string_table_index_positions() const
        {
            std::vector<uint8_t> result(m_cpu.size(), 0);
            std::vector<uint8_t> seen_strings(string_table_value.size(), 0);
            std::vector<uint8_t> seen_integers(integer_constant_value.size(), 0);
            std::vector<uint8_t> seen_floats(float_constant_value.size(), 0);
            for (size_t pos = 0; pos < m_cpu.size(); ++pos)
            {
                const size_t index = string_table_index[pos];
                if (index == NO_STRING_TABLE_INDEX)
                    continue;
                const auto category = token_string_category(m_cpu[pos]);
                auto& seen = category == TokenStringCategory::Integer ? seen_integers
                    : (category == TokenStringCategory::Float ? seen_floats : seen_strings);
                assert(index < seen.size());
                if (seen[index] == 0)
                    result[pos] = 1;
                seen[index] = 1;
            }
            return result;
        }

        std::vector<TokenID> m_cpu;
        std::vector<size_t> string_table_index;
        std::vector<std::string> string_table_value;
        std::vector<std::string> integer_constant_value;
        std::vector<std::string> float_constant_value;

      private:
        size_t intern_string_value(TokenID token, std::string_view value)
        {
            if (value.empty())
                return NO_STRING_TABLE_INDEX;
            const auto category = token_string_category(token);
            auto& values = category == TokenStringCategory::Integer ? integer_constant_value
                : (category == TokenStringCategory::Float ? float_constant_value : string_table_value);
            const auto existing = std::ranges::find(values, value);
            if (existing != values.end())
                return static_cast<size_t>(existing - values.begin());
            values.emplace_back(value);
            return values.size() - 1;
        }
    };

    class GpuInputLine : public fixed_size_vector<TokenID, PositionIndex>
    {
      public:
        using Base = fixed_size_vector<TokenID, PositionIndex>;

        GpuInputLine() = default;

        GpuInputLine(const CpuInputLine& other) = delete;
        /*
            : Base()
            , m_cpu(other.m_cpu)
        {
            sync_to_device(queue);
        }*/

        GpuInputLine& operator=(const CpuInputLine& other) = delete;
        /*
        {
            if (this != &other)
            {
                m_cpu = other.m_cpu;
                sync_to_device(queue);
            }
            return *this;
        }*/

        /** Upload from CpuInputLine to device. Call after modifying CpuInputLine. */
        void sync_to_device(VulkanQueue& queue, const CpuInputLine& cpu) const
        {
            auto* self = const_cast<GpuInputLine*>(this);
            self->m_upload_staging.clear();
            for (const auto token : cpu.m_cpu)
                self->m_upload_staging.push_back(token);
            self->Base::copy_from_cpu(queue, self->m_upload_staging);
            queue.wait("GpuInputLine upload staging");
        }

      private:
        cpu_fixed_vector<TokenID, PositionIndex> m_upload_staging;
    };

    /** CPU description of a ragged micro-batch packed into one row axis.
     *
     * Rows belonging to an example are contiguous. `row_begin(b)` and
     * `row_end(b)` define the block used by causal attention, while
     * `local_position(row)` resets RoPE at each example.
     */
    class PackedBatchInput
    {
      public:
        PackedBatchInput() = default;

        explicit PackedBatchInput(const std::vector<CpuInputLine>& examples)
        {
            assign(examples);
        }

        void assign(const std::vector<CpuInputLine>& examples)
        {
            assert(examples.size() <= static_cast<size_t>(BatchIndex::MAX));
            m_tokens.clear();
            m_row_begin.clear();
            m_row_end.clear();
            m_last_row.clear();
            m_local_position.clear();
            m_row_batch.clear();

            for (size_t batch = 0; batch < examples.size(); ++batch)
            {
                const auto& example = examples[batch];
                assert(!example.empty());
                const auto begin = m_tokens.size();
                m_row_begin.push_back(begin);
                for (const auto pos : enum_iterator1D<PositionIndex>(example.size()))
                {
                    assert(static_cast<size_t>(m_tokens.size()) < static_cast<size_t>(PositionIndex::MAX));
                    m_tokens.push_back_from(example, pos);
                    m_local_position.push_back(pos);
                    m_row_batch.push_back(static_cast<BatchIndex>(batch));
                }
                m_row_end.push_back(m_tokens.size());
                m_last_row.push_back(dec(m_tokens.size()));
            }
        }

        BatchIndex batch_size() const { return m_row_begin.size(); }
        PositionIndex packed_rows() const { return m_tokens.size(); }
        const CpuInputLine& tokens() const { return m_tokens; }
        PositionIndex row_begin(BatchIndex batch) const { return m_row_begin[batch]; }
        PositionIndex row_end(BatchIndex batch) const { return m_row_end[batch]; }
        PositionIndex last_row(BatchIndex batch) const { return m_last_row[batch]; }
        PositionIndex local_position(PositionIndex row) const { return m_local_position[row]; }
        BatchIndex row_batch(PositionIndex row) const { return m_row_batch[row]; }
        const cpu_fixed_vector<PositionIndex, BatchIndex>& row_begins() const { return m_row_begin; }
        const cpu_fixed_vector<PositionIndex, BatchIndex>& row_ends() const { return m_row_end; }
        const cpu_fixed_vector<PositionIndex, BatchIndex>& last_rows() const { return m_last_row; }
        const cpu_fixed_vector<PositionIndex, PositionIndex>& local_positions() const { return m_local_position; }
        const cpu_fixed_vector<BatchIndex, PositionIndex>& row_batches() const { return m_row_batch; }
        bool may_attend(PositionIndex query, PositionIndex key) const
        {
            return row_batch(query) == row_batch(key) && key <= query;
        }

      private:
        CpuInputLine m_tokens;
        cpu_fixed_vector<PositionIndex, BatchIndex> m_row_begin;
        cpu_fixed_vector<PositionIndex, BatchIndex> m_row_end;
        cpu_fixed_vector<PositionIndex, BatchIndex> m_last_row;
        cpu_fixed_vector<PositionIndex, PositionIndex> m_local_position;
        cpu_fixed_vector<BatchIndex, PositionIndex> m_row_batch;
    };

    class GpuPackedBatchInput
    {
      public:
        void sync_to_device(VulkanQueue& queue, const PackedBatchInput& cpu)
        {
            tokens.sync_to_device(queue, cpu.tokens());
            cpu_fixed_vector<int, BatchIndex> cpu_row_begin;
            cpu_fixed_vector<int, BatchIndex> cpu_row_end;
            cpu_fixed_vector<int, BatchIndex> cpu_last_row;
            for (const auto batch : enum_iterator1D<BatchIndex>(cpu.batch_size()))
            {
                cpu_row_begin.push_back(static_cast<int>(cpu.row_begin(batch)));
                cpu_row_end.push_back(static_cast<int>(cpu.row_end(batch)));
                cpu_last_row.push_back(static_cast<int>(cpu.last_row(batch)));
            }
            cpu_fixed_vector<int, PositionIndex> cpu_local_position;
            cpu_fixed_vector<int, PositionIndex> cpu_row_batch;
            for (const auto row : enum_iterator1D<PositionIndex>(cpu.packed_rows()))
            {
                cpu_local_position.push_back(static_cast<int>(cpu.local_position(row)));
                cpu_row_batch.push_back(static_cast<int>(cpu.row_batch(row)));
            }
            row_begin.copy_from_cpu(queue, cpu_row_begin);
            row_end.copy_from_cpu(queue, cpu_row_end);
            last_row.copy_from_cpu(queue, cpu_last_row);
            local_position.copy_from_cpu(queue, cpu_local_position);
            row_batch.copy_from_cpu(queue, cpu_row_batch);
            // The converted CPU vectors are temporary staging buffers. Keep
            // them alive until every H2D copy has completed.
            queue.wait("GpuPackedBatchInput metadata upload");
            batch_size = cpu.batch_size();
            packed_rows = cpu.packed_rows();
        }

        GpuInputLine tokens;
        fixed_size_vector<int, BatchIndex> row_begin;
        fixed_size_vector<int, BatchIndex> row_end;
        fixed_size_vector<int, BatchIndex> last_row;
        fixed_size_vector<int, PositionIndex> local_position;
        fixed_size_vector<int, PositionIndex> row_batch;
        BatchIndex batch_size{BatchIndex::START};
        PositionIndex packed_rows{PositionIndex::START};
    };

    class InputLineView
    {
      public:
        InputLineView(const CpuInputLine& data, PositionIndex start, PositionIndex length)
            : m_data(data)
            , m_start(start)
            , m_length(length)
        {}

        const TokenID& operator[](PositionIndex index) const
        {
            assert(((int) index + (int) m_start) < (int) m_length);
            return m_data.get(static_cast<size_t>(m_start) + static_cast<size_t>(index));
        }

        PositionIndex size() const
        {
            return m_length;
        }

      private:
        const CpuInputLine& m_data;
        PositionIndex m_start;
        PositionIndex m_length;
    };


    struct Score
    {
        Score()
        {
            values.set_size(TokenID::MAX);
            string_table_index_values.set_size(PositionIndex::MAX);
            temp_values.set_size(TempStorage::MAX);
            temp_values_cpu.set_size(TempStorage::MAX);
        }

        void reset(VulkanQueue& queue)
        {
            values.zero(queue);
            string_table_index_values.zero(queue);
            temp_values.zero(queue);
            string_table_index_prediction_active = false;
        }

        fixed_size_vector<float, TokenID> values;
        fixed_size_vector<float, PositionIndex> string_table_index_values;
        bool string_table_index_prediction_active = false;
        fixed_size_vector<float, TempStorage> temp_values; // for use in softmax computation, to avoid modifying the original logits
        cpu_fixed_vector<float, TempStorage> temp_values_cpu;
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
    };

    struct OutputStringTableIndex
    {
        PositionIndex index;
        float activation;
    };

} // namespace rllm

namespace rlmm = rllm;
