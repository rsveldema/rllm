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
        MAX = 1024 * 8,
        UNKNOWN_POSITION_INDEX = static_cast<size_t>(-1)
    };

    // Index of an attention head (0..HeadsIndex::MAX-1).
    enum class HeadsIndex : size_t
    {
        START = 0,
        MAX = 8
    };

    // Batch axis for batched forward/backward paths.
    enum class BatchIndex : size_t
    {
        START = 0,
        MAX = 1024
    };

    // we are predicting N next tokens in parallel,
    // so we have N parallel sets of attention heads and N parallel output tokens.
    enum class MultiTokenPredictionIndex : size_t
    {
        START = 0,
        ONE = 1,
        TWO = 2,
        THREE = 3,
        FOUR = 4,
        MAX = 4
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
            m_cpu.sub_array(result.m_cpu, length);
        }

        void push_back(TokenID t)
        {
            m_cpu.push_back(t);
        }

        const TokenID& back() const
        {
            return m_cpu.back();
        }

        void pop_back()
        {
            m_cpu.pop_back();
        }

        const TokenID& get(PositionIndex pos) const
        {
            return m_cpu[pos];
        }

        const TokenID& get(size_t pos) const
        {
            return m_cpu[pos];
        }
        
        const TokenID& operator[](PositionIndex pos) const
        {
            return m_cpu[pos];
        }

        void clear()
        {
            m_cpu.clear();
        }

        bool empty() const
        {
            return m_cpu.empty();
        }

        PositionIndex size() const
        {
            return m_cpu.size();
        }

        uint64_t hash() const
        {
            constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ull;
            constexpr uint64_t FNV_PRIME = 1099511628211ull;

            uint64_t hash = FNV_OFFSET_BASIS;
            for (const auto i : enum_iterator1D<PositionIndex>(m_cpu.size()))
            {
                uint64_t value = static_cast<uint64_t>(static_cast<int>(m_cpu[i])) + 1ull;
                for (int byte = 0; byte < 8; ++byte)
                {
                    hash ^= (value & 0xffull);
                    hash *= FNV_PRIME;
                    value >>= 8;
                }
            }
            return hash;
        }

        cpu_fixed_vector<TokenID, PositionIndex> m_cpu;
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
            const_cast<GpuInputLine*>(this)->Base::copy_from_cpu(queue, cpu.m_cpu);
        }
    };

    /** CPU description of a ragged micro-batch packed into one row axis.
     *
     * Rows belonging to an example are contiguous. `row_begin(b)` and
     * `row_end(b)` define the block used by causal attention, while
     * `local_position(row)` resets positional encoding at each example.
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
                    m_tokens.push_back(example[pos]);
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
            temp_values.set_size(TempStorage::MAX);
            temp_values_cpu.set_size(TempStorage::MAX);
        }

        void reset(VulkanQueue& queue)
        {
            values.zero(queue);
            temp_values.zero(queue);
        }

        fixed_size_vector<float, TokenID> values;
        fixed_size_vector<float, TempStorage> temp_values; // for use in softmax computation, to avoid modifying the original logits
        cpu_fixed_vector<float, TempStorage> temp_values_cpu;
    };

    struct OutputToken
    {
        TokenID token_id;
        float activation;
    };

} // namespace rllm

namespace rlmm = rllm;
