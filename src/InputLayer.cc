#include <InputLayer.hpp>
#include <RandomHelpers.hpp>
#include <cpu/cpu_flex_rows_matrix.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rllm
{
    static void fill_embeddings_with_positional_encoding(
        // OFFLOAD_PARAMETERS(tokens, embeddings, h, model_dim)
        const CpuInputLine& tokens,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& embeddings,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
        float model_dim
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(tokens.size());
        auto& queue = rllm::vulkan_runtime::get_queue(0);

        OFFLOAD_PARFOR_2D_PARAM(queue, pos, di, grid, (tokens, embeddings, h, model_dim))
        const int tok = static_cast<int>(tokens[pos]);
        const int di_int = static_cast<int>(di);
        const float emb_val = embeddings[tok, di];
        const float freq = (1.0f / std::pow(10000.0f, (static_cast<float>((di_int & ~1)) / model_dim)));
        const float pos_f = static_cast<float>(pos);
        const float pe = ((di_int % 2) == 0) ? std::sin((pos_f * freq)) : std::cos((pos_f * freq));
        h[pos, di] = static_cast<float>((emb_val + pe));
        ENDFOR
    }

    void InputLayer::reset_embeddings()
    {
        m_embeddings.zero();
        m_embeddings_cpu.zero();
    }

    void InputLayer::set_random_embeddings()
    {
        for (const auto tok : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                m_embeddings_cpu.set(tok, d, static_cast<float16>(get_random_value(-0.1f, 0.1f)));
        m_embeddings.copy_from_cpu(m_embeddings_cpu);
    }

    // Fill h[T × D_MODEL] = token_embedding + sinusoidal positional encoding.
    // PE[pos, 2i]   = sin(pos / 10000^(2i / D_MODEL))
    // PE[pos, 2i+1] = cos(pos / 10000^(2i / D_MODEL))
    void InputLayer::propagate_forward(
        const CpuInputLine& input,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        h.set_rows(static_cast<PositionIndex>(input.size()));

        fill_embeddings_with_positional_encoding(input, m_embeddings, h, static_cast<float>(EmbeddingDimension::MAX));
    }


    // Update token embeddings.  dh[T × D_MODEL] = ∂L/∂h.
    // Positional encodings are fixed, so only the embedding contribution is updated.
    void InputLayer::propagate_backward(
        const CpuInputLine& input,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        float learning_rate
    )
    {
        // D2H: download the gradient matrix to CPU before element access
        cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> dh_cpu;
        const_cast<flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>&>(dh).copy_to_cpu(dh_cpu);

        // Use per-instance state (not static) to avoid data race with PARFOR_2D.
        // Previously: static local variables shared across threads → heap corruption.
        m_updated_tokens.zero();
        m_conflicts.clear();

        size_t duplicate_count = 0;
        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
        {
            const auto tok = input.get(pos);
            m_updated_tokens[tok]++;;

            if (m_updated_tokens[tok] > 1)
            {
                assert(duplicate_count < static_cast<size_t>(ConflictIndex::MAX));
                m_conflicts.push_back(ConflictingToken{.tok = tok, .pos = pos});
                duplicate_count++;
            }
        }

        // Handle duplicates sequentially to avoid race conditions on the same embedding vector
        for (size_t i = 0; i < duplicate_count; i++)
        {
            for (const auto di : enum_iterator1D<EmbeddingDimension>())
            {
                const auto& conflict_tok = m_conflicts[i].tok;
                const auto& conflict_pos = m_conflicts[i].pos;
                const auto rate = learning_rate / static_cast<float>(m_updated_tokens[conflict_tok]);
                m_embeddings_cpu.set(conflict_tok, di,
                    math::clamp(
                        static_cast<float>(m_embeddings_cpu.get(conflict_tok, di)) + rate * dh_cpu.get(conflict_pos, di),
                        RLMM_NEG_ONE, RLMM_ONE));
            }
        }

        // Tokens appearing exactly once can be updated in parallel safely
        PARFOR_2D(pos, di, enum_iterator2D<PositionIndex, EmbeddingDimension>(input.size()))
        {
            const auto tok = input.get(pos);
            const auto count = m_updated_tokens[tok];
            assert(count > 0);
            if (count > 1)
                continue;

            assert(count == 1);

            m_embeddings_cpu.set(tok, di,
                math::clamp(
                    static_cast<float>(m_embeddings_cpu.get(tok, di)) + learning_rate * dh_cpu.get(pos, di),
                    RLMM_NEG_ONE, RLMM_ONE));
        }
        ENDFOR

        // Clean up per-token counters and copy modified rows to offload buffer
        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
        {
            const auto tok = input.get(pos);
            if (m_updated_tokens[tok] == 0)
                continue;
            m_embeddings.copy_row_to_offload_buffer(tok, m_embeddings_cpu);
            m_updated_tokens[tok] = 0;
        }
    }


    void InputLayer::get_embedding(TokenID tok, embedding_row_t& out) const
    {
        fixed_size_matrix<float16, TokenID, EmbeddingDimension>::export_row(tok, m_embeddings_cpu, out);
    }

} // namespace rllm
