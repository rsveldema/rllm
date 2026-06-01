#include <InputLayer.hpp>
#include <RandomHelpers.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rllm
{
    static void fill_embeddings_with_positional_encoding(
        // OFFLOAD_PARAMETERS(tokens, embeddings, h, model_dim)
        const InputLine& tokens,
        const fixed_size_matrix<rlmm_float_small, TokenID, EmbeddingDimension>& embeddings,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h,
        float model_dim
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(tokens.size());

        OFFLOAD_PARFOR_2D_PARAM(pos, di, grid, (tokens, embeddings, h, model_dim))
        const int tok = static_cast<int>(tokens[pos]);
        const int di_int = static_cast<int>(di);
        const float emb_val = embeddings[tok, di];
        const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(di_int & ~1) / model_dim);
        const float pos_f = static_cast<float>(pos);
        const float pe = (di_int % 2 == 0) ? std::sin(pos_f * freq) : std::cos(pos_f * freq);
        h[pos, di] = static_cast<rlmm_float>(emb_val + pe);
        ENDFOR
    }

    void InputLayer::reset_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok, d] = RLMM_ZERO;
    }

    void InputLayer::set_random_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok, d] = static_cast<rlmm_float>(get_random_value(-0.1f, 0.1f));
    }

    // Fill h[T × D_MODEL] = token_embedding + sinusoidal positional encoding.
    // PE[pos, 2i]   = sin(pos / 10000^(2i / D_MODEL))
    // PE[pos, 2i+1] = cos(pos / 10000^(2i / D_MODEL))
    void InputLayer::propagate_forward(
        const InputLine& input,
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        h.set_rows(static_cast<PositionIndex>(input.size()));

        fill_embeddings_with_positional_encoding(input, m_embeddings, h, static_cast<float>(EmbeddingDimension::MAX));
    }


        struct ConflicingToken
        {
            TokenID tok;
            PositionIndex pos;
        };

        enum class ConflictIndex : size_t
        {
            START = 0,
            MAX = 256
        };

        ConflictIndex inc(ConflictIndex c)
        {
            assert(c < ConflictIndex::MAX);
            return static_cast<ConflictIndex>(static_cast<size_t>(c) + 1);
        }

    // Update token embeddings.  dh[T × D_MODEL] = ∂L/∂h.
    // Positional encodings are fixed, so only the embedding contribution is updated.
    void InputLayer::propagate_backward(
        const InputLine& input,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dh,
        float learning_rate
    )
    {
        static fixed_size_vector<uint16_t, TokenID> updated_tokens;
        static fixed_size_vector<ConflicingToken, ConflictIndex> conflicts; // count how many times each token appears in the input

        // count how many tokens appear more than once in the input
        updated_tokens.fill(0);
        ConflictIndex duplicate_count = ConflictIndex::START;
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const auto tok = input[pos];
            updated_tokens[tok]++;

            if (updated_tokens[tok] > 1)
            {
                assert(duplicate_count < ConflictIndex::MAX); // sanity check: we should never have this many duplicates in a single input
                conflicts[duplicate_count] = {tok, pos};
                duplicate_count = inc(duplicate_count);
            }
        }

        // each duplicate we'll handle sequentially to avoid race conditions on the same embedding vector
        for (const auto i : enum_iterator<ConflictIndex>(duplicate_count))
        {
            for (const auto di : enum_iterator<EmbeddingDimension>())
            {
                auto& conflict_tok = conflicts[i].tok;
                auto& conflict_pos = conflicts[i].pos;
                auto& e = m_embeddings[conflict_tok, di];
                // if the same token appears multiple times, we average the gradient across those positions
                const auto rate = learning_rate / float(updated_tokens[conflict_tok]);
                e = math::clamp(e + rate * dh[conflict_pos, di], RLMM_NEG_ONE, RLMM_ONE);
            }
        }

        // the bulk of the tokens can be updated in parallel since they don't have
        // conflicting updates. We skip any token that appears more than once since
        // those were handled in the sequential loop above.
        PARFOR_2D(pos, di, enum_iterator2D<PositionIndex, EmbeddingDimension>(input.size()))
        // the same token can appear at multiple positions,
        // but we update its embedding using the gradient from
        // each position where it appears.  This is a bit of a hack
        // since it means the same embedding can be updated multiple times
        // per training step, but it should be fine as long as the
        // learning rate is small enough.

        const auto tok = input[pos];
        const auto count = updated_tokens[tok];
        assert(count > 0); // sanity check: this token should have been marked as updated
        if (count > 1)
            // cannot do this in parallel
            continue;

        assert(count == 1); // sanity check: this token should only appear once in the input, otherwise it would have been handled in the sequential loop above

        auto& e = m_embeddings[tok, di];
        e = math::clamp(e + learning_rate * dh[pos, di], RLMM_NEG_ONE, RLMM_ONE);
        ENDFOR
    }
} // namespace rllm
