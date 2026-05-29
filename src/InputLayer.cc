#include <InputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rllm
{
    void InputLayer::reset_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok][d] = RLMM_ZERO;
    }

    void InputLayer::set_random_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok][d] = static_cast<rlmm_float>(get_random_value(-0.1f, 0.1f));
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

        constexpr float D = static_cast<float>(EmbeddingDimension::MAX);

        // TODO: use a PARFOR_2D here
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const TokenID tok = input[pos];
            const auto& embed = m_embeddings[tok];

            for (const auto di : enum_iterator<EmbeddingDimension>())
            {
                const int di_int = static_cast<int>(di);
                const float emb_val = embed[di];
                const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(di_int & ~1) / D);
                const float pe = (di_int % 2 == 0) ? std::sin(static_cast<float>(pos) * freq)
                                                   : std::cos(static_cast<float>(pos) * freq);
                h[pos, di] = static_cast<rlmm_float>(emb_val + pe);
            }
        }
    }

    // Update token embeddings.  dh[T × D_MODEL] = ∂L/∂h.
    // Positional encodings are fixed, so only the embedding contribution is updated.
    void InputLayer::propagate_backward(
        const InputLine& input,
        const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dh,
        float learning_rate
    )
    {
        // TODO: use a PARFOR_2D here
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const auto tok = input[pos];
            auto& embed = m_embeddings[tok];

            for (const auto di : enum_iterator<EmbeddingDimension>())
            {
                auto& e = embed[di];
                e = math::clamp(e + learning_rate * static_cast<rlmm_float>(dh[pos, di]), RLMM_NEG_ONE, RLMM_ONE);
            }
        }
    }
} // namespace rllm
