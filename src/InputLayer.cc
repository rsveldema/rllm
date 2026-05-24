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
                m_embeddings[tok][d] = 0.0f;
    }

    void InputLayer::set_random_embeddings()
    {
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok][d] = get_random_value(-0.1f, 0.1f);
    }

    // Fill h[T × D_MODEL] = token_embedding + sinusoidal positional encoding.
    // PE[pos, 2i]   = sin(pos / 10000^(2i / D_MODEL))
    // PE[pos, 2i+1] = cos(pos / 10000^(2i / D_MODEL))
    void InputLayer::propagate_forward(
        const InputLine& input,
        flexible_size_matrix<float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        h.set_size(static_cast<PositionIndex>(input.size()), EmbeddingDimension::MAX);

        const float D = static_cast<float>(EmbeddingDimension::MAX);
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
                h[pos, di] = emb_val + pe;
            }
        }
    }

    // Update token embeddings.  dh[T × D_MODEL] = ∂L/∂h.
    // Positional encodings are fixed, so only the embedding contribution is updated.
    void InputLayer::propagate_backward(
        const InputLine& input,
        const flexible_size_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        float learning_rate
    )
    {
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const auto tok = input[pos];
            auto& embed = m_embeddings[tok];

            for (const auto di : enum_iterator<EmbeddingDimension>())
            {
                auto& e = embed[di];
                e = std::clamp(e + learning_rate * dh[pos, di], -1.0f, 1.0f);
            }
        }
    }
} // namespace rllm
