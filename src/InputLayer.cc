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
    void InputLayer::propagate_forward(const InputLine& input, std::vector<float>& h) const
    {
        const int T = static_cast<int>(input.size());
        const int D = static_cast<int>(EmbeddingDimension::MAX);
        assert(T > 0);
        assert(T <= static_cast<int>(PositionIndex::MAX));

        h.resize(T * D);

        for (int pos = 0; pos < T; ++pos)
        {
            const TokenID    tok    = input[static_cast<PositionIndex>(pos)];
            const auto&      embed  = m_embeddings[tok];
            float*           h_pos  = h.data() + pos * D;

            for (int di = 0; di < D; ++di)
            {
                const float emb_val = embed[static_cast<EmbeddingDimension>(di)];
                // Sinusoidal PE: dimension di uses frequency 10000^(floor(di/2)*2 / D)
                const float freq = 1.0f /
                    std::pow(10000.0f, static_cast<float>(di & ~1) / static_cast<float>(D));
                const float pe = (di % 2 == 0)
                    ? std::sin(static_cast<float>(pos) * freq)
                    : std::cos(static_cast<float>(pos) * freq);
                h_pos[di] = emb_val + pe;
            }
        }
    }

    // Update token embeddings.  dh[T × D_MODEL] = ∂L/∂h.
    // Positional encodings are fixed, so only the embedding contribution is updated.
    void InputLayer::propagate_backward(
        const InputLine&          input,
        const std::vector<float>& dh,
        float                     learning_rate
    )
    {
        const int D = static_cast<int>(EmbeddingDimension::MAX);
        const int T = static_cast<int>(input.size());

        for (int pos = 0; pos < T; ++pos)
        {
            const TokenID tok    = input[static_cast<PositionIndex>(pos)];
            auto&         embed  = m_embeddings[tok];
            const float*  dh_pos = dh.data() + pos * D;

            for (int di = 0; di < D; ++di)
            {
                auto& e = embed[static_cast<EmbeddingDimension>(di)];
                e = std::clamp(e + learning_rate * dh_pos[di], -1.0f, 1.0f);
            }
        }
    }

} // namespace rllm
