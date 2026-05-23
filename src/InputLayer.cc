#include <InputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>

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
        // Xavier-style init: small uniform values so the first intermediate layer
        // starts with a reasonable signal spread.
        for (const auto tok : enum_iterator<TokenID>())
            for (const auto d : enum_iterator<EmbeddingDimension>())
                m_embeddings[tok][d] = get_random_value(-0.1f, 0.1f);
    }

    void InputLayer::propagate_forward(const InputLine& input, IntermediateLayer& next_layer) const
    {
        next_layer.fill_inputs(0.0f);

        assert(!input.empty());
        assert(input.size() <= PositionIndex::MAX);

        // Each position p owns neurons [p*EmbeddingDimension::MAX, p*EmbeddingDimension::MAX + EmbeddingDimension::MAX).
        // Writing the token's learned embedding into those neurons gives the network a
        // continuous, updatable representation of each (token, position) pair.

        #pragma omp parallel for schedule(static)
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const TokenID tok = input[pos];
            assert(tok != TokenID::UNKNOWN_TOKEN_ID);

            const auto& embed = m_embeddings[tok];
            const size_t base = static_cast<size_t>(pos) * static_cast<size_t>(EmbeddingDimension::MAX);

            for (const auto d : enum_iterator<EmbeddingDimension>())
            {
                const auto idx = static_cast<IntermediateLayerIndex>(base + static_cast<size_t>(d));
                next_layer.accumulate_input(idx, embed[d], Range<float>{MIN_NEURON_INPUT, MAX_NEURON_INPUT});
            }
        }
    }

    void InputLayer::propagate_backward(
        const InputLine& input,
        const template_token_vector<float, IntermediateLayerIndex>& delta,
        float learning_rate
    )
    {
        // delta[i] = ∂L/∂(first_intermediate_layer.m_inputs[i]).
        // Since the forward pass sets m_inputs[base+d] = embed[tok][d] directly,
        // ∂L/∂embed[tok][d] = delta[base+d], so we nudge the embedding in that direction.
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const TokenID tok = input[pos];
            assert(tok != TokenID::UNKNOWN_TOKEN_ID);

            auto& embed = m_embeddings[tok];
            const size_t base = static_cast<size_t>(pos) * static_cast<size_t>(EmbeddingDimension::MAX);


            for (const auto d : enum_iterator<EmbeddingDimension>())
            {
                const auto idx = static_cast<IntermediateLayerIndex>(base + static_cast<size_t>(d));
                embed[d] = std::clamp(embed[d] + learning_rate * delta[idx], -1.0f, 1.0f);
            }
        }
    }

} // namespace rllm
