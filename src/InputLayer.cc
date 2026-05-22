#include <InputLayer.hpp>

namespace rllm
{
    // Maps (token, position) to a single IntermediateLayerIndex via a mixing hash
    // so that different (tok, pos) pairs spread uniformly across the layer.
    static IntermediateLayerIndex hash_input(TokenID tok, PositionIndex pos)
    {
        const size_t t = static_cast<size_t>(tok);
        const size_t p = static_cast<size_t>(pos);
        const size_t h = (t * 2654435761ULL ^ p * 40503ULL) % static_cast<size_t>(IntermediateLayerIndex::MAX);
        //const size_t h = (t + (p * 1023 * 31)) % static_cast<size_t>(IntermediateLayerIndex::MAX);
        return static_cast<IntermediateLayerIndex>(h);
    }

    void InputLayer::propagate_forward(const InputLine& input, IntermediateLayer& next_layer) const
    {
        next_layer.fill_inputs(0.0f);

        // this loop is over the length of the input line, which is at most
        // 128 tokens, and for each token we do a single hash and an accumulation into the next layer, so this should be
        // very fast.
        // Therefore no need to optimize further by parallelizing or anything like that.
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const TokenID tok = input[pos];
            assert(tok != TokenID::UNKNOWN_TOKEN_ID);

            const auto target = hash_input(tok, pos);
            next_layer.accumulate_input(target, 1.0f);
        }
    }

} // namespace rllm
