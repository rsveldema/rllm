#include <InputLayer.hpp>

namespace rllm
{
    void InputLayer::propagate_forward(const InputLine& input, IntermediateLayer& next_layer) const
    {
        next_layer.fill_inputs(0.0f);
        for (const auto pos : enum_iterator<PositionIndex>(input.size()))
        {
            const TokenID tok = input[pos];
            if (tok == TokenID::UNKNOWN_TOKEN_ID)
                continue;

            const auto target = hash_input(tok, pos);
            next_layer.accumulate_input(target, 1.0f);
        }
    }

} // namespace rllm
