#include <InputLayer.hpp>

namespace rllm
{
    void InputLayer::set_input_layer(const InputLine& input)
    {
        m_input = input;
    }

    void InputLayer::propagate_forward(IntermediateLayer& next_layer) const
    {
        next_layer.fill_inputs(0.0f);
        for (PositionIndex pos = PositionIndex::START;
             pos < m_input.size();
             pos = inc(pos))
        {
            const TokenID tok = m_input[pos];
            if (tok == TokenID::UNKNOWN_TOKEN_ID)
                continue;

            const auto target = hash_input(tok, pos);
            next_layer.accumulate_input(target, 1.0f);
        }
    }

} // namespace rllm
