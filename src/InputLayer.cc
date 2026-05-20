#include <InputLayer.hpp>
#include <RandomHelpers.hpp>

#include <print>

namespace rllm
{
    void InputLayer::set_input_layer(const InputLine& input)
    {
        m_inputs.fill(0.0f);
        for (PositionIndex i = PositionIndex::START; i < static_cast<PositionIndex>(input.size()); i = inc(i))
        {
            if (i >= PositionIndex::MAX)
            {
                std::println(
                    "Warning: input line has more tokens than the max position index. Ignoring tokens beyond position "
                    "{}.",
                    static_cast<size_t>(PositionIndex::MAX)
                );
                break; // ignore tokens beyond the max position index
            }

            const auto token_id = input[i];
            m_inputs.set(token_id, i, 1.0f);
        }
    }

    void InputLayer::set_random_weights_and_connections()
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value());
                m_weights.set(i, pos, get_random_value());
                auto [target, pos_index] = get_random_value_centered_around(i, pos);
                m_connections.set(i, pos, std::make_pair(static_cast<IntermediateLayerIndex>(target), pos_index));
            }
        }
    }

} // namespace rllm
