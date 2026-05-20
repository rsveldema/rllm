#include <InputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
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
            assert(token_id < TokenID::MAX);

            m_inputs.set(token_id, i, 1.0f);
        }
    }

    void InputLayer::propagate_forward(IntermediateLayer& next_layer) const
    {
        next_layer.fill_inputs(0.0f);
        for (auto token = TokenID::START; token < TokenID::MAX; token = inc(token))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const auto input_value = m_inputs.get(token, pos);
                if (input_value <= 0.0f)
                    continue;

                const auto next_neuron_index = m_connections.get(token, pos);
                next_layer.accumulate_input(next_neuron_index, input_value);
            }
        }
    }

    void InputLayer::set_random_weights_and_connections()
    {
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                auto [target, pos_index] = get_random_value_centered_around(i, pos);
                m_connections.set(i, pos, std::make_pair(static_cast<IntermediateLayerIndex>(target), pos_index));
            }
        }
    }

} // namespace rllm
