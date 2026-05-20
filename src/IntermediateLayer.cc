#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>

namespace rllm
{
    static constexpr float MIN_TRIGGER = 0.0001f;
    static constexpr float MAX_TRIGGER = 1.0f;

    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value(MIN_TRIGGER, MAX_TRIGGER));
                m_weights.set(i, pos, get_random_value(0.0f, 1.0f));
                auto target = get_random_value_centered_around(i, pos);
                m_connections.set(i, pos, target);
            }
        }
    }

    void IntermediateLayer::set_random_weights_and_connections_to_output_layer(Corpus& corpus)
    {
        // setup the layer JUST before the output layer.
        // It needs to have connections to the output layer that are distributed
        // across the tokens in the corpus.
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value(MIN_TRIGGER, MAX_TRIGGER));
                m_weights.set(i, pos, get_random_value(0.0f, 1.0f));
                m_connections.set(
                    i,
                    pos,
                    std::make_pair(
                        static_cast<IntermediateLayerIndex>(static_cast<int>(i) % corpus.size()), PositionIndex::START
                    )
                );
            }
        }
    }

    void IntermediateLayer::propagate_forward_to_output(OutputLayer& output_layer) const
    {
        output_layer.m_inputs.fill(0.0f);
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const auto input_value = m_inputs.get(i, pos);
                if (input_value < m_trigger_values.get(i, pos))
                    continue;

                const auto [target, _target_pos] = m_connections.get(i, pos);
                if (static_cast<size_t>(target) >= static_cast<size_t>(TokenID::MAX))
                    continue;

                const auto token_id = static_cast<TokenID>(target);
                const auto weight = m_weights.get(i, pos);

                output_layer.m_inputs.add_with_clamp(token_id, weight * input_value);
            }
        }
    }

    void IntermediateLayer::propagate_forward(IntermediateLayer& next_layer)
    {
        next_layer.m_inputs.fill(0.0f);

        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                if (m_inputs.get(i, pos) >= m_trigger_values.get(i, pos))
                {
                    const auto next_neuron_index = m_connections.get(i, pos);

                    const auto weight = m_weights.get(i, pos);
                    next_layer.m_inputs.add_with_clamp(
                        next_neuron_index, weight * m_inputs.get(i, pos)
                    );
                }
            }
        }
    }

    void IntermediateLayer::propagate_backward(
        const template_token_vector<float, TokenID>& delta,
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
        float learning_rate
    )
    {
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const auto [target_token, _target_pos] = m_connections.get(i, pos);
                const auto token_id = static_cast<TokenID>(target_token);
                assert(token_id < TokenID::MAX);
                const float d = delta[token_id];

                const bool fired = m_inputs.get(i, pos) >= m_trigger_values.get(i, pos);

                if (!fired)
                {
                    // Neuron didn't fire but downstream wants more signal:
                    // lower the trigger so it fires more easily next time.
                    if (d > 0.0f)
                        m_trigger_values.add_with_clamp(i, pos, -learning_rate * d, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
                    continue;
                }

                // Neuron fired — full update.
                m_weights.add_with_clamp(i, pos, learning_rate * d * m_inputs.get(i, pos));
                m_trigger_values.add_with_clamp(i, pos, -learning_rate * d, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
                // Gradients are unbounded — do not clamp prev_delta.
                prev_delta.set(i, pos, prev_delta.get(i, pos) + d * m_weights.get(i, pos));
            }
        }
    }

    void IntermediateLayer::propagate_backward(
        const template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& delta,
        template_token_matrix<float, IntermediateLayerIndex, PositionIndex>& prev_delta,
        float learning_rate
    )
    {
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const float d = delta.get(i, pos);
                const bool fired = m_inputs.get(i, pos) >= m_trigger_values.get(i, pos);

                if (!fired)
                {
                    if (d > 0.0f)
                        m_trigger_values.add_with_clamp(i, pos, -learning_rate * d, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
                    continue;
                }

                m_weights.add_with_clamp(i, pos, learning_rate * d * m_inputs.get(i, pos));
                m_trigger_values.add_with_clamp(i, pos, -learning_rate * d, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
                prev_delta.set(i, pos, prev_delta.get(i, pos) + d * m_weights.get(i, pos));
            }
        }
    }

} // namespace rllm
