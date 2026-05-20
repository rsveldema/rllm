#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>

namespace rllm
{
    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                m_inputs.set(i, pos, 0.0f);
                m_trigger_values.set(i, pos, get_random_value());
                m_weights.set(i, pos, get_random_value());
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
                m_trigger_values.set(i, pos, get_random_value());
                m_weights.set(i, pos, get_random_value());
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

                output_layer.m_inputs.add_with_clamp(token_id, weight * input_value, 0.0f, 1.0f);
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
                        next_neuron_index, weight * m_inputs.get(i, pos), 0.0f, 1.0f
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
                if (m_inputs.get(i, pos) < m_trigger_values.get(i, pos))
                    continue; // neuron did not fire, no gradient to propagate

                const auto [target_token, _target_pos] = m_connections.get(i, pos);
                const auto token_id = static_cast<TokenID>(target_token);
                assert(token_id < TokenID::MAX);
                const float d = delta[token_id];

                // Increase weight when downstream error is positive (need more signal).
                m_weights.add_with_clamp(i, pos, learning_rate * d * m_inputs.get(i, pos), 0.0f, 1.0f);

                // Lower trigger makes this neuron fire more easily -- helpful when
                // downstream error is positive.
                m_trigger_values.add_with_clamp(i, pos, -learning_rate * d, 0.0f, 1.0f);

                // Accumulate gradient for the layer below.
                prev_delta.add_with_clamp(i, pos, d * m_weights.get(i, pos), 0.0f, 1.0f);
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
                if (m_inputs.get(i, pos) < m_trigger_values.get(i, pos))
                    continue; // neuron did not fire, no gradient to propagate

                const auto next_neuron_index = m_connections.get(i, pos);
                const float d = delta.get(i, pos);

                // Increase weight when downstream error is positive (need more signal).
                m_weights.add_with_clamp(i, pos, learning_rate * d * m_inputs.get(i, pos), 0.0f, 1.0f);

                // Lower trigger makes this neuron fire more easily -- helpful when
                // downstream error is positive.
                m_trigger_values.add_with_clamp(i, pos, -learning_rate * d, 0.0f, 1.0f);

                // Accumulate gradient for the layer below.
                prev_delta.add_with_clamp(i, pos, d * m_weights.get(i, pos), 0.0f, 1.0f);
            }
        }
    }

} // namespace rllm
