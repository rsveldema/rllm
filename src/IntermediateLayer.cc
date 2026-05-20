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
                    next_layer.m_inputs.set(
                        next_neuron_index,
                        std::clamp(
                            next_layer.m_inputs.get(next_neuron_index) + weight * m_inputs.get(i, pos), 0.0f, 1.0f
                        )
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

                const auto next_neuron_index = m_connections.get(i, pos);
                assert(static_cast<TokenID>(i) < TokenID::MAX);
                const float d = delta[static_cast<TokenID>(i)];

                // Increase weight when downstream error is positive (need more signal).
                m_weights.set(
                    i, pos, std::clamp(m_weights.get(i, pos) + learning_rate * d * m_inputs.get(i, pos), 0.0f, 1.0f)
                );

                // Lower trigger makes this neuron fire more easily -- helpful when
                // downstream error is positive.
                m_trigger_values.set(i, pos, std::clamp(m_trigger_values.get(i, pos) - learning_rate * d, 0.0f, 1.0f));

                // Accumulate gradient for the layer below.
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
                if (m_inputs.get(i, pos) < m_trigger_values.get(i, pos))
                    continue; // neuron did not fire, no gradient to propagate

                const auto next_neuron_index = m_connections.get(i, pos);
                const float d = delta.get(i, pos);

                // Increase weight when downstream error is positive (need more signal).
                m_weights.set(
                    i, pos, std::clamp(m_weights.get(i, pos) + learning_rate * d * m_inputs.get(i, pos), 0.0f, 1.0f)
                );

                // Lower trigger makes this neuron fire more easily -- helpful when
                // downstream error is positive.
                m_trigger_values.set(i, pos, std::clamp(m_trigger_values.get(i, pos) - learning_rate * d, 0.0f, 1.0f));

                // Accumulate gradient for the layer below.
                prev_delta.set(i, pos, prev_delta.get(i, pos) + d * m_weights.get(i, pos));
            }
        }
    }

} // namespace rllm
