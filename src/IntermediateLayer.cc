#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>

namespace rllm
{
    static constexpr float MIN_TRIGGER = 0.0001f;
    static constexpr float MAX_TRIGGER = 1.0f;
    static constexpr float MIN_WEIGHT = 0.0f;
    static constexpr float MAX_WEIGHT = 1.0f;
    static constexpr size_t MAX_CONNECTIONS_PER_NEURON = 50;

    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            randomize_neuron(i);
    }

    void IntermediateLayer::set_random_weights_and_connections_to_output_layer()
    {
        // setup the layer JUST before the output layer.
        // Connections are encoded as IntermediateLayerIndex values; the output-layer
        // forward pass maps them to TokenIDs via modulo corpus size.
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            randomize_neuron_to_output(i);
    }

    void IntermediateLayer::propagate_forward_to_output(OutputLayer& output_layer) const
    {
        output_layer.m_inputs.fill(0.0f);
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            forward_neuron_to_output(i, output_layer);
    }

    void IntermediateLayer::propagate_forward(IntermediateLayer& next_layer)
    {
        next_layer.m_inputs.fill(0.0f);

        const auto max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < max_i; ++idx)
            forward_neuron(static_cast<IntermediateLayerIndex>(idx), next_layer);
    }

    void IntermediateLayer::propagate_backward_from_output_layer(
        const template_token_vector<float, TokenID>& delta,
        template_token_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        clear_last_weight_deltas();
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            backward_neuron_from_output(i, delta, prev_delta, learning_rate);
    }

    void IntermediateLayer::propagate_backward(
        const template_token_vector<float, IntermediateLayerIndex>& delta,
        template_token_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        clear_last_weight_deltas();
        const auto max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < max_i; ++idx)
            backward_neuron(static_cast<IntermediateLayerIndex>(idx), delta, prev_delta, learning_rate);
    }

    void IntermediateLayer::randomize_neuron(IntermediateLayerIndex i)
    {
        m_inputs[i] = 0.0f;
        m_trigger_values[i] = get_random_value(MIN_TRIGGER, MAX_TRIGGER);
        m_weights[i] = get_random_value(MIN_WEIGHT, MAX_WEIGHT);
        const int num_connections = 1 + (std::rand() % MAX_CONNECTIONS_PER_NEURON);
        std::vector<IntermediateLayerIndex> conns;
        conns.reserve(num_connections);
        for (int c = 0; c < num_connections; ++c)
            conns.push_back(get_random_enum_value<IntermediateLayerIndex>());
        m_connections[i] = std::move(conns);
    }

    void IntermediateLayer::randomize_neuron_to_output(IntermediateLayerIndex i)
    {
        m_inputs[i] = 0.0f;
        m_trigger_values[i] = get_random_value(MIN_TRIGGER, MAX_TRIGGER);
        m_weights[i] = get_random_value(MIN_WEIGHT, MAX_WEIGHT);
        const int num_connections = 1 + std::rand() % 5;
        std::vector<IntermediateLayerIndex> conns;
        conns.reserve(num_connections);
        for (int c = 0; c < num_connections; ++c)
        {
            const auto random_token_index = get_random_enum_value<TokenID>();
            conns.push_back(static_cast<IntermediateLayerIndex>(random_token_index));
        }
        m_connections[i] = std::move(conns);
    }

    void IntermediateLayer::forward_neuron_to_output(IntermediateLayerIndex i, OutputLayer& output_layer) const
    {
        const auto input_value = m_inputs[i];
        if (input_value < m_trigger_values[i])
            return;
        const auto weight = m_weights[i];
        for (const auto& target : m_connections[i])
        {
            const auto token_id = static_cast<TokenID>(
                static_cast<size_t>(target) % static_cast<size_t>(m_corpus.number_of_token_types())
            );
            output_layer.m_inputs.add_no_clamp(token_id, weight * input_value);
        }
    }

    void IntermediateLayer::forward_neuron(IntermediateLayerIndex i, IntermediateLayer& next_layer)
    {
        const auto input_value = m_inputs[i];
        if (input_value < m_trigger_values[i])
            return;
        const auto added_value = m_weights[i] * input_value;
        for (const auto& target : m_connections[i])
        {
            float& cell = next_layer.m_inputs[target];
            float expected = std::atomic_ref<float>{cell}.load(std::memory_order_relaxed);
            float desired;
            do {
                desired = std::clamp(expected + added_value, 0.0f, 1.0f);
            } while (!std::atomic_ref<float>{cell}.compare_exchange_weak(
                         expected, desired, std::memory_order_relaxed));
        }
    }

    void IntermediateLayer::backward_neuron_from_output(
        IntermediateLayerIndex i,
        const template_token_vector<float, TokenID>& delta,
        template_token_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        float d = 0.0f;
        for (const auto& target : m_connections[i])
        {
            const auto token_id = static_cast<TokenID>(
                static_cast<size_t>(target) % static_cast<size_t>(m_corpus.number_of_token_types())
            );
            assert(token_id < TokenID::MAX);
            d += delta[token_id];
        }

        const bool fired = m_inputs[i] >= m_trigger_values[i];

        if (!fired)
        {
            // Neuron didn't fire but downstream wants more signal:
            // lower the trigger so it fires more easily next time,
            // and propagate the gradient upstream so earlier layers also adapt.
            if (d > 0.0f)
            {
                const float weight_delta = learning_rate * d;
                const float trigger_delta = learning_rate * d;
                m_trigger_values.add_with_clamp(i, -trigger_delta, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
                m_last_weight_delta[i] = weight_delta;
                m_weights.add_with_clamp(i, weight_delta, Range<float>{MIN_WEIGHT, MAX_WEIGHT});
                prev_delta.add_no_clamp(i, d * m_weights[i]);
            }
            return;
        }

        // Neuron fired — full update.
        const float weight_delta = learning_rate * d * m_inputs[i];
        const float trigger_delta = learning_rate * d;
        m_weights.add_with_clamp(i, weight_delta, Range<float>{MIN_WEIGHT, MAX_WEIGHT});
        m_last_weight_delta[i] = weight_delta;
        m_trigger_values.add_with_clamp(i, -trigger_delta, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
        // Gradients are unbounded — do not clamp prev_delta.
        prev_delta.add_no_clamp(i, d * m_weights[i]);
    }

    void IntermediateLayer::backward_neuron(
        IntermediateLayerIndex i,
        const template_token_vector<float, IntermediateLayerIndex>& delta,
        template_token_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        float d = 0.0f;
        for (const auto& target : m_connections[i])
            d += delta[target];

        const bool fired = m_inputs[i] >= m_trigger_values[i];

        if (!fired)
        {
            if (d > 0.0f)
            {
                const float weight_delta = learning_rate * d;
                const float trigger_delta = learning_rate * d;
                m_trigger_values.add_with_clamp(i, -trigger_delta, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
                m_weights.add_with_clamp(i, weight_delta, Range<float>{MIN_WEIGHT, MAX_WEIGHT});
                m_last_weight_delta[i] = weight_delta;
                prev_delta.add_no_clamp(i, d * m_weights[i]);
            }
            return;
        }

        const float weight_delta = learning_rate * d * m_inputs[i];
        const float trigger_delta = learning_rate * d;
        m_weights.add_with_clamp(i, weight_delta, Range<float>{MIN_WEIGHT, MAX_WEIGHT});
        m_last_weight_delta[i] = weight_delta;
        m_trigger_values.add_with_clamp(i, -trigger_delta, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
        prev_delta.add_no_clamp(i, d * m_weights[i]);
    }

} // namespace rllm
