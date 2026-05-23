#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <print>

namespace rllm
{
    // Gaussian activation centred on `trigger` with width `sigma`.
    // Returns 1 when input == trigger and decays symmetrically away from it.
    static float gaussian_penalty(float input, float trigger, float sigma)
    {
        const float diff = input - trigger;
        return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
    }

    static constexpr float MIN_TRIGGER = 0.0001f;
    // Keep initial triggers spread across the full [0,1] range so that the
    // per-token output activation stays small (~0.02).  With MAX_TRIGGER=0.5
    // every layer saturates and all 1132 output tokens activate simultaneously;
    // the resulting delta[wrong] = -0.5 × 1131 swamps delta[expected] = +1,
    // collapsing the network to fires-nothing within a few hundred iterations.
    static constexpr float MAX_TRIGGER = 1.0f;
    static constexpr float MIN_WEIGHT = 0.0f;
    static constexpr float MAX_WEIGHT = 1.0f;

    static constexpr size_t MAX_NUM_CONNECTIONS_PER_NEURON = 200;

    enum class NeuronRandomizationType
    {
        RANDOMIZE_ALL,
        RANDOMIZE_WEIGHTS_ONLY,
        RANDOMIZE_CONNECTIONS_ONLY,
        NO_RANDOMIZATION
    };


    // static constexpr auto randomization_type = NeuronRandomizationType::NO_RANDOMIZATION;
    static constexpr auto randomization_type = NeuronRandomizationType::RANDOMIZE_ALL;


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
#pragma omp parallel for schedule(static)
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

        switch (randomization_type)
        {
        case NeuronRandomizationType::RANDOMIZE_ALL:
            m_trigger_values[i] = get_random_value(MIN_TRIGGER, MAX_TRIGGER);
            m_weights[i] = get_random_value(MIN_WEIGHT, MAX_WEIGHT);
            break;
        case NeuronRandomizationType::RANDOMIZE_WEIGHTS_ONLY:
            m_trigger_values[i] = (MAX_TRIGGER - MIN_TRIGGER) /
                2.0f; // set to middle of range to allow both upward and downward adjustments during training
            m_weights[i] = get_random_value(MIN_WEIGHT, MAX_WEIGHT);
            break;
        case NeuronRandomizationType::RANDOMIZE_CONNECTIONS_ONLY:
        case NeuronRandomizationType::NO_RANDOMIZATION:
            m_trigger_values[i] = (MAX_TRIGGER - MIN_TRIGGER) /
                2.0f; // set to middle of range to allow both upward and downward adjustments during training
            m_weights[i] = (MAX_WEIGHT - MIN_WEIGHT) /
                2.0f; // set to middle of range to allow both upward and downward adjustments during training
            break;
        }

        // Each firing neuron fans out to layer_size/corpus_size targets so that,
        // on average, every neuron in the next intermediate layer receives one
        // connection from each corpus token's active input path.
        const size_t layer_size = static_cast<size_t>(IntermediateLayerIndex::MAX);
        assert(m_corpus.number_of_token_types() > 10);
        const size_t corpus_size = m_corpus.number_of_token_types();
        const size_t MAX_INTERMEDIATE_LAYER_CONNECTIONS =
            std::min(MAX_NUM_CONNECTIONS_PER_NEURON, layer_size / corpus_size);
        assert(MAX_INTERMEDIATE_LAYER_CONNECTIONS > 1);
        const size_t MIN_INTERMEDIATE_CONNECTIONS = std::max(size_t{1}, MAX_INTERMEDIATE_LAYER_CONNECTIONS / 10);

        assert(MIN_INTERMEDIATE_CONNECTIONS > 0);
        assert(
            MAX_INTERMEDIATE_LAYER_CONNECTIONS <= 10000
        ); // sanity check to avoid accidentally creating a near-fully-connected layer


        LOG_ONCE(
            std::println(
                "^^^^^^^^^^^^^^ Randomizing neuron {}: trigger = {:.4f}, weight = {:.4f}, num_connection-range = {}-{}",
                static_cast<int>(i),
                m_trigger_values[i],
                m_weights[i],
                MIN_INTERMEDIATE_CONNECTIONS,
                MAX_INTERMEDIATE_LAYER_CONNECTIONS
            )
        );

        const int num_connections = MIN_INTERMEDIATE_CONNECTIONS +
            (std::rand() % (MAX_INTERMEDIATE_LAYER_CONNECTIONS - MIN_INTERMEDIATE_CONNECTIONS + 1));
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
        const int num_connections = 1 + std::rand() % 2;
        std::vector<IntermediateLayerIndex> conns;
        conns.reserve(num_connections);
        for (int c = 0; c < num_connections; ++c)
        {
            const auto random_token_index =
                get_random_enum_value<TokenID>(static_cast<TokenID>(m_corpus.number_of_token_types()));
            assert(static_cast<size_t>(random_token_index) < m_corpus.number_of_token_types());
            conns.push_back(static_cast<IntermediateLayerIndex>(random_token_index));
        }
        m_connections[i] = std::move(conns);
    }

    void IntermediateLayer::forward_neuron_to_output(IntermediateLayerIndex i, OutputLayer& output_layer) const
    {
        const auto input_value = m_inputs[i];
        const auto trigger_value = m_trigger_values[i];
        auto penalty_for_firing = 1.0f;
        if (input_value < trigger_value)
        {
            penalty_for_firing = 0.01f * (trigger_value - input_value);
        }
        // if (input_value < m_trigger_values[i])
        //     return;
        const auto weight = m_weights[i] * penalty_for_firing;
        for (const auto& target : m_connections[i])
        {
            assert(static_cast<size_t>(target) < m_corpus.number_of_token_types());
            const auto token_id = static_cast<TokenID>(target);
            output_layer.m_inputs.add_no_clamp(token_id, weight * input_value);
        }
    }

    void IntermediateLayer::forward_neuron(IntermediateLayerIndex i, IntermediateLayer& next_layer)
    {
        const auto input_value = m_inputs[i];
        auto penalty_for_firing = gaussian_penalty(input_value, m_trigger_values[i], 0.5f);
        const auto added_value = m_weights[i] * penalty_for_firing;
        for (const auto& target : m_connections[i])
        {
            float& cell = next_layer.m_inputs[target];
            float expected = std::atomic_ref<float>{cell}.load(std::memory_order_relaxed);
            float desired;
            do
            {
                desired = std::clamp(expected + added_value, MIN_NEURON_INPUT, MAX_NEURON_INPUT);
            } while (!std::atomic_ref<float>{cell}.compare_exchange_weak(expected, desired, std::memory_order_relaxed));
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
            const auto token_id = static_cast<TokenID>(target);
            assert(static_cast<size_t>(token_id) < m_corpus.number_of_token_types());
            d += delta[token_id];
        }

        const float penalty = gaussian_penalty(m_inputs[i], m_trigger_values[i], 0.5f);
        d *= penalty;

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

        const auto input_value = m_inputs[i];
        const auto trigger_value = m_trigger_values[i];

        const float penalty = gaussian_penalty(input_value, trigger_value, 0.5f);
        d *= penalty;

        const float weight_delta = learning_rate * d * input_value;
        const float trigger_delta = learning_rate * d;
        m_weights.add_with_clamp(i, weight_delta, Range<float>{MIN_WEIGHT, MAX_WEIGHT});
        m_last_weight_delta[i] = weight_delta;
        m_trigger_values.add_with_clamp(i, -trigger_delta, Range<float>{MIN_TRIGGER, MAX_TRIGGER});
        prev_delta.add_no_clamp(i, d * m_weights[i]);
    }

} // namespace rllm
