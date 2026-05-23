#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <print>

namespace rllm
{

    // Radius for picking target neurons in the next intermediate layer.  Using 1/4 of the
    // layer width ensures the gradient fan-out covers ~50% of the layer across two hops
    // instead of staying confined to the <2% band that radius-100 produced.
    static constexpr int INTERMEDIATE_CONNECTION_RADIUS =
        static_cast<int>(IntermediateLayerIndex::MAX) / 4;

    static constexpr float MOMENTUM_BETA = 0.9f;

    // Clip per-connection raw gradient and accumulated velocity to prevent
    // SiLU's unbounded positive outputs from causing runaway updates.
    static constexpr float GRAD_CLIP = 1.0f;
    static constexpr float VEL_CLIP  = 0.1f;

    // Atomically add `delta` to `*cell` without clamping.
    static void atomic_add_unclamped(float& cell, float delta)
    {
        std::atomic_ref<float>{cell}.fetch_add(delta, std::memory_order_relaxed);
    }

    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            randomize_neuron(i);
    }

    void IntermediateLayer::randomize_neuron(IntermediateLayerIndex i) {
        // for each neuron, randomly connect it to 1-4 neurons in the next layer with random weights.
        const auto num_connections = random_int(1, static_cast<int>(NeuronConnectionIndex::MAX));
        assert(num_connections <= static_cast<int>(NeuronConnectionIndex::MAX)); // sanity check to avoid infinite loop in case of a bug
        for (size_t j = 0; j < num_connections; ++j)
        {
            m_connections[i].push_back({
                .target_neuron = get_random_target_neuron_for_intermediate_layer(i),
                .weight = get_random_value(-1.0f, 1.0f)
            });
        }
    }

    void IntermediateLayer::randomize_neuron_to_output(IntermediateLayerIndex i) {
        const auto num_connections = random_int(1, 2);
        assert(num_connections < static_cast<int>(NeuronConnectionIndex::MAX)); // sanity check to avoid infinite loop in case of a bug
        for (size_t j = 0; j < num_connections; ++j)
        {
            m_connections[i].push_back({
                .target_neuron = static_cast<IntermediateLayerIndex>(get_random_target_neuron_for_output_layer(i)),
                .weight = get_random_value(-1.0f, 1.0f)
            });
        }
    }

    IntermediateLayerIndex IntermediateLayer::get_random_target_neuron_for_intermediate_layer(IntermediateLayerIndex from_neuron)
    {
        // for simplicity, we allow connections to any neuron in the next layer.
        // In a more complex implementation, you might want to enforce some structure here.
        int dummy = 0;
        while (true)
        {
            const auto target = get_random_enum_value<IntermediateLayerIndex>();
            //const auto target = get_random_enum_value_centered_around<IntermediateLayerIndex>(from_neuron, INTERMEDIATE_CONNECTION_RADIUS);
            if (! have_connection_to_neuron(from_neuron, target))
                return target;

            dummy++;
            assert(dummy < 1000); // sanity check to avoid infinite loop in case of a bug
        }
    }



    TokenID IntermediateLayer::get_random_target_neuron_for_output_layer(IntermediateLayerIndex from_neuron)
    {
        // for simplicity, we allow connections to any neuron in the output layer.
        // In a more complex implementation, you might want to enforce some structure here.
        while (true)
        {
            const auto target = get_random_enum_value<TokenID>(m_corpus.number_of_token_types());
            if (! have_connection_to_neuron(from_neuron, target))
                return target;
        }
    }

    void IntermediateLayer::set_random_weights_and_connections_to_output_layer()
    {
        // setup the layer JUST before the output layer.
        // Connections are encoded as IntermediateLayerIndex values; the output-layer
        // forward pass maps them to TokenIDs via modulo corpus size.
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            randomize_neuron_to_output(i);
    }



    void IntermediateLayer::rms_normalize_inputs()
    {
        constexpr float eps = 1e-6f;
        const int n = static_cast<int>(IntermediateLayerIndex::MAX);
        float sum_sq = 0.0f;
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            sum_sq += m_inputs[i] * m_inputs[i];
        const float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            m_inputs[i] /= rms;
    }

    void IntermediateLayer::propagate_forward(IntermediateLayer& next_layer)
    {
        rms_normalize_inputs();
        next_layer.fill_inputs(0.0f);
        const int max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < max_i; ++idx)
            forward_neuron(static_cast<IntermediateLayerIndex>(idx), next_layer);
    }

    void IntermediateLayer::propagate_forward_to_output(OutputLayer& output_layer)
    {
        rms_normalize_inputs();
        output_layer.m_inputs.fill(0.0f);
        const int max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < max_i; ++idx)
            forward_neuron_to_output(static_cast<IntermediateLayerIndex>(idx), output_layer);
    }


    void IntermediateLayer::forward_neuron(IntermediateLayerIndex i, IntermediateLayer& next_layer) const
    {
        const auto& conn = m_connections[i];
        const auto input = normal_activation_function(m_inputs[i]);
        for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
        {
            const auto& connection = conn[ci];
            const auto target_idx = connection.target_neuron;
            const float contrib = input * connection.weight;
            atomic_add_unclamped(next_layer.m_inputs[target_idx], contrib);
        }
    }

    void IntermediateLayer::forward_neuron_to_output(IntermediateLayerIndex i, OutputLayer& output_layer) const
    {
        const auto& conn = m_connections[i];
        const auto input = outputlayer_activation_function(m_inputs[i]);
        for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
        {
            const auto& connection = conn[ci];
            const auto target_idx = static_cast<TokenID>(connection.target_neuron);
            assert(target_idx < m_corpus.number_of_token_types());
            const float contrib = input * connection.weight;
            atomic_add_unclamped(output_layer.m_inputs[target_idx], contrib);
        }
    }


    void IntermediateLayer::propagate_backward(
        const template_token_vector<float, IntermediateLayerIndex>& delta,
        template_token_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        const int max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < max_i; ++idx)
        {
            const auto i = static_cast<IntermediateLayerIndex>(idx);
            float neuron_delta = 0.0f;
            const float act = normal_activation_function(m_inputs[i]);
            auto& conn = m_connections[i];
            for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
            {
                auto& connection = conn[ci];
                const auto target_idx = connection.target_neuron;
                assert(target_idx < IntermediateLayerIndex::MAX);
                const auto output_delta = delta[target_idx];
                neuron_delta += output_delta * connection.weight;
                const float grad = std::clamp(output_delta * act, -GRAD_CLIP, GRAD_CLIP);
                connection.velocity = std::clamp(
                    MOMENTUM_BETA * connection.velocity + learning_rate * grad, -VEL_CLIP, VEL_CLIP);
                connection.weight = std::clamp(connection.weight + connection.velocity, -2.0f, 2.0f);
            }

            prev_delta[i] += neuron_delta * activation_grad(m_inputs[i]);
        }
    }

    void IntermediateLayer::propagate_backward_from_output_layer(
        const template_token_vector<float, TokenID>& delta,
        template_token_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        const int max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < max_i; ++idx)
        {
            const auto i = static_cast<IntermediateLayerIndex>(idx);
            float neuron_delta = 0.0f;
            const float act = normal_activation_function(m_inputs[i]);

            auto& conn = m_connections[i];
            for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
            {
                auto& connection = conn[ci];
                const auto target_idx = static_cast<TokenID>(connection.target_neuron);
                assert(target_idx < m_corpus.number_of_token_types());
                const auto output_delta = delta[target_idx];
                neuron_delta += output_delta * connection.weight;
                const float grad = std::clamp(output_delta * act, -GRAD_CLIP, GRAD_CLIP);
                connection.velocity = std::clamp(
                    MOMENTUM_BETA * connection.velocity + learning_rate * grad, -VEL_CLIP, VEL_CLIP);
                connection.weight = std::clamp(connection.weight + connection.velocity, -2.0f, 2.0f);
            }

            prev_delta[i] += neuron_delta * activation_grad(m_inputs[i]);
        }
    }

} // namespace rllm
