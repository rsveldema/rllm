#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <print>

namespace rllm
{
    static constexpr auto MAX_SANE_CONNECTIONS = 100; // sanity check to avoid infinite loops in case of a bug

    // Radius for picking target neurons in the next intermediate layer.  Using 1/4 of the
    // layer width ensures the gradient fan-out covers ~50% of the layer across two hops
    // instead of staying confined to the <2% band that radius-100 produced.
    static constexpr int INTERMEDIATE_CONNECTION_RADIUS =
        static_cast<int>(IntermediateLayerIndex::MAX) / 4;

    static constexpr float MOMENTUM_BETA = 0.9f;

    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            randomize_neuron(i);
    }

    void IntermediateLayer::randomize_neuron(IntermediateLayerIndex i) {
        // for each neuron, randomly connect it to 1-4 neurons in the next layer with random weights.
        const auto num_connections = random_int(1, MAX_SANE_CONNECTIONS);
        assert(num_connections <= MAX_SANE_CONNECTIONS); // sanity check to avoid infinite loop in case of a bug
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
        assert(num_connections < MAX_SANE_CONNECTIONS); // sanity check to avoid infinite loop in case of a bug
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



    void IntermediateLayer::propagate_forward(IntermediateLayer& next_layer)
    {
        next_layer.fill_inputs(0.0f);
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            forward_neuron(i, next_layer);
    }

    void IntermediateLayer::propagate_forward_to_output(OutputLayer& output_layer) const
    {
        output_layer.m_inputs.fill(0.0f);
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
            forward_neuron_to_output(i, output_layer);
    }


    void IntermediateLayer::forward_neuron(IntermediateLayerIndex i, IntermediateLayer& next_layer) const
    {
        const auto input = normal_activation_function(m_inputs[i]);
        for (const auto& connection : m_connections[i])
        {
            const auto target_idx = connection.target_neuron;
            const auto weight = connection.weight;
            next_layer.accumulate_input(target_idx, input * weight, {-1.0f, 1.0f});
        }
    }

    void IntermediateLayer::forward_neuron_to_output(IntermediateLayerIndex i, OutputLayer& output_layer) const
    {
        const auto input = outputlayer_activation_function(m_inputs[i]);
        for (const auto& connection : m_connections[i])
        {
            const auto target_idx = static_cast<TokenID>(connection.target_neuron);
            assert(target_idx < m_corpus.number_of_token_types());

            const auto weight = connection.weight;
            output_layer.accumulate_input(target_idx, input * weight, {-1.0f, 1.0f});
        }
    }


        void IntermediateLayer::propagate_backward(
            const template_token_vector<float, IntermediateLayerIndex>& delta,
            template_token_vector<float, IntermediateLayerIndex>& prev_delta,
            float learning_rate
        ){

             // This function should propagate the error (delta) backward through the layer,
             // updating the weights of the connections and accumulating the delta for the previous layer.
             // The learning_rate parameter should be used to scale the weight updates.

             // For each neuron in this layer, we need to calculate how much it contributed to the error in the next layer,
             // and update the weights accordingly. We also need to accumulate the delta for the previous layer.

             // This is a bit complex, so let's break it down step by step.

             // 1. For each neuron in this layer, we will look at its connections to the next layer.
             // 2. For each connection, we will calculate how much that connection contributed to the error in the next layer.
             // 3. We will then update the weight of that connection based on the input to that connection and the error.
             // 4. We will also accumulate the delta for the previous layer based on how much this neuron contributed to the error.

            for (const auto i : enum_iterator<IntermediateLayerIndex>()) {
                float neuron_delta = 0.0f;

                for (auto& connection : m_connections[i])
                {
                    const auto target_idx = connection.target_neuron;
                    assert(target_idx < IntermediateLayerIndex::MAX);

                    const auto weight = connection.weight;
                    const auto output_delta = delta[target_idx];

                    // Accumulate the delta for this neuron based on the output layer's delta and the connection weight
                    neuron_delta += output_delta * weight;

                    // Update the weight using SGD with momentum.
                    const auto input = normal_activation_function(m_inputs[i]);
                    const auto weight_update = learning_rate * output_delta * input;
                    connection.velocity = MOMENTUM_BETA * connection.velocity + weight_update;
                    connection.weight = std::clamp(connection.weight + connection.velocity, -2.0f, 2.0f);
                }

                // Gate the upstream gradient by the Leaky ReLU derivative.
                prev_delta[i] += neuron_delta * activation_grad(m_inputs[i]);
            }
        }

        void IntermediateLayer::propagate_backward_from_output_layer(
            const template_token_vector<float, TokenID>& delta,
            template_token_vector<float, IntermediateLayerIndex>& prev_delta,
            float learning_rate
        ){
            // the output layer calls this function to propagate the error backward to the last intermediate layer.
            // This function is similar to propagate_backward, but it needs to map the output layer's TokenID-based delta to
            // the IntermediateLayerIndex-based connections.

             // For each neuron in this layer, we will look at its connections to the output layer.
             // For each connection, we will calculate how much that connection contributed to the error in the output layer.
             // We will then update the weight of that connection based on the input to that connection and the error.
             // We will also accumulate the delta for the previous layer based on how much this neuron contributed to the error.

            for (const auto i : enum_iterator<IntermediateLayerIndex>()) {
                float neuron_delta = 0.0f;

                for (auto& connection : m_connections[i])
                {
                    const auto target_idx = static_cast<TokenID>(connection.target_neuron);
                    assert(target_idx < m_corpus.number_of_token_types());

                    const auto weight = connection.weight;
                    const auto output_delta = delta[target_idx];

                    // Accumulate the delta for this neuron based on the output layer's delta and the connection weight
                    neuron_delta += output_delta * weight;

                    // Update the weight using SGD with momentum.
                    const auto input = normal_activation_function(m_inputs[i]);
                    const auto weight_update = learning_rate * output_delta * input;
                    connection.velocity = MOMENTUM_BETA * connection.velocity + weight_update;
                    connection.weight = std::clamp(connection.weight + connection.velocity, -2.0f, 2.0f);
                }

                // Gate the upstream gradient by the Leaky ReLU derivative.
                prev_delta[i] += neuron_delta * activation_grad(m_inputs[i]);
            }
        }

} // namespace rllm
