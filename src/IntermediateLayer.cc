#include <IntermediateLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
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

    void IntermediateLayer::set_random_weights_and_connections()
    {
        for (const auto i : enum_iterator<IntermediateLayerIndex>())
        {
            randomize_neuron(i);
        }
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
        m_attn_weights[i] = get_random_value(-1.0f, 1.0f);
        m_attn_vel[i]     = 0.0f;
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
            const auto target = get_random_enum_value<TokenID>(TokenID::MAX);
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
        constexpr int n = static_cast<int>(IntermediateLayerIndex::MAX);
        float sum_sq = 0.0f;
#pragma omp simd reduction(+:sum_sq)
        for (int i = 0; i < n; ++i)
            sum_sq += m_inputs[static_cast<IntermediateLayerIndex>(i)] * m_inputs[static_cast<IntermediateLayerIndex>(i)];
        const float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
#pragma omp simd
        for (int i = 0; i < n; ++i)
            m_inputs[static_cast<IntermediateLayerIndex>(i)] /= rms;
    }

    void IntermediateLayer::propagate_forward(IntermediateLayer& next_layer)
    {
        rms_normalize_inputs();
        constexpr int n_src = static_cast<int>(IntermediateLayerIndex::MAX);
        constexpr int n_dst = n_src;

        // Per-thread accumulator buffers: each thread writes to its own slice,
        // eliminating atomic contention on next_layer.m_inputs.
        const int n_threads = omp_get_max_threads();
        std::vector<float> local_sums(static_cast<size_t>(n_threads) * static_cast<size_t>(n_dst), 0.0f);

#pragma omp parallel
        {
            float* my = local_sums.data() + omp_get_thread_num() * n_dst;
#pragma omp for schedule(static)
            for (int idx = 0; idx < n_src; ++idx)
            {
                const auto i    = static_cast<IntermediateLayerIndex>(idx);
                const float x   = m_inputs[i];
                const float act = normal_activation_function(x) * attn_gate(x, m_attn_weights[i]);
                const auto& conn = m_connections[i];
                for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
                    my[static_cast<int>(conn[ci].target_neuron)] += act * conn[ci].weight;
            }
        }

        // Reduce thread partials into next_layer.m_inputs
        next_layer.fill_inputs(0.0f);
        for (int t = 0; t < n_threads; ++t)
        {
            const float* p = local_sums.data() + t * n_dst;
#pragma omp simd
            for (int i = 0; i < n_dst; ++i)
                next_layer.m_inputs[static_cast<IntermediateLayerIndex>(i)] += p[i];
        }
    }

    void IntermediateLayer::propagate_forward_to_output(OutputLayer& output_layer)
    {
        rms_normalize_inputs();
        constexpr int n_src = static_cast<int>(IntermediateLayerIndex::MAX);
        constexpr int n_dst = static_cast<int>(TokenID::MAX);

        const int n_threads = omp_get_max_threads();
        std::vector<float> local_sums(static_cast<size_t>(n_threads) * static_cast<size_t>(n_dst), 0.0f);

#pragma omp parallel
        {
            float* my = local_sums.data() + omp_get_thread_num() * n_dst;
#pragma omp for schedule(static)
            for (int idx = 0; idx < n_src; ++idx)
            {
                const auto i    = static_cast<IntermediateLayerIndex>(idx);
                const float x   = m_inputs[i];
                const float act = outputlayer_activation_function(x) * attn_gate(x, m_attn_weights[i]);
                const auto& conn = m_connections[i];
                for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
                {
                    const auto target_idx = static_cast<TokenID>(conn[ci].target_neuron);
                    assert(target_idx < TokenID::MAX);
                    my[static_cast<int>(target_idx)] += act * conn[ci].weight;
                }
            }
        }

        output_layer.m_inputs.fill(0.0f);
        for (int t = 0; t < n_threads; ++t)
        {
            const float* p = local_sums.data() + t * n_dst;
#pragma omp simd
            for (int i = 0; i < n_dst; ++i)
                output_layer.m_inputs[static_cast<TokenID>(i)] += p[i];
        }

        output_layer.rms_normalize_inputs();
    }




    void IntermediateLayer::propagate_backward(
        const template_vector<float, IntermediateLayerIndex>& delta,
        template_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        const int max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < max_i; ++idx)
        {
            const auto i = static_cast<IntermediateLayerIndex>(idx);
            const float x    = m_inputs[i];
            const float a    = m_attn_weights[i];
            const float g    = attn_gate(x, a);          // sigmoid(x*a)
            const float dg   = gate_grad_from_value(g);  // g*(1-g)
            const float act  = normal_activation_function(x) * g;

            float neuron_delta = 0.0f;
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

            // Update attention gate weight: ∂L/∂a_i = neuron_delta * silu(x) * dg * x
            const float attn_grad = std::clamp(
                neuron_delta * normal_activation_function(x) * dg * x, -GRAD_CLIP, GRAD_CLIP);
            m_attn_vel[i]     = std::clamp(MOMENTUM_BETA * m_attn_vel[i] + learning_rate * attn_grad, -VEL_CLIP, VEL_CLIP);
            m_attn_weights[i] = std::clamp(m_attn_weights[i] + m_attn_vel[i], -2.0f, 2.0f);

            // prev_delta: gradient through both silu and the gate
            // ∂L/∂x_i = neuron_delta * (silu_grad(x)*g + silu(x)*dg*a)
            const float dx = neuron_delta * (activation_grad(x) * g + normal_activation_function(x) * dg * a);
            prev_delta[i] += dx;
        }
    }

    void IntermediateLayer::propagate_backward_from_output_layer(
        const template_vector<float, TokenID>& delta,
        template_vector<float, IntermediateLayerIndex>& prev_delta,
        float learning_rate
    )
    {
        const int max_i = static_cast<int>(IntermediateLayerIndex::MAX);
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < max_i; ++idx)
        {
            const auto i = static_cast<IntermediateLayerIndex>(idx);
            const float x    = m_inputs[i];
            const float a    = m_attn_weights[i];
            const float g    = attn_gate(x, a);
            const float dg   = gate_grad_from_value(g);

            float neuron_delta = 0.0f;
            auto& conn = m_connections[i];
            for (const auto ci : enum_iterator<NeuronConnectionIndex>(conn.size()))
            {
                auto& connection = conn[ci];
                const auto target_idx = static_cast<TokenID>(connection.target_neuron);
                assert(target_idx < TokenID::MAX);
                const auto output_delta = delta[target_idx];
                neuron_delta += output_delta * connection.weight;
                const float act  = normal_activation_function(x) * g;
                const float grad = std::clamp(output_delta * act, -GRAD_CLIP, GRAD_CLIP);
                connection.velocity = std::clamp(
                    MOMENTUM_BETA * connection.velocity + learning_rate * grad, -VEL_CLIP, VEL_CLIP);
                connection.weight = std::clamp(connection.weight + connection.velocity, -2.0f, 2.0f);
            }

            // Update attention gate weight
            const float attn_grad = std::clamp(
                neuron_delta * normal_activation_function(x) * dg * x, -GRAD_CLIP, GRAD_CLIP);
            m_attn_vel[i]     = std::clamp(MOMENTUM_BETA * m_attn_vel[i] + learning_rate * attn_grad, -VEL_CLIP, VEL_CLIP);
            m_attn_weights[i] = std::clamp(m_attn_weights[i] + m_attn_vel[i], -2.0f, 2.0f);

            const float dx = neuron_delta * (activation_grad(x) * g + normal_activation_function(x) * dg * a);
            prev_delta[i] += dx;
        }
    }

} // namespace rllm
