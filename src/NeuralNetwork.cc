#include <NeuralNetwork.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <print>

namespace rllm
{
    static size_t clip_max(size_t value, size_t max)
    {
        if (value > max)
            return max;
        return value;
    }

    // Layers

    void NeuralNetwork::propagate_forward()
    {
        assert(!m_intermediate_layers.empty());

        // Propagate from input layer into the first intermediate layer.
        m_input_layer.propagate_forward(m_intermediate_layers.front());

        // Propagate across intermediate layers.
        for (size_t i = 0; i < (m_intermediate_layers.size() - 1); ++i)
        {
            m_intermediate_layers[i].propagate_forward(m_intermediate_layers[i + 1]);
        }

        // Propagate from the last intermediate layer into the output layer.
        m_intermediate_layers.back().propagate_forward_to_output(m_output_layer);
    }


    // NeuralNetwork

    static void try_add_to_top_k(std::vector<OutputToken>& top_k, TokenID id, float value, size_t k)
    {
        if (top_k.size() < k)
        {
            top_k.emplace_back(id, value);
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
        else if (!top_k.empty() && value > top_k.back().activation)
        {
            top_k.back() = {id, value};
            std::sort(top_k.begin(), top_k.end(), [](const auto& a, const auto& b) {
                return a.activation > b.activation;
            });
        }
    }

    void NeuralNetwork::set_input_layer(const InputLine& input)
    {
        assert(!m_intermediate_layers.empty());
        m_input_layer.set_input_layer(input);
    }

    // returns the top-K with the biggest activation in the output layer,
    // sorted descending by activation.
    // if no prediction can be made, returns an empty list.
    // Do not try to return any if the activation is 0 or negative, as that means the token did not activate at all.
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k) const
    {
        assert(!m_intermediate_layers.empty());

        std::vector<OutputToken> top_k_pairs;
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i))
        {
            const auto activation = m_output_layer.m_inputs[i];
            if (activation <= 0.0f)
            {
                continue; // skip tokens that did not activate at all
            }
            try_add_to_top_k(top_k_pairs, i, activation, top_k);
        }
        return top_k_pairs;
    }

    void NeuralNetwork::compute_score(Score& score, const TokenID expected_output_token)
    {
        m_output_layer.compute_score(score, expected_output_token);
    }


    void NeuralNetwork::propagate_backward(const Score& score)
    {
        static constexpr float LEARNING_RATE = 0.01f;

        // delta[i] = signed error: target - actual.
        // Positive  → neuron fires too little  → increase weight.
        // Negative  → neuron fires too much    → decrease weight.

        auto output_layer_delta = std::make_unique<template_token_vector<float, TokenID>>();
        static_assert(sizeof(*output_layer_delta) < 65536, "output_layer_delta is too large for the stack");
        m_output_layer.compute_deltas(score, *output_layer_delta);

        auto delta = std::make_unique<template_token_vector<float, IntermediateLayerIndex>>();
        auto prev_delta = std::make_unique<template_token_vector<float, IntermediateLayerIndex>>();
        m_intermediate_layers.back().propagate_backward_from_output_layer(*output_layer_delta, *delta, LEARNING_RATE);

        // Walk backwards through the remaining layers.
        assert(m_intermediate_layers.size() >= 2);
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            prev_delta->fill(0.0f);
            m_intermediate_layers[l].propagate_backward(*delta, *prev_delta, LEARNING_RATE);
            *delta = *prev_delta;
        }
    }

    bool NeuralNetwork::output_is_reachable_from_inputs(TokenID token_id) const
    {
        assert(!m_intermediate_layers.empty());
        constexpr size_t LAYER_SIZE = static_cast<size_t>(IntermediateLayerIndex::MAX);

        // Step 1: find nodes in the last intermediate layer that structurally connect to token_id.
        std::vector<bool> target(LAYER_SIZE, false);
        const auto& last = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (const auto& conn : last.m_connections[i])
                if (static_cast<TokenID>(conn) == token_id)
                    { target[static_cast<size_t>(i)] = true; break; }
        }
        if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
            return false;

        // Step 2: walk backwards through the remaining intermediate layers.
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            std::vector<bool> prev(LAYER_SIZE, false);
            const auto& layer = m_intermediate_layers[l];
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
            {
                for (const auto& next_i : layer.m_connections[i])
                    if (target[static_cast<size_t>(next_i)])
                        { prev[static_cast<size_t>(i)] = true; break; }
            }
            target = std::move(prev);
            if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
                return false;
        }

        // Step 3: any remaining target in layer 0 is reachable from inputs via the hash.
        return true;
    }

    bool NeuralNetwork::has_active_path_to_token(TokenID token_id) const
    {
        assert(!m_intermediate_layers.empty());
        constexpr size_t LAYER_SIZE = static_cast<size_t>(IntermediateLayerIndex::MAX);

        // Step 1: find nodes in the last layer that FIRED and connect to token_id.
        std::vector<bool> target(LAYER_SIZE, false);
        const auto& last = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            if (last.m_inputs[i] < last.m_trigger_values[i])
                continue;
            for (const auto& conn : last.m_connections[i])
                if (static_cast<TokenID>(conn) == token_id)
                    { target[static_cast<size_t>(i)] = true; break; }
        }
        if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
            return false;

        // Step 2: walk backwards — a node counts only if it fired AND leads to an active target.
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            std::vector<bool> prev(LAYER_SIZE, false);
            const auto& layer = m_intermediate_layers[l];
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
            {
                if (layer.m_inputs[i] < layer.m_trigger_values[i])
                    continue;
                for (const auto& next_i : layer.m_connections[i])
                    if (target[static_cast<size_t>(next_i)])
                        { prev[static_cast<size_t>(i)] = true; break; }
            }
            target = std::move(prev);
            if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
                return false;
        }

        // Step 3: check if any active input hashes to a target neuron in layer 0.
        for (PositionIndex pos = PositionIndex::START;
             pos < m_input_layer.m_input.size();
             pos = inc(pos))
        {
            const TokenID tok = m_input_layer.m_input[pos];
            if (tok == TokenID::UNKNOWN_TOKEN_ID)
                continue;
            const auto hashed = InputLayer::hash_input(tok, pos);
            if (target[static_cast<size_t>(hashed)])
                return true;
        }
        return false;
    }

    void NeuralNetwork::dump_neurons_whose_weights_were_increasing() const
    {
        for (size_t l = 0; l < m_intermediate_layers.size(); ++l)
        {
            const auto& layer = m_intermediate_layers[l];
            size_t count = 0;
            std::println("  [layer {}] neurons with increasing weights (capped at 10):", l);
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX && count < 10; i = inc(i))
            {
                const float delta = layer.m_last_weight_delta[i];
                if (delta <= 0.0f)
                    continue;
                std::println(
                    "    neuron({}) weight={:.4f} delta=+{:.6f}",
                    static_cast<size_t>(i),
                    layer.m_weights[i], delta
                );
                ++count;
            }
            if (count == 0)
                std::println("    (none)");
            else if (count >= 10)
                std::println("    ... (capped)");
        }
    }

    void NeuralNetwork::dump_path_weights_and_triggers(TokenID token_id) const
    {
        assert(!m_intermediate_layers.empty());

        constexpr size_t LAYER_SIZE = static_cast<size_t>(IntermediateLayerIndex::MAX);
        const size_t num_layers = m_intermediate_layers.size();

        // Build per-layer path sets via backward walk from token_id.
        std::vector<std::vector<bool>> path(num_layers, std::vector<bool>(LAYER_SIZE, false));

        const auto& last_layer = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (const auto& conn : last_layer.m_connections[i])
                if (static_cast<TokenID>(conn) == token_id)
                    { path[num_layers - 1][static_cast<size_t>(i)] = true; break; }
        }

        for (int l = static_cast<int>(num_layers) - 2; l >= 0; --l)
        {
            const auto& layer = m_intermediate_layers[l];
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
            {
                for (const auto& next_i : layer.m_connections[i])
                    if (path[l + 1][static_cast<size_t>(next_i)])
                        { path[l][static_cast<size_t>(i)] = true; break; }
            }
        }

        // Input layer: nodes whose hash maps to a path node in layer 0.
        std::println("  [input] nodes on path to token {} (capped at 10):", static_cast<int>(token_id));
        size_t count = 0;
        for (PositionIndex pos = PositionIndex::START;
             pos < m_input_layer.m_input.size() && count < 10;
             pos = inc(pos))
        {
            const TokenID tok = m_input_layer.m_input[pos];
            if (tok == TokenID::UNKNOWN_TOKEN_ID)
                continue;
            const auto target_neuron = InputLayer::hash_input(tok, pos);
            if (!path[0][static_cast<size_t>(target_neuron)])
                continue;
            const float input_val = m_input_layer.get_input_value(tok, pos);
            std::println(
                "    input(tok={}, pos={}) val={:.4f} -> layer0({})",
                static_cast<int>(tok), static_cast<size_t>(pos), input_val,
                static_cast<size_t>(target_neuron)
            );
            ++count;
        }
        if (count >= 10) std::println("    ... (capped)");

        // Each intermediate layer: path nodes only.
        auto dump_layer = [&](size_t l)
        {
            std::println("  [layer {}] path nodes (capped at 10):", l);
            size_t n = 0;
            const auto& layer = m_intermediate_layers[l];
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX && n < 10; i = inc(i))
            {
                if (!path[l][static_cast<size_t>(i)])
                    continue;
                const float inp     = layer.m_inputs[i];
                const float trigger = layer.m_trigger_values[i];
                const float weight  = layer.m_weights[i];
                std::println(
                    "    neuron({}) input={:.4f} trigger={:.4f} weight={:.4f} fired={}",
                    static_cast<size_t>(i), inp, trigger, weight, inp >= trigger
                );
                ++n;
            }
            if (n >= 10) std::println("    ... (capped)");
        };

        for (size_t l = 0; l < num_layers; ++l)
            dump_layer(l);

        std::println(
            "  [output] token {} : activation={:.4f}",
            static_cast<int>(token_id), m_output_layer.m_inputs[token_id]
        );
    }

    void NeuralNetwork::set_random_weights_and_connections()
    {
        for (size_t i = 0; i < m_intermediate_layers.size() - 1; ++i)
        {
            m_intermediate_layers[i].set_random_weights_and_connections();
        }
        m_intermediate_layers.back().set_random_weights_and_connections_to_output_layer();
    }

    void NeuralNetwork::dump_weights_and_triggers_for_token(TokenID token_id)
    {
        if (!output_is_reachable_from_inputs(token_id))
        {
            std::println(
                "Output layer token {} has no structural path from inputs. No connections to dump.",
                static_cast<int>(token_id)
            );
            return;
        }
        const bool active = has_active_path_to_token(token_id);
        std::println(
            "token {} is structurally reachable, active path: {}",
            static_cast<int>(token_id), active
        );
        if (!active)
            std::println("  (no neurons firing along the path yet — weights/triggers not trained)");
        dump_path_weights_and_triggers(token_id);
    }

    void NeuralNetwork::dump_top_predictions()
    {
        int prediction_index = 0;
        const auto predicted_token_id_lists = get_best_output_token_ids(5);
        for (const auto& entry : predicted_token_id_lists)
        {
            const auto predicted_token = m_corpus.get_token_from_id(entry.token_id);
            std::println(
                "\t prediction[{} of {}] / pred:'{}' (id: '{}'), {}",
                prediction_index,
                predicted_token_id_lists.size(),
                predicted_token,
                entry.token_id,
                entry.activation
            );
            prediction_index++;
        }
    }

    // Compute mean squared error loss between output activations and expected output
    float NeuralNetwork::compute_loss(TokenID expected_output_token) const
    {
        float loss = 0.0f;
        for (auto i = TokenID::START; i < TokenID::MAX; i = inc(i)) {
            float target = (i == expected_output_token) ? 1.0f : 0.0f;
            float pred = m_output_layer.m_inputs[i];
            float diff = pred - target;
            loss += diff * diff;
        }
        return loss / static_cast<float>(static_cast<int>(TokenID::MAX));
    }

    void NeuralNetwork::train(bool verbose)
    {
        std::println("Training the neural network...");

        set_random_weights_and_connections();

        // Get a training example from the corpus. The example needs at least 2
        // tokens. Note that the list can be padded to 2 tokens using
        // UNKNOWN_TOKEN_ID if necessary.
        const auto train_output = m_corpus.get_training_input_line(3);
        assert(static_cast<int>(train_output.size()) >= 2);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string = m_corpus.get_line(train_output);

        size_t num_iterations = 1000000;
        for (size_t i = 0; i < num_iterations; ++i)
        {
            Score score;
            set_input_layer(train_input);
            propagate_forward();
            compute_score(score, expected_output_token);
            propagate_backward(score);

            if (verbose && i % 100 == 0)
            {
                float loss = compute_loss(expected_output_token);
                const auto expected_token = m_corpus.get_token_from_id(expected_output_token);
                std::println(
                    "Training iteration[{}], wanted: '{}' ({}), full string: '{}'",
                    i,
                    expected_token,
                    expected_output_token,
                    full_string
                );
                std::println("  Loss: {:.6f}", loss);
                dump_neurons_whose_weights_were_increasing();
                //dump_weights_and_triggers_for_token(expected_output_token);
                dump_top_predictions();
            }
        }
    }

} // namespace rllm
