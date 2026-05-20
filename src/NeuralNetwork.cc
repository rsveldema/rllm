#include <RLLM.hpp>
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
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k, Corpus& corpus) const
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

        auto delta = std::make_unique<template_token_matrix<float, IntermediateLayerIndex, PositionIndex>>();
        auto prev_delta = std::make_unique<template_token_matrix<float, IntermediateLayerIndex, PositionIndex>>();
        m_intermediate_layers.back().propagate_backward(*output_layer_delta, *delta, LEARNING_RATE);

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

        // Packs a (IntermediateLayerIndex, PositionIndex) pair into a flat index.
        constexpr size_t POS_MAX = static_cast<size_t>(PositionIndex::MAX);
        constexpr size_t LAYER_SIZE = static_cast<size_t>(IntermediateLayerIndex::MAX) * POS_MAX;
        auto pack = [](IntermediateLayerIndex i, PositionIndex p) {
            return static_cast<size_t>(i) * POS_MAX + static_cast<size_t>(p);
        };

        // Step 1: find nodes in the last intermediate layer that structurally connect to token_id.
        std::vector<bool> target(LAYER_SIZE, false);
        const auto& last = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const auto [conn, _conn_pos] = last.m_connections.get(i, pos);
                if (static_cast<TokenID>(conn) == token_id)
                    target[pack(i, pos)] = true;
            }
        }
        if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
            return false;

        // Step 2: walk backwards through the remaining intermediate layers.
        // A node in layer l is "needed" if its connection leads to a needed node in layer l+1.
        for (int l = static_cast<int>(m_intermediate_layers.size()) - 2; l >= 0; --l)
        {
            std::vector<bool> prev(LAYER_SIZE, false);
            const auto& layer = m_intermediate_layers[l];
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
            {
                for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
                {
                    const auto [next_i, next_pos] = layer.m_connections.get(i, pos);
                    if (target[pack(next_i, next_pos)])
                        prev[pack(i, pos)] = true;
                }
            }
            target = std::move(prev);
            if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
                return false;
        }

        // Step 3: check if any input layer connection targets a needed node in layer 0.
        for (auto tok = TokenID::START; tok < TokenID::MAX; tok = inc(tok))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const auto [next_i, next_pos] = m_input_layer.m_connections.get(tok, pos);
                if (target[pack(next_i, next_pos)])
                    return true;
            }
        }
        return false;
    }

    bool NeuralNetwork::has_active_path_to_token(TokenID token_id) const
    {
        assert(!m_intermediate_layers.empty());

        constexpr size_t POS_MAX = static_cast<size_t>(PositionIndex::MAX);
        constexpr size_t LAYER_SIZE = static_cast<size_t>(IntermediateLayerIndex::MAX) * POS_MAX;
        auto pack = [](IntermediateLayerIndex i, PositionIndex p) {
            return static_cast<size_t>(i) * POS_MAX + static_cast<size_t>(p);
        };

        // Step 1: find nodes in the last layer that FIRED and connect to token_id.
        std::vector<bool> target(LAYER_SIZE, false);
        const auto& last = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                if (last.m_inputs.get(i, pos) < last.m_trigger_values.get(i, pos))
                    continue;
                const auto [conn, _conn_pos] = last.m_connections.get(i, pos);
                if (static_cast<TokenID>(conn) == token_id)
                    target[pack(i, pos)] = true;
            }
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
                for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
                {
                    if (layer.m_inputs.get(i, pos) < layer.m_trigger_values.get(i, pos))
                        continue;
                    const auto [next_i, next_pos] = layer.m_connections.get(i, pos);
                    if (target[pack(next_i, next_pos)])
                        prev[pack(i, pos)] = true;
                }
            }
            target = std::move(prev);
            if (std::none_of(target.begin(), target.end(), [](bool b) { return b; }))
                return false;
        }

        // Step 3: check if any active input connects to an active-path node in layer 0.
        for (auto tok = TokenID::START; tok < TokenID::MAX; tok = inc(tok))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                if (m_input_layer.m_inputs.get(tok, pos) <= 0.0f)
                    continue;
                const auto [next_i, next_pos] = m_input_layer.m_connections.get(tok, pos);
                if (target[pack(next_i, next_pos)])
                    return true;
            }
        }
        return false;
    }

    void NeuralNetwork::dump_path_weights_and_triggers(TokenID token_id) const
    {
        assert(!m_intermediate_layers.empty());

        constexpr size_t POS_MAX = static_cast<size_t>(PositionIndex::MAX);
        constexpr size_t LAYER_SIZE = static_cast<size_t>(IntermediateLayerIndex::MAX) * POS_MAX;
        auto pack = [](IntermediateLayerIndex i, PositionIndex p) {
            return static_cast<size_t>(i) * POS_MAX + static_cast<size_t>(p);
        };

        const size_t num_layers = m_intermediate_layers.size();

        // Build per-layer path sets via backward walk from token_id.
        std::vector<std::vector<bool>> path(num_layers, std::vector<bool>(LAYER_SIZE, false));

        const auto& last_layer = m_intermediate_layers.back();
        for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
            {
                const auto [conn, _conn_pos] = last_layer.m_connections.get(i, pos);
                if (static_cast<TokenID>(conn) == token_id)
                    path[num_layers - 1][pack(i, pos)] = true;
            }
        }

        for (int l = static_cast<int>(num_layers) - 2; l >= 0; --l)
        {
            const auto& layer = m_intermediate_layers[l];
            for (auto i = IntermediateLayerIndex::START; i < IntermediateLayerIndex::MAX; i = inc(i))
            {
                for (auto pos = PositionIndex::START; pos < PositionIndex::MAX; pos = inc(pos))
                {
                    const auto [next_i, next_pos] = layer.m_connections.get(i, pos);
                    if (path[l + 1][pack(next_i, next_pos)])
                        path[l][pack(i, pos)] = true;
                }
            }
        }

        // Input layer: nodes whose connection targets a path node in layer 0.
        std::println("  [input] nodes on path to token {} (capped at 10):", static_cast<int>(token_id));
        size_t count = 0;
        for (auto tok = TokenID::START; tok < TokenID::MAX && count < 10; tok = inc(tok))
        {
            for (auto pos = PositionIndex::START; pos < PositionIndex::MAX && count < 10; pos = inc(pos))
            {
                const auto [next_i, next_pos] = m_input_layer.m_connections.get(tok, pos);
                if (!path[0][pack(next_i, next_pos)])
                    continue;
                const float input = m_input_layer.m_inputs.get(tok, pos);
                std::println(
                    "    input(tok={}, pos={}) val={:.4f} -> layer0({},{})",
                    static_cast<int>(tok), static_cast<size_t>(pos), input,
                    static_cast<size_t>(next_i), static_cast<size_t>(next_pos)
                );
                ++count;
            }
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
                for (auto pos = PositionIndex::START; pos < PositionIndex::MAX && n < 10; pos = inc(pos))
                {
                    if (!path[l][pack(i, pos)])
                        continue;
                    const float inp     = layer.m_inputs.get(i, pos);
                    const float trigger = layer.m_trigger_values.get(i, pos);
                    const float weight  = layer.m_weights.get(i, pos);
                    std::println(
                        "    neuron({},{}) input={:.4f} trigger={:.4f} weight={:.4f} fired={}",
                        static_cast<size_t>(i), static_cast<size_t>(pos),
                        inp, trigger, weight, inp >= trigger
                    );
                    ++n;
                }
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

    void NeuralNetwork::set_random_weights_and_connections(Corpus& corpus)
    {
        m_input_layer.set_random_weights_and_connections();

        for (size_t i = 0; i < m_intermediate_layers.size() - 1; ++i)
        {
            m_intermediate_layers[i].set_random_weights_and_connections();
        }
        m_intermediate_layers.back().set_random_weights_and_connections_to_output_layer(corpus);
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

    void NeuralNetwork::dump_top_predictions(Corpus& corpus)
    {
        int prediction_index = 0;
        const auto predicted_token_id_lists = get_best_output_token_ids(5, corpus);
        for (const auto& entry : predicted_token_id_lists)
        {
            const auto predicted_token = corpus.get_token_from_id(entry.token_id);
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

    void NeuralNetwork::train(Corpus& corpus)
    {
        std::println("Training the neural network...");

        set_random_weights_and_connections(corpus);

        // Get a training example from the corpus. The example needs at least 2
        // tokens. Note that the list can be padded to 2 tokens using
        // UNKNOWN_TOKEN_ID if necessary.
        const auto train_output = corpus.get_training_input_line(3);
        auto train_input = train_output;
        const auto expected_output_token = train_input.back();
        train_input.pop_back();

        const auto full_string = corpus.get_line(train_output);

        size_t num_iterations = 1000000;
        for (size_t i = 0; i < num_iterations; ++i)
        {
            Score score;
            set_input_layer(train_input);
            propagate_forward();
            compute_score(score, expected_output_token);
            propagate_backward(score);

            if (i % 100 == 0)
            {
                const auto expected_token = corpus.get_token_from_id(expected_output_token);
                std::println(
                    "Training iteration[{}], wanted: '{}' ({}), full string: '{}'",
                    i,
                    expected_token,
                    expected_output_token,
                    full_string
                );
                dump_weights_and_triggers_for_token(expected_output_token);
                dump_top_predictions(corpus);
            }
        }
    }

} // namespace rllm
