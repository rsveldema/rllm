#include <gtest/gtest.h>

#include <OutputLayer.hpp>
#include <RuntimeConfig.hpp>
#include <rllm_vulkan_runtime.hpp>

#include <cpu/cpu_fixed_vector.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <enum_iterator1D.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

TEST(OutputLayerBatchTest, BatchedForwardMatchesIndependentForwards)
{
    using namespace rllm;
    OutputLayer layer;
    layer.set_random_weights();
    auto& queue = vulkan_runtime::get_queue(0);

    cpu_fixed_matrix<float, BatchIndex, EmbeddingDimension> h_cpu;
    for (size_t batch = 0; batch < 2; ++batch)
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            h_cpu.set(static_cast<BatchIndex>(batch), d, static_cast<float>((batch + 1) * (static_cast<size_t>(d) + 1)) / 4096.0f);

    fixed_size_matrix<float, BatchIndex, EmbeddingDimension> h;
    h.copy_from_cpu(queue, h_cpu);
    fixed_size_matrix<float, BatchIndex, TokenID> logits;
    layer.forward_batched(h, static_cast<BatchIndex>(2), logits, queue);
    cpu_fixed_matrix<float, BatchIndex, TokenID> logits_cpu;
    logits.copy_to_cpu(queue, logits_cpu);

    for (size_t batch = 0; batch < 2; ++batch)
    {
        cpu_fixed_vector<float, EmbeddingDimension> one_h_cpu;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            one_h_cpu.push_back(h_cpu[static_cast<BatchIndex>(batch), d]);
        fixed_size_vector<float, EmbeddingDimension> one_h;
        one_h.copy_from_cpu(queue, one_h_cpu);
        fixed_size_vector<float, TokenID> one_logits;
        cpu_fixed_vector<float, TokenID> one_logits_cpu;
        layer.forward_from_hidden(one_h, one_logits, one_logits_cpu, queue);
        for (const auto token : enum_iterator1D<TokenID>())
            EXPECT_NEAR((logits_cpu[static_cast<BatchIndex>(batch), token]), one_logits_cpu[token], 1e-3f);
    }
}

TEST(OutputLayerBatchTest, BatchedDeltaAndLossStayOnDevice)
{
    using namespace rllm;
    auto& queue = vulkan_runtime::get_queue(0);
    OutputLayer layer;
    BatchedOutputWorkspace workspace;

    cpu_fixed_matrix<float, BatchIndex, TokenID> logits_cpu;
    logits_cpu.zero();
    workspace.logits.copy_from_cpu(queue, logits_cpu);

    cpu_fixed_vector<int, BatchIndex> expected;
    expected.push_back(1);
    expected.push_back(0);
    workspace.expected_tokens.copy_from_cpu(queue, expected);
    cpu_fixed_vector<int, BatchIndex> active;
    active.push_back(1);
    active.push_back(0);
    workspace.active_examples.copy_from_cpu(queue, active);

    constexpr float loss_gradient_scale = 0.25f;
    layer.compute_batched_delta(
        workspace.logits, static_cast<BatchIndex>(2), workspace, queue, loss_gradient_scale);

    cpu_fixed_matrix<float, BatchIndex, TokenID> delta_cpu;
    workspace.delta.copy_to_cpu(queue, delta_cpu);
    cpu_fixed_vector<float, BatchIndex> losses_cpu;
    losses_cpu.set_size(static_cast<BatchIndex>(2));
    workspace.losses.copy_to_cpu(queue, losses_cpu);

    const float uniform_probability = 1.0f / static_cast<float>(TokenID::MAX);
    EXPECT_NEAR(losses_cpu[BatchIndex::START], std::log(static_cast<float>(TokenID::MAX)), 1e-4f);
    for (const auto token : enum_iterator1D<TokenID>())
    {
        float expected_delta = OutputLayer::smooth - uniform_probability;
        if (static_cast<int>(token) == 1)
            expected_delta += 1.0f - OutputLayer::LABEL_SMOOTHING;
        EXPECT_NEAR((delta_cpu[BatchIndex::START, token]), expected_delta * loss_gradient_scale, 1e-5f);
        EXPECT_FLOAT_EQ((delta_cpu[static_cast<BatchIndex>(1), token]), 0.0f);
    }
}


namespace
{
    std::vector<rllm::TokenID> first_n_tokens(size_t count)
    {
        std::vector<rllm::TokenID> tokens;
        for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
        {
            tokens.push_back(tok);
            if (tokens.size() == count)
                break;
        }
        return tokens;
    }

    std::vector<rllm::EmbeddingDimension> first_n_dimensions(size_t count)
    {
        std::vector<rllm::EmbeddingDimension> dims;
        for (const auto dim : rllm::enum_iterator1D<rllm::EmbeddingDimension>())
        {
            dims.push_back(dim);
            if (dims.size() == count)
                break;
        }
        return dims;
    }

    nlohmann::json zero_output_layer_weights_json()
    {
        const size_t vocab = static_cast<size_t>(rllm::TokenID::MAX);
        const size_t dim = static_cast<size_t>(rllm::EmbeddingDimension::MAX);
        return nlohmann::json{{"W_lm_head", nlohmann::json::array_t(vocab * dim, 0.0f)}};
    }

    nlohmann::json& weight_at(nlohmann::json& j, rllm::TokenID tok, rllm::EmbeddingDimension dim)
    {
        const size_t width = static_cast<size_t>(rllm::EmbeddingDimension::MAX);
        const size_t index = static_cast<size_t>(tok) * width + static_cast<size_t>(dim);
        return j.at("W_lm_head").get_ref<nlohmann::json::array_t&>()[index];
    }

    std::vector<float> logits_from_output_layer(rllm::OutputLayer& layer)
    {
        const auto top = layer.get_top_k_by_logit(static_cast<size_t>(rllm::TokenID::MAX));
        std::vector<float> logits(static_cast<size_t>(rllm::TokenID::MAX), 0.0f);
        for (const auto& entry : top)
            logits[static_cast<size_t>(entry.token_id)] = entry.activation;
        return logits;
    }

    VulkanQueue& test_queue()
    {
        return rllm::vulkan_runtime::get_queue(0);
    }

    class ScopedNanFindingMode
    {
      public:
        explicit ScopedNanFindingMode(bool enabled)
            : m_previous(rllm::nan_finding_mode_enabled())
        {
            rllm::set_nan_finding_mode_enabled(enabled);
        }

        ~ScopedNanFindingMode()
        {
            rllm::set_nan_finding_mode_enabled(m_previous);
        }

        ScopedNanFindingMode(const ScopedNanFindingMode&) = delete;
        ScopedNanFindingMode& operator=(const ScopedNanFindingMode&) = delete;

      private:
        bool m_previous;
    };

    float reference_compute_score(
        const std::vector<float>& logits,
        std::vector<float>& deltas,
        rllm::TokenID expected_output_token
    )
    {
        const size_t vocab = logits.size();
        const size_t expected_index = static_cast<size_t>(expected_output_token);
        const float max_val = *std::max_element(logits.begin(), logits.end());

        std::vector<float> exp_values(vocab, 0.0f);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab; ++i)
        {
            exp_values[i] = std::exp(logits[i] - max_val);
            sum_exp += exp_values[i];
        }

        deltas.resize(vocab);
        for (size_t i = 0; i < vocab; ++i)
            deltas[i] = rllm::OutputLayer::smooth - exp_values[i] / sum_exp;

        deltas[expected_index] += (1.0f - rllm::OutputLayer::LABEL_SMOOTHING);
        return -(logits[expected_index] - max_val - std::log(sum_exp));
    }
} // namespace

TEST(OutputLayerForwardFromHiddenTest, PublicForwardMatchesImplementationHelper)
{
    auto weights_json = zero_output_layer_weights_json();
    rllm::cpu_fixed_matrix<float16, rllm::TokenID, rllm::EmbeddingDimension> cpu_weights;
    cpu_weights.zero();

    const auto tokens = first_n_tokens(4);
    const auto dims = first_n_dimensions(5);
    ASSERT_EQ(tokens.size(), 4u);
    ASSERT_EQ(dims.size(), 5u);

    const float weight_values[4][5] = {
        {0.5f, -0.25f, 1.0f, 0.0f, -0.5f},
        {-1.0f, 0.75f, 0.25f, -0.125f, 0.5f},
        {1.5f, 0.0f, -0.5f, 0.25f, -0.75f},
        {-0.25f, -0.5f, 0.75f, 1.0f, 0.125f},
    };

    for (size_t token_i = 0; token_i < tokens.size(); ++token_i)
    {
        for (size_t dim_i = 0; dim_i < dims.size(); ++dim_i)
        {
            const float value = weight_values[token_i][dim_i];
            weight_at(weights_json, tokens[token_i], dims[dim_i]) = value;
            cpu_weights[tokens[token_i], dims[dim_i]] = static_cast<float16>(value);
        }
    }
    rllm::fixed_size_matrix<float16, rllm::TokenID, rllm::EmbeddingDimension> weights;
    weights.copy_from_cpu(test_queue(), cpu_weights);

    rllm::cpu_fixed_vector<float, rllm::EmbeddingDimension> h_last_cpu;
    h_last_cpu.set_size(rllm::EmbeddingDimension::MAX);
    h_last_cpu.zero();
    h_last_cpu[dims[0]] = 1.0f;
    h_last_cpu[dims[1]] = -2.0f;
    h_last_cpu[dims[2]] = 0.5f;
    h_last_cpu[dims[3]] = 4.0f;
    h_last_cpu[dims[4]] = -1.5f;
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> h_last;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);

    rllm::fixed_size_vector<float, rllm::TokenID> impl_logits;
    impl_logits.set_size(rllm::TokenID::MAX);
    impl_logits.zero(test_queue());
    rllm::output_layer_forward_from_hidden_impl(test_queue(), h_last, weights, impl_logits);
    rllm::cpu_fixed_vector<float, rllm::TokenID> cpu_impl_logits;
    impl_logits.copy_to_cpu(test_queue(), cpu_impl_logits);

    rllm::OutputLayer layer;
    layer.load(weights_json);
    layer.forward_from_hidden(h_last, test_queue());

    const auto public_logits = logits_from_output_layer(layer);
    for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
    {
        const size_t index = static_cast<size_t>(tok);
        EXPECT_NEAR(public_logits[index], cpu_impl_logits[tok], 1e-5f) << "token index " << index;
    }
}

TEST(OutputLayerScoreTest, ZeroLogitsMatchReference)
{
    rllm::OutputLayer layer;
    layer.load(zero_output_layer_weights_json());

    rllm::cpu_fixed_vector<float, rllm::EmbeddingDimension> h_last_cpu;
    h_last_cpu.set_size(rllm::EmbeddingDimension::MAX);
    h_last_cpu.zero();
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> h_last;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);

    layer.forward_from_hidden(h_last, test_queue());

    rllm::Score score;
    const auto expected_token = first_n_tokens(1).front();
    const float loss = layer.compute_score(score, expected_token);

    std::vector<float> expected_deltas;
    const std::vector<float> logits(static_cast<size_t>(rllm::TokenID::MAX), 0.0f);
    const float expected_loss = reference_compute_score(logits, expected_deltas, expected_token);

    EXPECT_NEAR(loss, expected_loss, 1e-5f);
    EXPECT_NEAR(score.temp_values_cpu[rllm::TempStorage::START], 0.0f, 1e-6f);
    EXPECT_NEAR(
        score.temp_values_cpu[rllm::TempStorage::ONE],
        static_cast<float>(static_cast<int>(rllm::TokenID::MAX)),
        1e-4f
    );

    rllm::cpu_fixed_vector<float, rllm::TokenID> cpu_values;
    score.values.copy_to_cpu(test_queue(), cpu_values);
    for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
        EXPECT_NEAR(cpu_values[tok], expected_deltas[static_cast<size_t>(tok)], 1e-5f);
}

TEST(OutputLayerScoreTest, NonUniformLogitsMatchReference)
{
    auto weights = zero_output_layer_weights_json();
    const auto tokens = first_n_tokens(3);
    ASSERT_EQ(tokens.size(), 3u);

    weight_at(weights, tokens[0], rllm::EmbeddingDimension::START) = -1.0f;
    weight_at(weights, tokens[1], rllm::EmbeddingDimension::START) = 2.0f;
    weight_at(weights, tokens[2], rllm::EmbeddingDimension::START) = 0.5f;

    rllm::OutputLayer layer;
    layer.load(weights);

    rllm::cpu_fixed_vector<float, rllm::EmbeddingDimension> h_last_cpu;
    h_last_cpu.set_size(rllm::EmbeddingDimension::MAX);
    h_last_cpu.zero();
    h_last_cpu[rllm::EmbeddingDimension::START] = 1.0f;
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> h_last;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);

    layer.forward_from_hidden(h_last, test_queue());

    rllm::Score score;
    const auto expected_token = tokens[1];
    const float loss = layer.compute_score(score, expected_token);

    std::vector<float> expected_deltas;
    const auto logits = logits_from_output_layer(layer);
    EXPECT_NEAR(logits[static_cast<size_t>(tokens[0])], -1.0f, 1e-5f);
    EXPECT_NEAR(logits[static_cast<size_t>(tokens[1])], 2.0f, 1e-5f);
    EXPECT_NEAR(logits[static_cast<size_t>(tokens[2])], 0.5f, 1e-5f);
    const float expected_loss = reference_compute_score(logits, expected_deltas, expected_token);

    EXPECT_NEAR(loss, expected_loss, 1e-5f);
    EXPECT_NEAR(score.temp_values_cpu[rllm::TempStorage::START], 2.0f, 1e-5f);

    rllm::cpu_fixed_vector<float, rllm::TokenID> cpu_values;
    score.values.copy_to_cpu(test_queue(), cpu_values);
    for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
        EXPECT_NEAR(cpu_values[tok], expected_deltas[static_cast<size_t>(tok)], 1e-5f);
}

TEST(OutputLayerScoreTest, AllNegativeLogitsMatchReference)
{
    const ScopedNanFindingMode disable_nan_finding_mode(false);
    auto weights = zero_output_layer_weights_json();
    const auto tokens = first_n_tokens(3);
    ASSERT_EQ(tokens.size(), 3u);

    weight_at(weights, tokens[0], rllm::EmbeddingDimension::START) = -3.0f;
    weight_at(weights, tokens[1], rllm::EmbeddingDimension::START) = -1.5f;
    weight_at(weights, tokens[2], rllm::EmbeddingDimension::START) = -2.25f;

    rllm::OutputLayer layer;
    layer.load(weights);

    rllm::cpu_fixed_vector<float, rllm::EmbeddingDimension> h_last_cpu;
    h_last_cpu.set_size(rllm::EmbeddingDimension::MAX);
    h_last_cpu.zero();
    h_last_cpu[rllm::EmbeddingDimension::START] = 1.0f;
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> h_last;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);

    layer.forward_from_hidden(h_last, test_queue());

    rllm::Score score;
    const auto expected_token = tokens[1];
    const float loss = layer.compute_score(score, expected_token);

    std::vector<float> expected_deltas;
    const auto logits = logits_from_output_layer(layer);
    EXPECT_NEAR(logits[static_cast<size_t>(tokens[0])], -3.0f, 1e-5f);
    EXPECT_NEAR(logits[static_cast<size_t>(tokens[1])], -1.5f, 1e-5f);
    EXPECT_NEAR(logits[static_cast<size_t>(tokens[2])], -2.25f, 1e-5f);
    const float expected_loss = reference_compute_score(logits, expected_deltas, expected_token);

    EXPECT_NEAR(loss, expected_loss, 1e-5f);

    rllm::cpu_fixed_vector<float, rllm::TokenID> cpu_values;
    score.values.copy_to_cpu(test_queue(), cpu_values);
    for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
        EXPECT_NEAR(cpu_values[tok], expected_deltas[static_cast<size_t>(tok)], 1e-5f);
}

TEST(OutputLayerScoreTest, ReusedScoreMatchesReferenceAcrossCalls)
{
    auto weights = zero_output_layer_weights_json();
    const auto tokens = first_n_tokens(3);
    ASSERT_EQ(tokens.size(), 3u);

    weight_at(weights, tokens[0], rllm::EmbeddingDimension::START) = -1.0f;
    weight_at(weights, tokens[1], rllm::EmbeddingDimension::START) = 2.0f;
    weight_at(weights, tokens[2], rllm::EmbeddingDimension::START) = 0.5f;

    rllm::OutputLayer layer;
    layer.load(weights);

    rllm::cpu_fixed_vector<float, rllm::EmbeddingDimension> h_last_cpu;
    h_last_cpu.set_size(rllm::EmbeddingDimension::MAX);
    h_last_cpu.zero();

    rllm::Score score;

    h_last_cpu[rllm::EmbeddingDimension::START] = 1.0f;
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> h_last;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);
    layer.forward_from_hidden(h_last, test_queue());

    std::vector<float> expected_deltas_first;
    const auto logits_first = logits_from_output_layer(layer);
    const float expected_loss_first = reference_compute_score(logits_first, expected_deltas_first, tokens[1]);
    const float loss_first = layer.compute_score(score, tokens[1]);

    EXPECT_NEAR(loss_first, expected_loss_first, 1e-5f);
    {
        rllm::cpu_fixed_vector<float, rllm::TokenID> cpu_values;
        score.values.copy_to_cpu(test_queue(), cpu_values);
        for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
            EXPECT_NEAR(cpu_values[tok], expected_deltas_first[static_cast<size_t>(tok)], 1e-5f);
    }

    h_last_cpu.zero();
    h_last_cpu[rllm::EmbeddingDimension::START] = -1.0f;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);
    layer.forward_from_hidden(h_last, test_queue());

    std::vector<float> expected_deltas_second;
    const auto logits_second = logits_from_output_layer(layer);
    const float expected_loss_second = reference_compute_score(logits_second, expected_deltas_second, tokens[0]);
    const float loss_second = layer.compute_score(score, tokens[0]);

    EXPECT_NEAR(loss_second, expected_loss_second, 1e-5f);
    {
        rllm::cpu_fixed_vector<float, rllm::TokenID> cpu_values;
        score.values.copy_to_cpu(test_queue(), cpu_values);
        for (const auto tok : rllm::enum_iterator1D<rllm::TokenID>())
            EXPECT_NEAR(cpu_values[tok], expected_deltas_second[static_cast<size_t>(tok)], 1e-5f);
    }
}

TEST(OutputLayerScoreTest, RepeatedUpdatesReduceLoss)
{
    rllm::OutputLayer layer;
    layer.load(zero_output_layer_weights_json());

    rllm::cpu_fixed_vector<float, rllm::EmbeddingDimension> h_last_cpu;
    h_last_cpu.set_size(rllm::EmbeddingDimension::MAX);
    h_last_cpu.zero();
    h_last_cpu[rllm::EmbeddingDimension::START] = 1.0f;
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> h_last;
    h_last.copy_from_cpu(test_queue(), h_last_cpu);

    const auto expected_token = first_n_tokens(1).front();
    rllm::Score score;
    rllm::fixed_size_vector<float, rllm::EmbeddingDimension> dh_last;
    dh_last.set_size(rllm::EmbeddingDimension::MAX);
    rllm::OutputLayerGradientAccumulator accumulator;

    layer.forward_from_hidden(h_last, test_queue());
    const float initial_loss = layer.compute_score(score, expected_token);

    for (int step = 0; step < 8; ++step)
    {
        layer.forward_from_hidden(h_last, test_queue());
        const float loss = layer.compute_score(score, expected_token);
        (void)loss;
        dh_last.zero(test_queue());
        accumulator.reset(test_queue());
        layer.backward_accumulate(score.values, h_last, dh_last, accumulator);
        layer.apply_accumulated_update(accumulator, 0.0003f, 0.1f, 0.001f);
    }

    layer.forward_from_hidden(h_last, test_queue());
    const float final_loss = layer.compute_score(score, expected_token);

    EXPECT_LT(final_loss, initial_loss);
}

TEST(OutputLayerScoreTest, StringIndexLossIsGatedByCorrectValueTokenType)
{
    using namespace rllm;
    auto& queue = test_queue();
    for (const auto type : {TokenID::IDENTIFIER, TokenID::STRING, TokenID::INTEGER})
    {
        OutputLayer layer;
        auto weights = zero_output_layer_weights_json();
        weights["W_lm_head"][static_cast<size_t>(type) * static_cast<size_t>(EmbeddingDimension::MAX)] = 2.f;
        layer.load(weights);
        layer.set_string_table_size(2);
        cpu_fixed_vector<float, EmbeddingDimension> cpu_hidden;
        cpu_hidden.set_size(EmbeddingDimension::MAX);
        cpu_hidden.zero();
        cpu_hidden[EmbeddingDimension::START] = 1.f;
        fixed_size_vector<float, EmbeddingDimension> hidden;
        hidden.copy_from_cpu(queue, cpu_hidden);
        layer.forward_from_hidden(hidden, queue);
        Score score;
        const float type_loss = layer.compute_score(score, type);
        const float joint_loss = layer.compute_score(score, type, 1, 2);
        EXPECT_GT(joint_loss, type_loss);
        const auto other_type = type == TokenID::INTEGER ? TokenID::STRING : TokenID::INTEGER;
        const float wrong_type_loss = layer.compute_score(score, other_type);
        EXPECT_FLOAT_EQ(layer.compute_score(score, other_type, 1, 2), wrong_type_loss);
        EXPECT_FLOAT_EQ(layer.compute_score(score, type, NO_STRING_INDEX, 2), type_loss);

        // Scalar and batched paths must use the same loss and index gradient.
        const float scalar_loss = layer.compute_score(score, type, 1, 2);
        OutputLayerGradientAccumulator scalar_gradient;
        scalar_gradient.reset(queue);
        fixed_size_vector<float, EmbeddingDimension> scalar_dh;
        scalar_dh.set_size(EmbeddingDimension::MAX);
        scalar_dh.zero(queue);
        layer.backward_accumulate(score.values, hidden, scalar_dh, scalar_gradient);
        ASSERT_FALSE(scalar_gradient.string_gradient.empty());
        cpu_fixed_matrix<float, BatchIndex, EmbeddingDimension> batch_hidden;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            batch_hidden.set(BatchIndex::START, d, cpu_hidden[d]);
        BatchedOutputWorkspace workspace;
        workspace.h_last.copy_from_cpu(queue, batch_hidden);
        cpu_fixed_vector<int, BatchIndex> targets, active;
        targets.push_back(static_cast<int>(type)); active.push_back(1);
        workspace.expected_tokens.copy_from_cpu(queue, targets);
        workspace.active_examples.copy_from_cpu(queue, active);
        workspace.expected_strings = {1}; workspace.string_table_sizes = {2};
        const auto count = static_cast<BatchIndex>(1);
        layer.forward_batched(workspace.h_last, count, workspace.logits, queue);
        layer.compute_batched_delta(workspace.logits, count, workspace, queue);
        cpu_fixed_vector<float, BatchIndex> losses;
        workspace.losses.copy_to_cpu(queue, losses);
        EXPECT_NEAR(losses[BatchIndex::START], scalar_loss, 1e-5f);
        OutputLayerGradientAccumulator batch_gradient;
        batch_gradient.reset(queue);
        workspace.dh_last.zero(queue);
        layer.backward_batched_accumulate(workspace.delta, workspace.h_last, count, workspace.dh_last, batch_gradient);
        ASSERT_EQ(batch_gradient.string_gradient.size(), scalar_gradient.string_gradient.size());
        for (size_t i = 0; i < batch_gradient.string_gradient.size(); ++i)
            EXPECT_NEAR(batch_gradient.string_gradient[i], scalar_gradient.string_gradient[i], 1e-6f);
        cpu_fixed_vector<float, EmbeddingDimension> scalar_cpu_dh;
        scalar_dh.copy_to_cpu(queue, scalar_cpu_dh);
        StringIndexHead reference_head{static_cast<size_t>(EmbeddingDimension::MAX)};
        reference_head.reserve_entries(2);
        std::vector<float> reference_hidden(static_cast<size_t>(EmbeddingDimension::MAX));
        reference_hidden[0] = 1.f;
        std::vector<float> reference_delta, reference_gradient, reference_dh(reference_hidden.size());
        reference_head.loss(reference_hidden, 2, 1, reference_delta);
        reference_head.backward(reference_hidden, reference_delta, reference_dh, reference_gradient);
        // Token weights are zero outside dimension zero; the shared trainer
        // must receive the negative mathematical derivative from the value head.
        EXPECT_NEAR(scalar_cpu_dh[static_cast<EmbeddingDimension>(1)], -reference_dh[1], 1e-6f);
        cpu_fixed_matrix<float, BatchIndex, EmbeddingDimension> batch_cpu_dh;
        workspace.dh_last.copy_to_cpu(queue, batch_cpu_dh);
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
            EXPECT_NEAR(scalar_cpu_dh[d], batch_cpu_dh.get(BatchIndex::START, d), 1e-5f);
        EXPECT_NE(layer.get_top_k_by_logit(1).front().string_index, NO_STRING_INDEX);
    }
}
