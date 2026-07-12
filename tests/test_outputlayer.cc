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
        layer.apply_accumulated_update(accumulator, 0.003f);
    }

    layer.forward_from_hidden(h_last, test_queue());
    const float final_loss = layer.compute_score(score, expected_token);

    EXPECT_LT(final_loss, initial_loss);
}
