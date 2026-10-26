#include <gtest/gtest.h>

#include <backprop/Muon.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <cpu/cpu_fixed_vector.hpp>
#include <rllm_vulkan_runtime.hpp>

#include <array>
#include <cmath>

namespace
{
    template <typename Rows, typename Cols>
    void expect_muon_update_matches_reference(float dimension_scale)
    {
        using namespace rllm;

        constexpr size_t rows = static_cast<size_t>(Rows::MAX);
        constexpr size_t cols = static_cast<size_t>(Cols::MAX);
        constexpr float learning_rate = 0.01f;
        constexpr float clip_scale = 0.4f;
        constexpr std::array<float, 3> gradients{0.8f, -0.35f, 0.12f};
        constexpr std::array<float, 3> initial_momentum{0.2f, -0.1f, 0.05f};

        cpu_fixed_matrix<float16, Rows, Cols> host_weight;
        cpu_fixed_matrix<float, Rows, Cols> host_momentum;
        cpu_fixed_matrix<float, Rows, Cols> host_gradient;
        for (size_t i = 0; i < rows * cols; ++i)
        {
            host_weight.data()[i] = static_cast<float16>(0.25f);
            host_momentum.data()[i] = 0.0f;
            host_gradient.data()[i] = 0.0f;
        }

        std::array<float, gradients.size()> expected_momentum{};
        std::array<float, gradients.size()> expected_update{};
        float norm_squared = 0.0f;
        for (size_t i = 0; i < gradients.size(); ++i)
        {
            const float grad = gradients[i] * clip_scale;
            expected_momentum[i] = 0.95f * initial_momentum[i] + 0.05f * grad;
            expected_update[i] = 0.05f * grad + 0.95f * expected_momentum[i];
            norm_squared += expected_update[i] * expected_update[i];
            host_momentum[static_cast<Rows>(i), static_cast<Cols>(i)] = initial_momentum[i];
            host_gradient[static_cast<Rows>(i), static_cast<Cols>(i)] = gradients[i];
        }

        const float norm = std::sqrt(norm_squared);
        for (auto& value : expected_update)
            value /= norm;
        for (int step = 0; step < 5; ++step)
        {
            for (auto& value : expected_update)
            {
                const float gram = value * value;
                const float polynomial = -4.7750f * gram + 2.0315f * gram * gram;
                value = 3.4445f * value + polynomial * value;
            }
        }

        auto& queue = vulkan_runtime::get_queue(0);
        fixed_size_matrix<float16, Rows, Cols> weight;
        fixed_size_matrix<float, Rows, Cols> momentum;
        fixed_size_matrix<float, Rows, Cols> gradient;
        weight.copy_from_cpu(queue, host_weight);
        momentum.copy_from_cpu(queue, host_momentum);
        gradient.copy_from_cpu(queue, host_gradient);

        cpu_fixed_vector<float, TempStorage> host_diagnostics;
        host_diagnostics.set_size(TempStorage::MAX);
        host_diagnostics[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE] = clip_scale;
        fixed_size_vector<float, TempStorage> diagnostics;
        diagnostics.copy_from_cpu(queue, host_diagnostics);

        backprop::MuonWorkspace workspace;
        backprop::muon_update(weight, momentum, gradient, workspace, learning_rate, diagnostics);

        cpu_fixed_matrix<float16, Rows, Cols> actual_weight;
        cpu_fixed_matrix<float, Rows, Cols> actual_momentum;
        cpu_fixed_matrix<float, Rows, Cols> actual_gradient;
        weight.copy_to_cpu(queue, actual_weight);
        momentum.copy_to_cpu(queue, actual_momentum);
        gradient.copy_to_cpu(queue, actual_gradient);
        queue.wait("Muon unit test readback");

        for (size_t i = 0; i < gradients.size(); ++i)
        {
            const auto row = static_cast<Rows>(i);
            const auto col = static_cast<Cols>(i);
            const float old_weight = static_cast<float>(host_weight[row, col]);
            const float expected_weight = static_cast<float>(static_cast<float16>(
                (1.0f - learning_rate * 0.01f) * old_weight -
                learning_rate * dimension_scale * expected_update[i]));
            const float actual_momentum_value = actual_momentum[row, col];
            const float actual_gradient_value = actual_gradient[row, col];
            const float actual_weight_value = static_cast<float>(actual_weight[row, col]);
            EXPECT_NEAR(actual_momentum_value, expected_momentum[i], 1e-5f);
            EXPECT_NEAR(actual_gradient_value, expected_update[i], 1e-3f);
            EXPECT_NEAR(actual_weight_value, expected_weight, 1e-3f);
        }

        const float expected_untouched_weight = static_cast<float>(static_cast<float16>(
            (1.0f - learning_rate * 0.01f) * 0.25f));
        const float untouched_momentum =
            actual_momentum[static_cast<Rows>(0), static_cast<Cols>(1)];
        const float untouched_gradient =
            actual_gradient[static_cast<Rows>(0), static_cast<Cols>(1)];
        EXPECT_FLOAT_EQ(untouched_momentum, 0.0f);
        EXPECT_FLOAT_EQ(untouched_gradient, 0.0f);
        EXPECT_NEAR(
            static_cast<float>(actual_weight[static_cast<Rows>(0), static_cast<Cols>(1)]),
            expected_untouched_weight,
            1e-6f);
    }
}

TEST(MuonTest, SquareProjectionUsesNesterovAndNewtonSchulzUpdate)
{
    expect_muon_update_matches_reference<rllm::EmbeddingDimension, rllm::EmbeddingDimension>(1.0f);
}

TEST(MuonTest, TallProjectionScalesUpdateBySquareRootOfAspectRatio)
{
    const float scale = std::sqrt(
        static_cast<float>(rllm::FFDimension::MAX) /
        static_cast<float>(rllm::EmbeddingDimension::MAX));
    expect_muon_update_matches_reference<rllm::FFDimension, rllm::EmbeddingDimension>(scale);
}

TEST(MuonTest, WideProjectionUsesRowGramMatrix)
{
    expect_muon_update_matches_reference<rllm::EmbeddingDimension, rllm::FFDimension>(1.0f);
}
