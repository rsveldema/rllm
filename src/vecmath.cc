#include <TransformerBlock.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>
#include <vecmath.hpp>


namespace parallel
{

#if defined(RLLM_ENABLE_STATISTICS)
    thread_local DispatchParams g_vulkan_dispatch_params{};

    void set_vulkan_dispatch_params(std::string_view site, std::string_view param)
    {
        g_vulkan_dispatch_params.site = site;
        g_vulkan_dispatch_params.parameter = param;
    }


    DispatchParams get_dispatch_params() { return g_vulkan_dispatch_params; }

    void clear_vulkan_dispatch_params()
    {
        g_vulkan_dispatch_params = DispatchParams{};
    }
#endif
}

namespace rllm
{
    void fill(
        // OFFLOAD_PARAMETERS(dst, value, length)
        fixed_size_vector<int, PositionIndex>& dst,
        int value,
        PositionIndex length
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(length), (dst, value))
        dst[static_cast<size_t>(i)] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value, length)
        fixed_size_vector<float, PositionIndex>& dst,
        float value,
        PositionIndex length
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<PositionIndex>(length), (dst, value))
        dst[static_cast<size_t>(i)] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dst,
        float value
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(dst.num_rows());
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (dst, value))
        dst[r, c] = value;
        ENDFOR
    }

    void fill(
        // OFFLOAD_PARAMETERS(dst, value)
        flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dst,
        float value
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, PositionIndex>(dst.num_rows(), dst.num_cols());
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (dst, value))
        dst[r, c] = value;
        ENDFOR
    }

    void copy_hidden_row_to_vector(
        // OFFLOAD_PARAMETERS(src, row, dst)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& src,
        PositionIndex row,
        fixed_size_vector<float, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, d, enum_iterator1D<EmbeddingDimension>(), (src, row, dst))
        dst[d] = src[row, d];
        ENDFOR
    }

    void adamw_update(
        // OFFLOAD_PARAMETERS(weight, first, second, gradient, learning_rate, bias_correction1, bias_correction2)
        fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& weight,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& first,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& second,
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient,
        float learning_rate, float bias_correction1, float bias_correction2
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (weight, first, second, gradient, learning_rate, bias_correction1, bias_correction2))
        const float g = math::clamp(gradient[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        first[r, c] = ((TransformerBlock::ADAM_BETA1 * first[r, c]) + ((1.0f - TransformerBlock::ADAM_BETA1) * g));
        second[r, c] = ((TransformerBlock::ADAM_BETA2 * second[r, c]) + ((1.0f - TransformerBlock::ADAM_BETA2) * (g * g)));
        const float update = ((first[r, c] / bias_correction1) / (sqrt((second[r, c] / bias_correction2)) + TransformerBlock::ADAM_EPSILON));
        const float decayed = (static_cast<float>(weight[r, c]) * (1.0f - (learning_rate * TransformerBlock::WEIGHT_DECAY)));
        weight[r, c] = math::clamp((decayed + (learning_rate * update)), -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void adamw_update(
        // OFFLOAD_PARAMETERS(weight, first, second, gradient, learning_rate, bias_correction1, bias_correction2)
        fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& weight,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& first,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& second,
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient,
        float learning_rate, float bias_correction1, float bias_correction2
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (weight, first, second, gradient, learning_rate, bias_correction1, bias_correction2))
        const float g = math::clamp(gradient[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        first[r, c] = ((TransformerBlock::ADAM_BETA1 * first[r, c]) + ((1.0f - TransformerBlock::ADAM_BETA1) * g));
        second[r, c] = ((TransformerBlock::ADAM_BETA2 * second[r, c]) + ((1.0f - TransformerBlock::ADAM_BETA2) * (g * g)));
        const float update = ((first[r, c] / bias_correction1) / (sqrt((second[r, c] / bias_correction2)) + TransformerBlock::ADAM_EPSILON));
        const float decayed = (static_cast<float>(weight[r, c]) * (1.0f - (learning_rate * TransformerBlock::WEIGHT_DECAY)));
        weight[r, c] = math::clamp((decayed + (learning_rate * update)), -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void adamw_update(
        // OFFLOAD_PARAMETERS(weight, first, second, gradient, learning_rate, bias_correction1, bias_correction2)
        fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& weight,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& first,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& second,
        const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient,
        float learning_rate, float bias_correction1, float bias_correction2
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (weight, first, second, gradient, learning_rate, bias_correction1, bias_correction2))
        const float g = math::clamp(gradient[r, c], -TransformerBlock::GRAD_CLIP, TransformerBlock::GRAD_CLIP);
        first[r, c] = ((TransformerBlock::ADAM_BETA1 * first[r, c]) + ((1.0f - TransformerBlock::ADAM_BETA1) * g));
        second[r, c] = ((TransformerBlock::ADAM_BETA2 * second[r, c]) + ((1.0f - TransformerBlock::ADAM_BETA2) * (g * g)));
        const float update = ((first[r, c] / bias_correction1) / (sqrt((second[r, c] / bias_correction2)) + TransformerBlock::ADAM_EPSILON));
        const float decayed = (static_cast<float>(weight[r, c]) * (1.0f - (learning_rate * TransformerBlock::WEIGHT_DECAY)));
        weight[r, c] = math::clamp((decayed + (learning_rate * update)), -TransformerBlock::WEIGHT_CLAMP, TransformerBlock::WEIGHT_CLAMP);
        ENDFOR
    }

    void matmul_ABt_3_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2,B3,C3)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C1,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C2,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B3,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C3
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B1, C1, B2, C2, B3, C3))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float a = A[i, k];
                const float term1 = (a * B1[j, k]);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (a * B2[j, k]);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = (a * B3[j, k]);
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C1[i, j] = static_cast<float>(sum1);
            C2[i, j] = static_cast<float>(sum2);
            C3[i, j] = static_cast<float>(sum3);
        ENDFOR
    }

    /**
     * @brief C1 = A × B1^T and C2 = A × B2^T, where A is [m × k] and B1, B2 are [n × k].
     */
    void matmul_ABt_2_matrix_muls(
        // OFFLOAD_PARAMETERS(A,B1,C1,B2,C2)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B1,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C1,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B2,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C2
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B1, C1, B2, C2))
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float a = A[i, k];
                const float term1 = (a * B1[j, k]);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (a * B2[j, k]);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;
            }
            C1[i, j] = static_cast<float>(sum1);
            C2[i, j] = static_cast<float>(sum2);
        ENDFOR
    }

    void matmul_ABt(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[j, k]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_ABt(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(FFDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[j, k]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& B,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] = math::clamp(static_cast<float>(sum), -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB_add(
        // OFFLOAD_PARAMETERS(A, B, C)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A,
        const fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& B,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(FFDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[i, k] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            const float raw = (C[i, j] + static_cast<float>(sum));
            C[i, j] = math::clamp(raw, -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AB_add_3_matrix_muls(
        // OFFLOAD_PARAMETERS(A1, B1, A2, B2, A3, B3, C)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A1,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B1,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A2,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A3,
        const fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& B3,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& C
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const PositionIndex m = A1.num_rows();
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(m);
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A1, B1, A2, B2, A3, B3, C))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
            {
                const int k = int(l_idx);

                const float term1 = (A1[i, k] * B1[k, j]);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (A2[i, k] * B2[k, j]);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = (A3[i, k] * B3[k, j]);
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            const float raw = (C[i, j] + static_cast<float>((sum1 + (sum2 + sum3))));
            C[i, j] = math::clamp(raw, -10000.0f, 10000.0f);
        ENDFOR
    }

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C, k_count))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[k, i] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<float>(sum);
        ENDFOR
    }

    void matmul_AtB_acc_2_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, B, C1, C2, k_count)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A1,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& A2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& C2,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A1, A2, B, C1, C2, k_count))
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float b = B[k, j];
                const float term1 = (A1[k, i] * b);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (A2[k, i] * b);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;
            }
            C1[i, j] += static_cast<float>(sum1);
            C2[i, j] += static_cast<float>(sum2);
        ENDFOR
    }

    void matmul_AtB_acc(
        // OFFLOAD_PARAMETERS(A, B, C, k_count)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C, k_count))
            float sum = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float term = (A[k, i] * B[k, j]);
                OVERFLOW_CHECK_ADD(sum, term);
                sum += term;
            }
            C[i, j] += static_cast<float>(sum);
        ENDFOR
    }

    void matmul_AtB_acc_3_matrix(
        // OFFLOAD_PARAMETERS(A1, A2, A3, B, C1, C2, C3, k_count)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A1,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A2,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& A3,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& B,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C1,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C2,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& C3,
        PositionIndex k_count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A1, A2, A3, B, C1, C2, C3, k_count))
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (size_t l_idx = 0; l_idx < static_cast<size_t>(k_count); ++l_idx)
            {
                const int k = int(l_idx);
                const float b = B[k, j];
                const float term1 = (A1[k, i] * b);
                OVERFLOW_CHECK_ADD(sum1, term1);
                sum1 += term1;

                const float term2 = (A2[k, i] * b);
                OVERFLOW_CHECK_ADD(sum2, term2);
                sum2 += term2;

                const float term3 = (A3[k, i] * b);
                OVERFLOW_CHECK_ADD(sum3, term3);
                sum3 += term3;
            }
            C1[i, j] += static_cast<float>(sum1);
            C2[i, j] += static_cast<float>(sum2);
            C3[i, j] += static_cast<float>(sum3);
        ENDFOR
    }

    void element_wise_sum(
        // OFFLOAD_PARAMETERS(lhs, rhs, dst)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& lhs,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& rhs,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dst
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(lhs.num_rows());
        OFFLOAD_PARFOR_2D_PARAM(queue, t, d, grid, (lhs, rhs, dst))
        dst[t, d] = math::clamp((lhs[t, d] + rhs[t, d]), -10000.0f, 10000.0f);
        ENDFOR
    }

    void swiglu_forward(
        // OFFLOAD_PARAMETERS(gate_pre, up_pre, ffn_act)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gate_pre,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& up_pre,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& ffn_act,
        PositionIndex seq_len
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(seq_len);
        OFFLOAD_PARFOR_2D_PARAM(queue, t, f, grid, (gate_pre, up_pre, ffn_act))
        const float g = gate_pre[t, f];
        const float silu = (g / (1.0f + std::exp(-g)));
        ffn_act[t, f] = math::clamp((silu * up_pre[t, f]), -10000.0f, 10000.0f);
        ENDFOR
    }
}
