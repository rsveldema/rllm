#include <DynamicSkipAttention.hpp>

#include <cmath>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>
#include <rllm_hip_runtime.hpp>

namespace rllm
{
    DynamicSkipAttention::DynamicSkipAttention()
    {
        initialize();
    }

    void DynamicSkipAttention::initialize()
    {
        cpu_fixed_matrix<float, TokenID, HeadsIndex> values;
        for (const auto token : enum_iterator1D<TokenID>())
            for (const auto layer : enum_iterator1D<HeadsIndex>())
                values[token, layer] = INITIAL_LOGIT_STEP * static_cast<float>(layer);
        m_logits.copy_from_cpu(hip_runtime::get_queue(0), values);
        m_adam_first.zero(hip_runtime::get_queue(0));
        m_adam_second.zero(hip_runtime::get_queue(0));
    }

    void DynamicSkipAttention::begin_mix(
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& output,
        PositionIndex rows) const
    {
        output.set_rows(rows);
        output.zero(hip_runtime::get_queue(0));
    }

    static void add_dsa_source(
        // OFFLOAD_PARAMETERS(source, output, tokens, logits, rows, source_layer, destination_layer)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& output,
        const GpuInputLine& tokens,
        const fixed_size_matrix<float, TokenID, HeadsIndex>& logits,
        int rows,
        int source_layer,
        int destination_layer
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = hip_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(
            static_cast<PositionIndex>(rows));
        OFFLOAD_PARFOR_2D_PARAM(queue, row, dimension, grid, (source, output, tokens, logits, rows, source_layer, destination_layer))
        const TokenID token = tokens[row];
        float maximum = logits[token, static_cast<HeadsIndex>(0)];
        for (int index = 1; index < destination_layer; ++index)
        {
            const float candidate = logits[token, static_cast<HeadsIndex>(index)];
            if (candidate > maximum) maximum = candidate;
        }
        float denominator = 0.0f;
        for (int index = 0; index < destination_layer; ++index)
        {
            float shifted = logits[token, static_cast<HeadsIndex>(index)];
            shifted -= maximum;
            denominator += std::exp(shifted);
        }
        float shifted_source = logits[token, static_cast<HeadsIndex>(source_layer)];
        shifted_source -= maximum;
        float weight = std::exp(shifted_source);
        weight /= denominator;
        float contribution = source[row, dimension];
        contribution *= weight;
        output[row, dimension] += contribution;
        ENDFOR
    }

    void DynamicSkipAttention::add_source(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& output,
        const GpuInputLine& tokens, int rows, int source_layer,
        int destination_layer) const
    {
        add_dsa_source(source, output, tokens, m_logits, rows,
            source_layer, destination_layer);
    }

    static void backward_dsa_source(
        // OFFLOAD_PARAMETERS(mixed_gradient, source, mixed, source_gradient, tokens, logits, rows, source_layer, destination_layer, logit_gradient)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& mixed_gradient,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& mixed,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source_gradient,
        const GpuInputLine& tokens,
        const fixed_size_matrix<float, TokenID, HeadsIndex>& logits,
        int rows,
        int source_layer,
        int destination_layer,
        fixed_size_matrix<float, TokenID, HeadsIndex>& logit_gradient
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = hip_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(
            static_cast<PositionIndex>(rows));
        OFFLOAD_PARFOR_2D_PARAM(queue, row, dimension, grid, (mixed_gradient, source, mixed, source_gradient, tokens, logits, rows, source_layer, destination_layer, logit_gradient))
        const TokenID token = tokens[row];
        float maximum = logits[token, static_cast<HeadsIndex>(0)];
        for (int index = 1; index < destination_layer; ++index)
        {
            const float candidate = logits[token, static_cast<HeadsIndex>(index)];
            if (candidate > maximum) maximum = candidate;
        }
        float denominator = 0.0f;
        for (int index = 0; index < destination_layer; ++index)
        {
            float shifted = logits[token, static_cast<HeadsIndex>(index)];
            shifted -= maximum;
            denominator += std::exp(shifted);
        }
        const auto source_index = static_cast<HeadsIndex>(source_layer);
        float shifted_source = logits[token, source_index];
        shifted_source -= maximum;
        float weight = std::exp(shifted_source);
        weight /= denominator;
        const float upstream = mixed_gradient[row, dimension];
        float source_contribution = upstream;
        source_contribution *= weight;
        source_gradient[row, dimension] += source_contribution;
        float logit_contribution = source[row, dimension];
        logit_contribution -= mixed[row, dimension];
        logit_contribution *= upstream;
        logit_contribution *= weight;
        atomicAdd(logit_gradient[token, source_index], logit_contribution);
        ENDFOR
    }

    void DynamicSkipAttention::backward_source(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& mixed_gradient,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& mixed,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source_gradient,
        const GpuInputLine& tokens, int rows, int source_layer,
        int destination_layer,
        fixed_size_matrix<float, TokenID, HeadsIndex>& logit_gradient) const
    {
        backward_dsa_source(mixed_gradient, source, mixed, source_gradient,
            tokens, m_logits, rows, source_layer, destination_layer, logit_gradient);
    }

    static void update_dsa(
        // OFFLOAD_PARAMETERS(logits, first, second, gradient, learning_rate, bias_correction1, bias_correction2)
        fixed_size_matrix<float, TokenID, HeadsIndex>& logits,
        fixed_size_matrix<float, TokenID, HeadsIndex>& first,
        fixed_size_matrix<float, TokenID, HeadsIndex>& second,
        fixed_size_matrix<float, TokenID, HeadsIndex>& gradient,
        float learning_rate,
        float bias_correction1,
        float bias_correction2
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = hip_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, HeadsIndex>();
        OFFLOAD_PARFOR_2D_PARAM(queue, token, layer, grid, (logits, first, second, gradient, learning_rate, bias_correction1, bias_correction2))
        const float value = gradient[token, layer];
        float first_value = first[token, layer];
        first_value *= 0.9f;
        float first_delta = value;
        first_delta *= 0.1f;
        first_value += first_delta;
        first[token, layer] = first_value;
        float second_value = second[token, layer];
        second_value *= 0.95f;
        float second_delta = value;
        second_delta *= value;
        second_delta *= 0.05f;
        second_value += second_delta;
        second[token, layer] = second_value;
        float m_hat = first_value;
        m_hat /= bias_correction1;
        float v_hat = second_value;
        v_hat /= bias_correction2;
        float denominator = std::sqrt(v_hat);
        denominator += 1e-8f;
        float update = learning_rate;
        update *= m_hat;
        update /= denominator;
        logits[token, layer] -= update;
        ENDFOR
    }

    void DynamicSkipAttention::apply_update(
        fixed_size_matrix<float, TokenID, HeadsIndex>& gradient,
        float learning_rate, float bias_correction1, float bias_correction2)
    {
        update_dsa(m_logits, m_adam_first, m_adam_second, gradient,
            learning_rate, bias_correction1, bias_correction2);
    }
}
