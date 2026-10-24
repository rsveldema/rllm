#include <backprop/Backprop.hpp>
#include <TextTrainer.hpp>
#include <TextTrainerInternal.hpp>

#include <chrono>
#include <format>
#include <rllm_vulkan_runtime.hpp>
#include <vecmath.hpp>

namespace rllm::backprop
{
    static_assert(Backend::name[0] == 'b');
}

namespace rllm
{
    namespace
    {
        double backprop_elapsed_ms(const std::chrono::steady_clock::time_point& started_at)
        {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started_at).count();
        }
    }

    void TextTrainer::propagate_backward_mtp(
        const fixed_size_obj_vector<Score, MultiTokenPredictionIndex>& scores,
        MultiTokenPredictionIndex num_valid, TrainingStepTiming* timing)
    {
        assert(m_seq_len > PositionIndex::START);
        auto& queue = vulkan_runtime::get_queue(0);
        m_backward_workspace->reset(queue, m_seq_len);
        auto& ws = *m_backward_workspace;
        ws.dh.set_rows(m_seq_len);
        ws.din.set_rows(m_seq_len);
        const auto last_pos = dec(m_seq_len);
        copy_hidden_row_to_vector(m_forward_workspace->h, last_pos, ws.h_last_vec);

        ws.dh_last.zero(queue);
        const auto output_started_at = std::chrono::steady_clock::now();
        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>(num_valid))
        {
            ws.output_layer_delta.set_size(scores[oi].values.size());
            auto bytes = static_cast<VkDeviceSize>(
                static_cast<size_t>(scores[oi].values.size()) * sizeof(float));
            if (bytes > 0)
                ws.output_layer_delta.device_buffer().copy_from(
                    queue, scores[oi].values.device_buffer(), bytes);
            ws.output_string_table_index_delta.set_size(
                scores[oi].string_table_index_values.size());
            bytes = static_cast<VkDeviceSize>(
                static_cast<size_t>(scores[oi].string_table_index_values.size()) * sizeof(float));
            if (bytes > 0)
                ws.output_string_table_index_delta.device_buffer().copy_from(
                    queue, scores[oi].string_table_index_values.device_buffer(), bytes);
            const auto delta_phase = std::format(
                "output head {} backward delta", static_cast<size_t>(oi));
            check_token_vector_nan_finding_mode(queue, ws.output_layer_delta,
                10.0f, "output delta", delta_phase.c_str());
            m_output_layers[oi].backward_accumulate(
                ws.output_layer_delta, ws.output_string_table_index_delta,
                ws.h_last_vec, ws.dh_last,
                m_gradient_accumulation_workspace->output_layers[oi]);
            const auto phase = std::format(
                "after output head {} backward", static_cast<size_t>(oi));
            check_hidden_vector_nan_finding_mode(queue, ws.dh_last, phase.c_str());
        }
        if (timing)
            timing->backward_output_ms += backprop_elapsed_ms(output_started_at);

        ws.dh.zero(queue);
        ws.din.zero(queue);
        scatter_dh_last_to_row(ws.dh_last, ws.dh, last_pos);
        check_hidden_nan_finding_mode(queue, ws.dh, m_seq_len, "after dh_last scatter");
        auto* p_dh = &ws.dh;
        auto* p_din = &ws.din;
        const auto transformer_started_at = std::chrono::steady_clock::now();
        for (size_t reverse_execution = m_transformer_execution_order.size();
             reverse_execution-- > 0;)
        {
            const size_t i = m_transformer_execution_order[reverse_execution];
            const auto before_phase = std::format(
                "before transformer block {} backward", i);
            check_hidden_nan_finding_mode(
                queue, *p_dh, m_seq_len, before_phase.c_str());
            m_transformer_blocks[i].backward(
                *p_dh, *p_din, ws.transformer_block,
                m_forward_workspace->transformer_workspaces[reverse_execution]);
            if (i >= m_frozen_transformer_block_count)
                m_transformer_blocks[i].accumulate_gradients(
                    ws.transformer_block,
                    m_gradient_accumulation_workspace->transformer_blocks[i]);
            const auto after_phase = std::format(
                "after transformer block {} backward", i);
            check_hidden_nan_finding_mode(
                queue, *p_din, m_seq_len, after_phase.c_str());
            if (const auto iteration = loop_iteration_for_execution(reverse_execution))
                backward_loop_transition(
                    *p_din, ws.loop_input_gradient, *iteration == 0);
            std::swap(p_dh, p_din);
        }
        if (timing)
            timing->backward_transformer_ms += backprop_elapsed_ms(transformer_started_at);

        const auto input_started_at = std::chrono::steady_clock::now();
        check_hidden_nan_finding_mode(
            queue, *p_dh, m_seq_len, "before input layer backward");
        m_input_layer.accumulate_backward(
            m_last_input, *p_dh, m_gradient_accumulation_workspace->embeddings);
        if (timing)
            timing->backward_input_ms += backprop_elapsed_ms(input_started_at);
    }

    void TextTrainer::train_batch_transformer_backward(
        PositionIndex packed_rows, BatchTrainingTiming& timing)
    {
        auto& back = *m_backward_workspace;
        auto& forward = *m_forward_workspace;
        auto* dh = &back.dh;
        auto* din = &back.din;
        const auto started = std::chrono::steady_clock::now();
        const bool collect_diagnostics = m_backward_diagnostics_pending;
        if (collect_diagnostics)
            log_hidden_gradient_diagnostics(*dh, packed_rows,
                std::format("output heads -> transformer layer {}",
                    m_transformer_blocks.size() - 1));
        for (size_t reverse_execution = m_transformer_execution_order.size();
             reverse_execution-- > 0;)
        {
            const size_t index = m_transformer_execution_order[reverse_execution];
            m_transformer_blocks[index].backward(
                *dh, *din, back.transformer_block,
                forward.transformer_workspaces[reverse_execution], true,
                collect_diagnostics, static_cast<int>(index));
            if (index >= m_frozen_transformer_block_count)
                m_transformer_blocks[index].accumulate_gradients(
                    back.transformer_block,
                    m_gradient_accumulation_workspace->transformer_blocks[index]);
            if (collect_diagnostics)
                log_hidden_gradient_diagnostics(*din, packed_rows,
                    reverse_execution == 0
                        ? "transformer execution 0 -> input embeddings"
                        : std::format("transformer execution {} -> {}",
                            reverse_execution, reverse_execution - 1));
            if (const auto iteration = loop_iteration_for_execution(reverse_execution))
            {
                if (*iteration > 0)
                    element_wise_sum(*din,
                        forward.transformer_workspaces[reverse_execution].auxiliary_gradient,
                        *din);
                backward_loop_transition(
                    *din, back.loop_input_gradient, *iteration == 0);
            }
            std::swap(dh, din);
        }
        m_backward_diagnostics_pending = false;
        timing.backward_transformer_ms += backprop_elapsed_ms(started);
        (void)packed_rows;
    }
}
