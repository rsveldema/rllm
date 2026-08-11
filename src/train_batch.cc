#include <TextTrainer.hpp>
#include <OptimizerDiagnostics.hpp>
#include "TextTrainerInternal.hpp"
#include <LogFormatting.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <enum_iterator2D.hpp>
#include <rllm_vulkan_runtime.hpp>

namespace rllm
{
    namespace
    {
        double elapsed_ms(const std::chrono::steady_clock::time_point& started_at)
        {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
        }

        int batch_input_len(int sequence_length)
        {
            return std::max(1, sequence_length - static_cast<int>(MultiTokenPredictionIndex::MAX));
        }

        TokenID target_for_head(const CpuInputLine& line, int input_length, MultiTokenPredictionIndex head)
        {
            return line[static_cast<PositionIndex>(input_length + static_cast<int>(head))];
        }

        void gather_indexed_hidden(
            // OFFLOAD_PARAMETERS(h, rows, gathered, count)
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
            const fixed_size_vector<int, BatchIndex>& rows,
            fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& gathered,
            int count
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(count));
            OFFLOAD_PARFOR_2D_PARAM(queue, item, d, grid, (h, rows, gathered, count))
            gathered[item, d] = h[rows[item], d];
            ENDFOR
        }

        void add_indexed_hidden_gradient(
            // OFFLOAD_PARAMETERS(gathered, rows, dh, count)
            const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& gathered,
            const fixed_size_vector<int, BatchIndex>& rows,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            int count
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(count));
            OFFLOAD_PARFOR_2D_PARAM(queue, item, d, grid, (gathered, rows, dh, count))
            dh[rows[item], d] += gathered[item, d];
            ENDFOR
        }
    }

    void TextTrainer::collect_active_batch_inputs(
        const std::vector<BatchTrainingItem>& batch, std::vector<size_t>& indices,
        std::vector<CpuInputLine>& contexts) const
    {
        for (size_t i = 0; i < batch.size(); ++i)
        {
            if (batch[i].finished)
                continue;
            const int length = static_cast<int>(batch[i].line.size());
            const int input_length = batch[i].context_length.has_value()
                ? static_cast<int>(*batch[i].context_length)
                : batch_input_len(length);
            CpuInputLine context;
            batch[i].line.sub_array(context, static_cast<PositionIndex>(input_length));
            indices.push_back(i);
            contexts.push_back(context);
        }
    }

    void TextTrainer::train_batch_output_heads(
        const std::vector<BatchTrainingItem>& batch, const std::vector<size_t>& active_indices,
        const std::vector<MultiTokenPredictionIndex>& valid_heads, BatchIndex batch_size,
        VulkanQueue& queue, std::vector<float>& primary_losses, BatchTrainingTiming& timing)
    {
        const auto started = std::chrono::steady_clock::now();
        std::array<cpu_fixed_vector<int, BatchIndex>, static_cast<size_t>(MultiTokenPredictionIndex::MAX)> expected_by_head;
        std::array<cpu_fixed_vector<int, BatchIndex>, static_cast<size_t>(MultiTokenPredictionIndex::MAX)> active_by_head;
        std::vector<MultiTokenPredictionIndex> head_order;
        for (const auto head : enum_iterator1D<MultiTokenPredictionIndex>())
            if (head != MultiTokenPredictionIndex::START)
                head_order.push_back(head);
        head_order.push_back(MultiTokenPredictionIndex::START);

        size_t active_prediction_count = 0;
        for (const auto count : valid_heads)
            active_prediction_count += static_cast<size_t>(count);
        assert(active_prediction_count > 0);
        const float loss_gradient_scale = 1.0f / static_cast<float>(active_prediction_count);

        for (const auto head : head_order)
        {
            auto& expected = expected_by_head[static_cast<size_t>(head)];
            auto& active_flags = active_by_head[static_cast<size_t>(head)];
            for (size_t active = 0; active < active_indices.size(); ++active)
            {
                const bool used = head < valid_heads[active];
                active_flags.push_back(used ? 1 : 0);
                if (!used)
                {
                    expected.push_back(0);
                    continue;
                }
                const auto& line = batch[active_indices[active]].line;
                const auto& item = batch[active_indices[active]];
                const int input_length = item.context_length.has_value()
                    ? static_cast<int>(*item.context_length)
                    : batch_input_len(static_cast<int>(line.size()));
                expected.push_back(static_cast<int>(target_for_head(line, input_length, head)));
            }
        }

        for (const auto head : head_order)
        {
            if (std::none_of(valid_heads.begin(), valid_heads.end(), [head](auto count) { return head < count; }))
                continue;
            m_output_layers[head].forward_batched(m_batched_output_workspace->h_last, batch_size, m_batched_output_workspace->logits, queue);
            m_batched_output_workspace->expected_tokens.copy_from_cpu(queue, expected_by_head[static_cast<size_t>(head)]);
            m_batched_output_workspace->active_examples.copy_from_cpu(queue, active_by_head[static_cast<size_t>(head)]);
            m_output_layers[head].compute_batched_delta(
                m_batched_output_workspace->logits, batch_size, *m_batched_output_workspace, queue,
                loss_gradient_scale);
            m_output_layers[head].backward_batched_accumulate(
                m_batched_output_workspace->delta, m_batched_output_workspace->h_last, batch_size,
                m_batched_output_workspace->dh_last, m_gradient_accumulation_workspace->output_layers[head]);
            if (head == MultiTokenPredictionIndex::START)
            {
                cpu_fixed_vector<float, BatchIndex> losses;
                losses.set_size(batch_size);
                m_batched_output_workspace->losses.copy_to_cpu(queue, losses);
                for (size_t active = 0; active < active_indices.size(); ++active)
                    primary_losses[active] = losses[static_cast<BatchIndex>(active)];
            }
        }
        timing.backward_output_ms += elapsed_ms(started);
    }

    void TextTrainer::train_window_all_positions(
        const std::vector<BatchTrainingItem>& batch, const std::vector<size_t>& active_indices,
        const PackedBatchInput& packed, VulkanQueue& queue, std::vector<float>& primary_losses,
        BatchTrainingTiming& timing)
    {
        const auto started = std::chrono::steady_clock::now();
        struct Prediction { int row; size_t active; MultiTokenPredictionIndex head; TokenID target; };
        std::vector<Prediction> predictions;
        for (size_t active = 0; active < active_indices.size(); ++active)
        {
            const auto& item = batch[active_indices[active]];
            const int context = static_cast<int>(*item.context_length);
            const auto packed_batch = static_cast<BatchIndex>(active);
            const int begin = static_cast<int>(packed.row_begin(packed_batch));
            for (int local = 0; local < context; local += static_cast<int>(m_window_stride))
                for (const auto head : enum_iterator1D<MultiTokenPredictionIndex>())
                {
                    const int target = local + 1 + static_cast<int>(head);
                    if (target < static_cast<int>(item.line.size()))
                        predictions.push_back({begin + local, active, head,
                            item.line[static_cast<PositionIndex>(target)]});
                }
        }
        assert(!predictions.empty());
        const float scale = 1.0f / static_cast<float>(predictions.size());
        std::vector<double> primary_sums(active_indices.size());
        std::vector<size_t> primary_counts(active_indices.size());
        auto& back = *m_backward_workspace;
        while (!predictions.empty())
        {
            const auto head = predictions.front().head;
            // Predictions are generated position-major; collect one head into a contiguous GPU batch.
            std::vector<size_t> selected;
            selected.reserve(static_cast<size_t>(BatchIndex::MAX));
            for (size_t i = 0; i < predictions.size() && selected.size() < static_cast<size_t>(BatchIndex::MAX); ++i)
                if (predictions[i].head == head)
                    selected.push_back(i);
            cpu_fixed_vector<int, BatchIndex> rows, expected, active_flags;
            for (const size_t i : selected)
            {
                rows.push_back(predictions[i].row);
                expected.push_back(static_cast<int>(predictions[i].target));
                active_flags.push_back(1);
            }
            const auto count = static_cast<BatchIndex>(selected.size());
            m_batched_output_workspace->row_indices.copy_from_cpu(queue, rows);
            gather_indexed_hidden(m_forward_workspace->h, m_batched_output_workspace->row_indices,
                m_batched_output_workspace->h_last, static_cast<int>(count));
            m_batched_output_workspace->expected_tokens.copy_from_cpu(queue, expected);
            m_batched_output_workspace->active_examples.copy_from_cpu(queue, active_flags);
            m_batched_output_workspace->dh_last.zero(queue);
            m_output_layers[head].forward_batched(m_batched_output_workspace->h_last, count, m_batched_output_workspace->logits, queue);
            m_output_layers[head].compute_batched_delta(m_batched_output_workspace->logits, count,
                *m_batched_output_workspace, queue, scale);
            m_output_layers[head].backward_batched_accumulate(m_batched_output_workspace->delta,
                m_batched_output_workspace->h_last, count, m_batched_output_workspace->dh_last,
                m_gradient_accumulation_workspace->output_layers[head]);
            add_indexed_hidden_gradient(m_batched_output_workspace->dh_last,
                m_batched_output_workspace->row_indices, back.dh, static_cast<int>(count));
            if (head == MultiTokenPredictionIndex::START)
            {
                cpu_fixed_vector<float, BatchIndex> losses;
                losses.set_size(count);
                m_batched_output_workspace->losses.copy_to_cpu(queue, losses);
                for (size_t sample = 0; sample < selected.size(); ++sample)
                {
                    const auto active = predictions[selected[sample]].active;
                    primary_sums[active] += losses[static_cast<BatchIndex>(sample)];
                    ++primary_counts[active];
                }
            }
            // Remove the selected predictions while retaining other heads.
            for (auto it = selected.rbegin(); it != selected.rend(); ++it)
                predictions.erase(predictions.begin() + static_cast<std::ptrdiff_t>(*it));
        }
        for (size_t i = 0; i < primary_losses.size(); ++i)
            if (primary_counts[i]) primary_losses[i] = static_cast<float>(primary_sums[i] / primary_counts[i]);
        timing.backward_output_ms += elapsed_ms(started);
    }

    void TextTrainer::train_batch_transformer_backward(PositionIndex packed_rows, BatchTrainingTiming& timing)
    {
        auto& back = *m_backward_workspace;
        auto& forward = *m_forward_workspace;
        auto* dh = &back.dh;
        auto* din = &back.din;
        const auto started = std::chrono::steady_clock::now();
        const bool collect_diagnostics = m_backward_diagnostics_pending;
        if (collect_diagnostics)
            log_hidden_gradient_diagnostics(*dh, packed_rows,
                std::format("output heads -> transformer layer {}", m_transformer_blocks.size() - 1));
        for (int index = static_cast<int>(m_transformer_blocks.size()) - 1;
             index >= 0; --index)
        {
            m_transformer_blocks[index].backward(*dh, *din, back.transformer_block,
                forward.transformer_workspaces[index], true, collect_diagnostics, index);
            if (static_cast<size_t>(index) >= m_frozen_transformer_block_count)
                m_transformer_blocks[index].accumulate_gradients(
                    back.transformer_block,
                    m_gradient_accumulation_workspace->transformer_blocks[index]);
            if (collect_diagnostics)
                log_hidden_gradient_diagnostics(*din, packed_rows,
                    index == 0 ? "transformer layer 0 -> input embeddings" :
                    std::format("transformer layer {} -> layer {}", index, index - 1));
            std::swap(dh, din);
        }
        m_backward_diagnostics_pending = false;
        timing.backward_transformer_ms += elapsed_ms(started);
        (void) packed_rows;
    }

    size_t TextTrainer::finish_converged_batch_items(
        std::vector<BatchTrainingItem>& batch, const std::vector<size_t>& indices,
        const std::vector<float>& losses, size_t step)
    {
        if (!m_example_convergence_enabled)
            return 0;
        size_t finished = 0;
        for (size_t active = 0; active < indices.size(); ++active)
        {
            if (losses[active] >= CONVERGENCE_THRESHOLD)
                continue;
            auto& item = batch[indices[active]];
            const int input_length = item.context_length.has_value()
                ? static_cast<int>(*item.context_length)
                : batch_input_len(static_cast<int>(item.line.size()));
            const auto target = target_for_head(item.line, input_length, MultiTokenPredictionIndex::START);
            LOG_INFO("Convergence reached after {} steps for expected '{}', full string: '{}', input size: {}",
                step + 1, escape_whitespace_for_log(m_corpus.get_token_from_id(target)),
                escape_whitespace_for_log(m_corpus.get_line(item.line).value_or("")), input_length);
            item.finished = true;
            ++finished;
            m_stats.record_learning_success();
        }
        return finished;
    }

    size_t TextTrainer::train_batch_step(std::vector<BatchTrainingItem>& batch, size_t step, BatchTrainingTiming& timing)
    {
        reset_gradient_accumulators();
        std::vector<size_t> active_indices;
        std::vector<CpuInputLine> contexts;
        collect_active_batch_inputs(batch, active_indices, contexts);
        if (active_indices.empty())
            return 0;

        PackedBatchInput packed(contexts);
        const auto rows = packed.packed_rows();
        auto& queue = vulkan_runtime::get_queue(0);
        auto& forward = *m_forward_workspace;
        const auto forward_started = std::chrono::steady_clock::now();
        forward.reset(queue, rows);
        m_input_layer.propagate_forward(packed, *m_gpu_packed_batch, forward.h);
        for (size_t i = 0; i < m_transformer_blocks.size(); ++i)
            m_transformer_blocks[i].forward_batched(forward.h, rows, *m_gpu_packed_batch, forward.transformer_workspaces[i]);
        timing.forward_ms += elapsed_ms(forward_started);

        auto& back = *m_backward_workspace;
        back.reset(queue, rows);
        std::vector<float> losses(active_indices.size(), std::numeric_limits<float>::infinity());
        const auto backward_started = std::chrono::steady_clock::now();
        back.dh.zero(queue);
        back.din.zero(queue);
        train_window_all_positions(batch, active_indices, packed, queue, losses, timing);
        for (const float loss : losses)
        {
            if (std::isfinite(loss))
            {
                timing.primary_loss_sum += loss;
                ++timing.primary_loss_count;
            }
        }
        train_batch_transformer_backward(rows, timing);
        const auto input_started = std::chrono::steady_clock::now();
        auto& input_gradient = (m_transformer_blocks.size() % 2 == 0) ? back.dh : back.din;
        m_input_layer.accumulate_backward_packed(
            packed, input_gradient, m_gradient_accumulation_workspace->embeddings);
        timing.backward_input_ms += elapsed_ms(input_started);
        timing.backward_ms += elapsed_ms(backward_started);
        const auto apply_started = std::chrono::steady_clock::now();
        apply_accumulated_gradients(1.0f);
        timing.apply_ms += elapsed_ms(apply_started);
        finish_converged_batch_items(batch, active_indices, losses, step);
        return active_indices.size();
    }

    size_t TextTrainer::train_batch_items(std::vector<BatchTrainingItem>& batch, size_t steps, BatchTrainingTiming& timing)
    {
        size_t iterations = 0;
        for (size_t step = 0; step < steps; ++step)
        {
            const size_t active = train_batch_step(batch, step, timing);
            if (active == 0)
                break;
            ++timing.rounds;
            iterations += active;
        }
        return iterations;
    }

}
