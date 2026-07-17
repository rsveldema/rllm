#include <TextTrainer.hpp>
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

        MultiTokenPredictionIndex valid_head_count(int sequence_length, int input_length)
        {
            return static_cast<MultiTokenPredictionIndex>(std::min(
                static_cast<int>(MultiTokenPredictionIndex::MAX), sequence_length - input_length));
        }

        TokenID target_for_head(const CpuInputLine& line, int input_length, MultiTokenPredictionIndex head)
        {
            return line[static_cast<PositionIndex>(input_length + static_cast<int>(head))];
        }

        bool checkpoint_due(const std::optional<std::chrono::seconds>& interval, std::chrono::steady_clock::time_point& last)
        {
            if (!interval.has_value() || std::chrono::steady_clock::now() - last < *interval)
                return false;
            last = std::chrono::steady_clock::now();
            return true;
        }

        void gather_last_hidden(
            // OFFLOAD_PARAMETERS(h, last_rows, h_last, batch_size)
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
            const fixed_size_vector<int, BatchIndex>& last_rows,
            fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& h_last,
            int batch_size
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(batch_size));
            OFFLOAD_PARFOR_2D_PARAM(queue, batch, d, grid, (h, last_rows, h_last, batch_size))
            h_last[batch, d] = h[last_rows[batch], d];
            ENDFOR
        }

        void scatter_last_hidden_gradient(
            // OFFLOAD_PARAMETERS(dh_last, last_rows, dh, batch_size)
            const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& dh_last,
            const fixed_size_vector<int, BatchIndex>& last_rows,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            int batch_size
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(batch_size));
            OFFLOAD_PARFOR_2D_PARAM(queue, batch, d, grid, (dh_last, last_rows, dh, batch_size))
            dh[last_rows[batch], d] = dh_last[batch, d];
            ENDFOR
        }
    }

    std::vector<TextTrainer::BatchTrainingItem> TextTrainer::make_training_batch(
        const std::vector<CpuInputLine>& lines, const std::vector<size_t>& indices,
        size_t start, size_t count, std::mt19937& rng) const
    {
        std::vector<BatchTrainingItem> batch;
        batch.reserve(count);
        for (size_t offset = 0; offset < count; ++offset)
        {
            const auto& line = lines[indices[start + offset]];
            if (static_cast<int>(line.size()) < 2)
                continue;
            switch (m_training_method)
            {
            case TrainingMethod::RANDOM_LINE_FULL:
                batch.push_back({line});
                break;
            case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
            {
                std::uniform_int_distribution<int> length(2, static_cast<int>(line.size()));
                CpuInputLine input;
                line.sub_array(input, static_cast<PositionIndex>(length(rng)));
                batch.push_back({input});
                break;
            }
            case TrainingMethod::TWO_TOK:
            case TrainingMethod::THREE_TOK:
            {
                const int requested = m_training_method == TrainingMethod::TWO_TOK ? 2 : 3;
                CpuInputLine input;
                line.sub_array(input, static_cast<PositionIndex>(std::min(requested, static_cast<int>(line.size()))));
                batch.push_back({input});
                break;
            }
            case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                // Preserve the existing exclusive upper bound: prefixes [2, line.size()).
                for (size_t length = 2; length < static_cast<size_t>(line.size()); ++length)
                {
                    CpuInputLine input;
                    line.sub_array(input, static_cast<PositionIndex>(length));
                    batch.push_back({input});
                }
                break;
            case TrainingMethod::WINDOW:
            case TrainingMethod::REVERSE_WINDOW:
                assert(false);
                break;
            }
        }
        return batch;
    }

    void TextTrainer::collect_active_batch_inputs(
        const std::vector<BatchTrainingItem>& batch, std::vector<size_t>& indices,
        std::vector<CpuInputLine>& contexts, std::vector<MultiTokenPredictionIndex>& heads) const
    {
        for (size_t i = 0; i < batch.size(); ++i)
        {
            if (batch[i].finished)
                continue;
            const int length = static_cast<int>(batch[i].line.size());
            const int input_length = batch_input_len(length);
            CpuInputLine context;
            batch[i].line.sub_array(context, static_cast<PositionIndex>(input_length));
            indices.push_back(i);
            contexts.push_back(context);
            heads.push_back(valid_head_count(length, input_length));
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
                expected.push_back(static_cast<int>(target_for_head(
                    line, batch_input_len(static_cast<int>(line.size())), head)));
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
                m_batched_output_workspace->logits, batch_size, *m_batched_output_workspace, queue);
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

    void TextTrainer::train_batch_transformer_backward(PositionIndex packed_rows, BatchTrainingTiming& timing)
    {
        auto& back = *m_backward_workspace;
        auto& forward = *m_forward_workspace;
        auto* dh = &back.dh;
        auto* din = &back.din;
        const auto started = std::chrono::steady_clock::now();
        for (int index = static_cast<int>(m_transformer_blocks.size()) - 1; index >= 0; --index)
        {
            m_transformer_blocks[index].backward(*dh, *din, back.transformer_block, forward.transformer_workspaces[index]);
            m_transformer_blocks[index].accumulate_gradients(back.transformer_block, m_gradient_accumulation_workspace->transformer_blocks[index]);
            std::swap(dh, din);
        }
        timing.backward_transformer_ms += elapsed_ms(started);
        (void) packed_rows;
    }

    size_t TextTrainer::finish_converged_batch_items(
        std::vector<BatchTrainingItem>& batch, const std::vector<size_t>& indices,
        const std::vector<float>& losses, size_t step)
    {
        size_t finished = 0;
        for (size_t active = 0; active < indices.size(); ++active)
        {
            if (losses[active] >= CONVERGENCE_THRESHOLD)
                continue;
            auto& item = batch[indices[active]];
            const int input_length = batch_input_len(static_cast<int>(item.line.size()));
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
        std::vector<MultiTokenPredictionIndex> valid_heads;
        collect_active_batch_inputs(batch, active_indices, contexts, valid_heads);
        if (active_indices.empty())
            return 0;

        PackedBatchInput packed(contexts);
        const auto rows = packed.packed_rows();
        const auto batch_size = packed.batch_size();
        auto& queue = vulkan_runtime::get_queue(0);
        auto& forward = *m_forward_workspace;
        const auto forward_started = std::chrono::steady_clock::now();
        forward.reset(queue, rows);
        m_input_layer.propagate_forward(packed, *m_gpu_packed_batch, forward.h);
        for (size_t i = 0; i < m_transformer_blocks.size(); ++i)
            m_transformer_blocks[i].forward_batched(forward.h, rows, *m_gpu_packed_batch, forward.transformer_workspaces[i]);
        gather_last_hidden(forward.h, m_gpu_packed_batch->last_row, m_batched_output_workspace->h_last, static_cast<int>(batch_size));
        timing.forward_ms += elapsed_ms(forward_started);

        auto& back = *m_backward_workspace;
        back.reset(queue, rows);
        m_batched_output_workspace->dh_last.zero(queue);
        std::vector<float> losses(active_indices.size(), std::numeric_limits<float>::infinity());
        const auto backward_started = std::chrono::steady_clock::now();
        train_batch_output_heads(batch, active_indices, valid_heads, batch_size, queue, losses, timing);
        for (const float loss : losses)
        {
            if (std::isfinite(loss))
            {
                timing.primary_loss_sum += loss;
                ++timing.primary_loss_count;
            }
        }
        back.dh.zero(queue);
        back.din.zero(queue);
        scatter_last_hidden_gradient(m_batched_output_workspace->dh_last, m_gpu_packed_batch->last_row, back.dh, static_cast<int>(batch_size));
        train_batch_transformer_backward(rows, timing);
        const auto input_started = std::chrono::steady_clock::now();
        auto& input_gradient = (m_transformer_blocks.size() % 2 == 0) ? back.dh : back.din;
        m_input_layer.accumulate_backward_packed(packed, input_gradient, m_gradient_accumulation_workspace->embeddings);
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

    void TextTrainer::batch_train(
        size_t epoch, const std::vector<CpuInputLine>& lines, bool verbose, size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval,
        std::chrono::steady_clock::time_point& last_checkpoint_at, std::mt19937& rng,
        std::optional<size_t> epoch_size)
    {
        (void) verbose;
        const size_t epoch_lines = std::min(epoch_size.value_or(lines.size()), lines.size());
        std::vector<size_t> indices(lines.size());
        std::iota(indices.begin(), indices.end(), 0);
        auto selected_inputs = make_training_batch(lines, indices, 0, epoch_lines, rng);

        if (false)
        {
            std::stable_sort(selected_inputs.begin(), selected_inputs.end(), [](const auto& a, const auto& b) {
                return a.line.size() < b.line.size();
            });
        } else {
            std::shuffle(selected_inputs.begin(), selected_inputs.end(), rng);
        }


        const size_t selected_lines = selected_inputs.size();
        const auto epoch_started_at = std::chrono::steady_clock::now();
        for (size_t start = 0; start < selected_lines; start += m_micro_batch_size)
        {
            const size_t count = std::min(m_micro_batch_size, selected_lines - start);
            if (checkpoint_due(checkpointing_interval, last_checkpoint_at))
            {
                LOG_INFO("Creating timed checkpoint at epoch: {}, line: {}, epoch lines todo: {}", epoch, start + 1, selected_lines);
                checkpoint();
            }
            std::vector<BatchTrainingItem> batch(
                selected_inputs.begin() + static_cast<std::ptrdiff_t>(start),
                selected_inputs.begin() + static_cast<std::ptrdiff_t>(start + count));
            const auto started = std::chrono::steady_clock::now();
            BatchTrainingTiming timing;
            const size_t iterations = train_batch_items(batch, m_learn_depth, timing);
            const double total_ms = elapsed_ms(started);
            LOG_INFO("Epoch[{}%] {}[{}..{}]: {:0.2f}% done (micro-batch {}, training loss {:.6f}, avg {:.2f} ms/example, iterations total {}, avg {:.2f}/example, rounds {}, batch {:.2f} ms, forward {:.2f} ms, backward {:.2f} ms, apply {:.2f} ms, out {:.2f} ms, transformer {:.2f} ms, input {:.2f} ms)",
                epoch / static_cast<float>(num_epochs) * 100.0f, training_method_to_string(m_training_method), start + 1, start + count,
                static_cast<float>(start + count) / static_cast<float>(selected_lines) * 100.0f,
                batch.size(), timing.average_primary_loss(), total_ms / batch.size(), iterations, static_cast<double>(iterations) / batch.size(),
                timing.rounds, total_ms, timing.forward_ms, timing.backward_ms,
                timing.apply_ms, timing.backward_output_ms, timing.backward_transformer_ms, timing.backward_input_ms);
            log_training_progress(
                training_method_to_string(m_training_method), epoch, num_epochs,
                start + 1, start + count, selected_lines, batch.size(), iterations, total_ms, timing);

            const size_t batch_number = start / m_micro_batch_size + 1;
            if (batch_number % 10 == 0)
            {
                const double elapsed_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - epoch_started_at).count();
                const size_t completed = start + count;
                const double remaining_seconds = elapsed_seconds *
                    static_cast<double>(selected_lines - completed) / static_cast<double>(completed);
                const double epoch_fraction = static_cast<double>(completed) / static_cast<double>(selected_lines);
                const double estimated_epoch_seconds = elapsed_seconds / epoch_fraction;
                const double all_epochs_remaining_seconds = estimated_epoch_seconds *
                    (static_cast<double>(num_epochs - epoch) - epoch_fraction);
                LOG_INFO(
                    "Epoch {} batch {} ETA until epoch finishes: {}; ETA until all planned epochs finish: {}",
                    epoch, batch_number, format_eta_for_log(remaining_seconds),
                    format_eta_for_log(all_epochs_remaining_seconds));
            }
        }
    }
}
