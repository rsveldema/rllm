#include <TextTrainer.hpp>
#include <OptimizerDiagnostics.hpp>
#include "TextTrainerInternal.hpp"
#include <LogFormatting.hpp>
#include <NumericLoss.hpp>
#include <dust/Dust.hpp>
#include <vecmath.hpp>

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

        int string_table_index_for_head(
            const CpuInputLine& line,
            const std::vector<uint8_t>& first_string_table_index,
            int input_length,
            MultiTokenPredictionIndex head)
        {
            const int target_index = input_length + static_cast<int>(head);
            assert(target_index < static_cast<int>(line.size()));
            const auto pos = static_cast<PositionIndex>(target_index);
            const size_t index = line.get_string_table_index(pos);
            if (index == NO_STRING_TABLE_INDEX)
                return -1;
            assert(static_cast<size_t>(target_index) < first_string_table_index.size());
            const auto category = token_string_category(line[pos]);
            if (category != TokenStringCategory::Integer && category != TokenStringCategory::Float &&
                first_string_table_index[static_cast<size_t>(target_index)] != 0)
                return -1;
            assert(index < static_cast<size_t>(PositionIndex::MAX));
            return static_cast<int>(index);
        }

        PositionIndex string_table_index_count_for_expected(
            const cpu_fixed_vector<int, BatchIndex>& expected_value_counts)
        {
            int max_count = 0;
            for (const auto batch : enum_iterator1D<BatchIndex>(expected_value_counts.size()))
                max_count = std::max(max_count, expected_value_counts[batch]);
            return static_cast<PositionIndex>(max_count);
        }

        int value_count_for_head(const CpuInputLine& line, int input_length,
            MultiTokenPredictionIndex head, int expected_index)
        {
            if (expected_index < 0)
                return 0;
            const auto pos = static_cast<PositionIndex>(input_length + static_cast<int>(head));
            return static_cast<int>(line.value_count_for_token(line[pos]));
        }

        void set_numeric_distance_costs(const CpuInputLine& line, PositionIndex target_pos,
            BatchIndex batch, cpu_fixed_matrix<float, BatchIndex, PositionIndex>& costs)
        {
            const TokenID token = line[target_pos];
            const auto category = token_string_category(token);
            if (category != TokenStringCategory::Integer && category != TokenStringCategory::Float)
                return;
            const size_t expected_index = line.get_string_table_index(target_pos);
            if (expected_index == NO_STRING_TABLE_INDEX)
                return;
            const auto& values = line.value_table_for_token(token);
            for (size_t index = 0; index < values.size(); ++index)
                costs[batch, static_cast<PositionIndex>(index)] =
                    numeric_distance_cost(values[index], values[expected_index]);
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

        void add_entity_consistency_gradient(
            // OFFLOAD_PARAMETERS(h, first_rows, repeated_rows, dh, count, scale)
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
            const fixed_size_vector<int, BatchIndex>& first_rows,
            const fixed_size_vector<int, BatchIndex>& repeated_rows,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            int count,
            float scale
            // END_OFFLOAD_PARAMETERS
        )
        {
            auto& queue = vulkan_runtime::get_queue(0);
            const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(static_cast<BatchIndex>(count));
            OFFLOAD_PARFOR_2D_PARAM(queue, pair, d, grid, (h, first_rows, repeated_rows, dh, count, scale))
            const int first = first_rows[pair];
            const int repeated = repeated_rows[pair];
            float delta = h[repeated, d];
            delta -= h[first, d];
            delta *= scale;
            dh[repeated, d] += delta;
            dh[first, d] -= delta;
            ENDFOR
        }

        void add_entity_consistency_loss(
            const std::vector<CpuInputLine>& contexts,
            const PackedBatchInput& packed,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            BatchedOutputWorkspace& workspace,
            VulkanQueue& queue)
        {
            std::vector<std::pair<int, int>> pairs;
            for (size_t active = 0; active < contexts.size(); ++active)
            {
                const auto& line = contexts[active];
                const size_t context = static_cast<size_t>(line.size());
                std::vector<int> first_strings(line.string_table_value.size(), -1);
                std::vector<int> first_integers(line.integer_constant_value.size(), -1);
                std::vector<int> first_floats(line.float_constant_value.size(), -1);
                const int begin = static_cast<int>(packed.row_begin(static_cast<BatchIndex>(active)));
                for (size_t local = 0; local < context; ++local)
                {
                    const auto pos = static_cast<PositionIndex>(local);
                    const size_t index = line.get_string_table_index(pos);
                    if (index == NO_STRING_TABLE_INDEX)
                        continue;
                    const auto category = token_string_category(line[pos]);
                    auto& first = category == TokenStringCategory::Integer ? first_integers
                        : (category == TokenStringCategory::Float ? first_floats : first_strings);
                    if (first[index] < 0)
                        first[index] = begin + static_cast<int>(local);
                    else
                        pairs.emplace_back(first[index], begin + static_cast<int>(local));
                }
            }
            if (pairs.empty())
                return;
            const float scale = OutputLayer::ENTITY_CONSISTENCY_WEIGHT /
                (static_cast<float>(pairs.size()) * static_cast<float>(EmbeddingDimension::MAX));
            for (size_t offset = 0; offset < pairs.size(); offset += static_cast<size_t>(BatchIndex::MAX))
            {
                cpu_fixed_vector<int, BatchIndex> first_rows, repeated_rows;
                const size_t end = std::min(pairs.size(), offset + static_cast<size_t>(BatchIndex::MAX));
                for (size_t i = offset; i < end; ++i)
                {
                    first_rows.push_back(pairs[i].first);
                    repeated_rows.push_back(pairs[i].second);
                }
                workspace.row_indices.copy_from_cpu(queue, first_rows);
                workspace.paired_row_indices.copy_from_cpu(queue, repeated_rows);
                add_entity_consistency_gradient(h, workspace.row_indices,
                    workspace.paired_row_indices, dh, static_cast<int>(end - offset), scale);
            }
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


    void TextTrainer::train_window_all_positions(
        const std::vector<BatchTrainingItem>& batch, const std::vector<size_t>& active_indices,
        const PackedBatchInput& packed,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& hidden,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& hidden_gradient,
        float loss_weight, bool record_primary_loss,
        VulkanQueue& queue, std::vector<float>& primary_losses,
        BatchTrainingTiming& timing)
    {
        const auto started = std::chrono::steady_clock::now();
        struct Prediction { int row; size_t active; MultiTokenPredictionIndex head; TokenID target; int string_table_index; int value_count; int target_pos; };
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
                    {
                        const int index = string_table_index_for_head(
                            item.line, item.first_string_table_index, local + 1, head);
                        predictions.push_back({begin + local, active, head,
                            item.line[static_cast<PositionIndex>(target)], index,
                            value_count_for_head(item.line, local + 1, head, index), target});
                    }
                }
        }
        assert(!predictions.empty());
        const float scale = loss_weight / static_cast<float>(predictions.size());
        std::vector<double> primary_sums(active_indices.size());
        std::vector<size_t> primary_counts(active_indices.size());
        while (!predictions.empty())
        {
            const auto head = predictions.front().head;
            // Predictions are generated position-major; collect one head into a contiguous GPU batch.
            std::vector<size_t> selected;
            selected.reserve(static_cast<size_t>(BatchIndex::MAX));
            for (size_t i = 0; i < predictions.size() && selected.size() < static_cast<size_t>(BatchIndex::MAX); ++i)
                if (predictions[i].head == head)
                    selected.push_back(i);
            cpu_fixed_vector<int, BatchIndex> rows, expected, expected_string_table_index, expected_value_count, active_flags;
            for (const size_t i : selected)
            {
                rows.push_back(predictions[i].row);
                expected.push_back(static_cast<int>(predictions[i].target));
                expected_string_table_index.push_back(predictions[i].string_table_index);
                expected_value_count.push_back(predictions[i].value_count);
                active_flags.push_back(1);
            }
            const auto count = static_cast<BatchIndex>(selected.size());
            m_batched_output_workspace->row_indices.copy_from_cpu(queue, rows);
            gather_indexed_hidden(hidden, m_batched_output_workspace->row_indices,
                m_batched_output_workspace->h_last, static_cast<int>(count));
            m_batched_output_workspace->expected_tokens.copy_from_cpu(queue, expected);
            m_batched_output_workspace->expected_string_table_indices.copy_from_cpu(
                queue, expected_string_table_index);
            m_batched_output_workspace->expected_value_counts.copy_from_cpu(queue, expected_value_count);
            m_batched_output_workspace->active_examples.copy_from_cpu(queue, active_flags);
            cpu_fixed_matrix<float, BatchIndex, PositionIndex> numeric_costs;
            numeric_costs.zero();
            for (size_t sample = 0; sample < selected.size(); ++sample)
            {
                const auto& prediction = predictions[selected[sample]];
                const auto& line = batch[active_indices[prediction.active]].line;
                set_numeric_distance_costs(line, static_cast<PositionIndex>(prediction.target_pos),
                    static_cast<BatchIndex>(sample), numeric_costs);
            }
            m_batched_output_workspace->numeric_distance_costs.copy_from_cpu(queue, numeric_costs);
            m_batched_output_workspace->dh_last.zero(queue);
            m_output_layers[head].forward_batched(m_batched_output_workspace->h_last, count, m_batched_output_workspace->logits, queue);
            const PositionIndex string_table_index_count =
                string_table_index_count_for_expected(expected_value_count);
            m_output_layers[head].compute_batched_delta(m_batched_output_workspace->logits, count,
                *m_batched_output_workspace, queue,
                m_batched_output_workspace->expected_string_table_indices, scale,
                string_table_index_count);
            if (m_training_algorithm == TrainingAlgorithm::Dust)
            {
                const auto loss_bytes = static_cast<VkDeviceSize>(
                    static_cast<size_t>(count) * sizeof(float));
                m_batched_output_workspace->clean_losses.device_buffer().copy_from(
                    queue, m_batched_output_workspace->losses.device_buffer(), loss_bytes);
                m_batched_output_workspace->dust_error.zero(queue);
                const float sigma = m_dust_settings.activation_noise;
                const float inverse_population_sigma =
                    scale / (static_cast<float>(m_dust_settings.population) * sigma);
                const float inverse_population_attention_sigma =
                    scale / (static_cast<float>(m_dust_settings.population) *
                        m_dust_settings.attention_noise);
                auto& dust_forward = *m_dust_forward_workspace;
                auto& scratch = m_backward_workspace->transformer_block;
                const auto packed_rows = packed.packed_rows();
                for (size_t member = 0; member < m_dust_settings.population; ++member)
                {
                    // Evaluate one forward-only member with independent,
                    // reproducible perturbations at every linear projection.
                    const uint64_t member_seed = ++m_dust_seed;
                    dust::ForwardPerturbation perturbation(packed_rows, member_seed,
                        sigma, m_dust_settings.attention_noise);
                    dust_forward.reset(queue, packed_rows);
                    m_input_layer.propagate_forward(
                        packed, *m_gpu_packed_batch, dust_forward.h);
                    for (size_t execution = 0;
                         execution < m_transformer_execution_order.size(); ++execution)
                    {
                        if (const auto iteration = loop_iteration_for_execution(execution))
                        {
                            if (*iteration == 0)
                                dust_forward.loop_input = dust_forward.h;
                            prepare_loop_iteration_embedding(dust_forward, *iteration, queue);
                            apply_loop_transition(dust_forward.h, dust_forward.loop_input,
                                dust_forward.loop_iteration_embedding);
                        }
                        const size_t block = m_transformer_execution_order[execution];
                        m_transformer_blocks[block].forward_batched(
                            dust_forward.h, packed_rows, *m_gpu_packed_batch,
                            dust_forward.transformer_workspaces[execution],
                            transformer_residual_scale(block), &perturbation, execution);
                    }
                    gather_indexed_hidden(dust_forward.h,
                        m_batched_output_workspace->row_indices,
                        m_batched_output_workspace->dust_perturbed_hidden,
                        static_cast<int>(count));
                    m_output_layers[head].forward_batched(
                        m_batched_output_workspace->dust_perturbed_hidden, count,
                        m_batched_output_workspace->logits, queue);
                    dust::gaussian_logit_noise(
                        m_batched_output_workspace->dust_noise, count,
                        ++m_dust_seed);
                    dust::perturb_logits(m_batched_output_workspace->logits,
                        m_batched_output_workspace->dust_noise, count, sigma);
                    m_output_layers[head].compute_batched_delta(
                        m_batched_output_workspace->logits, count,
                        *m_batched_output_workspace, queue,
                        m_batched_output_workspace->expected_string_table_indices,
                        scale, string_table_index_count);
                    dust::compute_rewards(
                        m_batched_output_workspace->clean_losses,
                        m_batched_output_workspace->losses,
                        m_batched_output_workspace->dust_rewards, count);
                    dust::accumulate_logit_error(
                        m_batched_output_workspace->dust_rewards,
                        m_batched_output_workspace->dust_noise,
                        m_batched_output_workspace->dust_error, count,
                        inverse_population_sigma);

                    // Convert sparse prediction rewards to packed row rewards.
                    // K/V perturbations receive causal credit from later rows in
                    // the same example because those rows can attend to them.
                    m_batched_output_workspace->dust_row_rewards.zero(queue);
                    dust::scatter_rewards(
                        m_batched_output_workspace->dust_rewards,
                        m_batched_output_workspace->row_indices,
                        m_batched_output_workspace->dust_row_rewards, count);
                    dust::causal_credit_rewards(
                        m_batched_output_workspace->dust_row_rewards,
                        m_batched_output_workspace->dust_attention_rewards,
                        m_gpu_packed_batch->row_batch, m_gpu_packed_batch->row_end,
                        packed_rows, m_dust_settings.credit_decay);

                    for (size_t execution = 0;
                         execution < m_transformer_execution_order.size(); ++execution)
                    {
                        const size_t block = m_transformer_execution_order[execution];
                        if (block < m_frozen_transformer_block_count)
                            continue;
                        const auto& cached =
                            m_forward_workspace->transformer_workspaces[execution];
                        auto& accumulator =
                            m_gradient_accumulation_workspace->transformer_blocks[block];

                        const auto hidden_site = [&](dust::ProjectionSite site,
                            const fixed_size_vector<float, PositionIndex>& rewards,
                            auto& gradient, const auto& input, float inverse_sigma)
                        {
                            scratch.d_h_norm_attn.zero(queue);
                            dust::gaussian_hidden_noise(perturbation.hidden_noise,
                                packed_rows, perturbation.site_seed(site, execution));
                            dust::accumulate_hidden_error(rewards,
                                perturbation.hidden_noise, scratch.d_h_norm_attn,
                                packed_rows, inverse_sigma);
                            dust::accumulate_hidden_outer_product(
                                scratch.d_h_norm_attn, input, gradient, packed_rows);
                        };
                        hidden_site(dust::ProjectionSite::Query,
                            m_batched_output_workspace->dust_row_rewards,
                            accumulator.dW_q, cached.h_norm_attn,
                            inverse_population_attention_sigma);
                        hidden_site(dust::ProjectionSite::Key,
                            m_batched_output_workspace->dust_attention_rewards,
                            accumulator.dW_k, cached.h_norm_attn,
                            inverse_population_attention_sigma);
                        hidden_site(dust::ProjectionSite::Value,
                            m_batched_output_workspace->dust_attention_rewards,
                            accumulator.dW_v, cached.h_norm_attn,
                            inverse_population_attention_sigma);
                        hidden_site(dust::ProjectionSite::AttentionOutput,
                            m_batched_output_workspace->dust_row_rewards,
                            accumulator.dW_o, cached.attn_concat,
                            inverse_population_sigma);

                        const auto ff_site = [&](dust::ProjectionSite site,
                            auto& gradient)
                        {
                            scratch.d_gate_pre.zero(queue);
                            dust::gaussian_ff_noise(perturbation.ff_noise,
                                packed_rows, perturbation.site_seed(site, execution));
                            dust::accumulate_ff_error(
                                m_batched_output_workspace->dust_row_rewards,
                                perturbation.ff_noise, scratch.d_gate_pre,
                                packed_rows, inverse_population_sigma);
                            dust::accumulate_ff_hidden_outer_product(
                                scratch.d_gate_pre, cached.h_norm_ff,
                                gradient, packed_rows);
                        };
                        ff_site(dust::ProjectionSite::Gate, accumulator.dW_gate);
                        ff_site(dust::ProjectionSite::Up, accumulator.dW_up);
                        scratch.d_h_norm_attn.zero(queue);
                        dust::gaussian_hidden_noise(perturbation.hidden_noise,
                            packed_rows, perturbation.site_seed(
                                dust::ProjectionSite::Down, execution));
                        dust::accumulate_hidden_error(
                            m_batched_output_workspace->dust_row_rewards,
                            perturbation.hidden_noise, scratch.d_h_norm_attn,
                            packed_rows, inverse_population_sigma);
                        dust::accumulate_hidden_ff_outer_product(
                            scratch.d_h_norm_attn, cached.ffn_act,
                            accumulator.dW_down, packed_rows);
                        accumulator.touched = true;
                    }
                    ++timing.dust_population_evaluations;
                }
                auto& accumulator = m_gradient_accumulation_workspace->output_layers[head];
                dust::accumulate_output_outer_product(
                    m_batched_output_workspace->dust_error,
                    m_batched_output_workspace->h_last,
                    accumulator.dW_lm_head, count);
                accumulator.touched = true;
            }
            else
            {
                m_output_layers[head].backward_batched_accumulate(
                    m_batched_output_workspace->delta,
                    m_batched_output_workspace->string_table_index_delta,
                    m_batched_output_workspace->h_last, count,
                    m_batched_output_workspace->dh_last,
                    m_gradient_accumulation_workspace->output_layers[head],
                    string_table_index_count);
                add_indexed_hidden_gradient(m_batched_output_workspace->dh_last,
                    m_batched_output_workspace->row_indices, hidden_gradient,
                    static_cast<int>(count));
            }
            if (record_primary_loss && head == MultiTokenPredictionIndex::START)
            {
                cpu_fixed_vector<float, BatchIndex> losses;
                losses.set_size(count);
                (m_training_algorithm == TrainingAlgorithm::Dust
                    ? m_batched_output_workspace->clean_losses
                    : m_batched_output_workspace->losses).copy_to_cpu(queue, losses);
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
        for (size_t execution = 0; execution < m_transformer_execution_order.size(); ++execution)
        {
            if (const auto iteration = loop_iteration_for_execution(execution))
            {
                if (*iteration == 0)
                    forward.loop_input = forward.h;
                prepare_loop_iteration_embedding(forward, *iteration, queue);
                apply_loop_transition(forward.h, forward.loop_input, forward.loop_iteration_embedding);
            }
            const size_t block = m_transformer_execution_order[execution];
            m_transformer_blocks[block].forward_batched(
                forward.h, rows, *m_gpu_packed_batch, forward.transformer_workspaces[execution],
                transformer_residual_scale(block));
        }
        timing.forward_ms += elapsed_ms(forward_started);

        auto& back = *m_backward_workspace;
        back.reset(queue, rows);
        std::vector<float> losses(active_indices.size(), std::numeric_limits<float>::infinity());
        const auto backward_started = std::chrono::steady_clock::now();
        back.dh.zero(queue);
        back.din.zero(queue);
        train_window_all_positions(batch, active_indices, packed, forward.h, back.dh,
            1.0f, true, queue, losses, timing);
        if (m_training_algorithm == TrainingAlgorithm::Backprop && m_loop_count > 1)
        {
            const float auxiliary_weight =
                LOOP_AUXILIARY_LOSS_WEIGHT / static_cast<float>(m_loop_count - 1);
            for (size_t execution = 0; execution < m_transformer_execution_order.size(); ++execution)
            {
                const auto iteration = loop_iteration_for_execution(execution);
                if (!iteration || *iteration == 0)
                    continue;
                auto& execution_workspace = forward.transformer_workspaces[execution];
                train_window_all_positions(batch, active_indices, packed,
                    execution_workspace.h_in, execution_workspace.auxiliary_gradient,
                    auxiliary_weight, false, queue, losses, timing);
            }
        }
        if (m_training_algorithm == TrainingAlgorithm::Backprop)
            add_entity_consistency_loss(contexts, packed, forward.h, back.dh,
                *m_batched_output_workspace, queue);
        for (const float loss : losses)
        {
            if (std::isfinite(loss))
            {
                timing.primary_loss_sum += loss;
                ++timing.primary_loss_count;
            }
        }
        if (m_training_algorithm == TrainingAlgorithm::Backprop)
        {
            train_batch_transformer_backward(rows, timing);
            const auto input_started = std::chrono::steady_clock::now();
            auto& input_gradient = (m_transformer_execution_order.size() % 2 == 0)
                ? back.dh : back.din;
            m_input_layer.accumulate_backward_packed(
                packed, input_gradient, m_gradient_accumulation_workspace->embeddings);
            timing.backward_input_ms += elapsed_ms(input_started);
        }
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
