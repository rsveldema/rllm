#include <TextTrainer.hpp>
#include <OptimizerDiagnostics.hpp>
#include "TextTrainerInternal.hpp"
#include <RuntimeConfig.hpp>
#include <LogFormatting.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <enum_iterator2D.hpp>
#include <format>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <numbers>
#include <parallel.hpp>
#include <random>
#include <rllm_vulkan_runtime.hpp>
#include <set>
#include <vecmath.hpp>

#include <nlohmann/json.hpp>

namespace rllm
{
    const char* training_method_to_string(TrainingMethod method)
    {
        switch (method)
        {
        case TrainingMethod::TWO_TOK:
            return "two_tok";
        case TrainingMethod::THREE_TOK:
            return "three_tok";
        case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
            return "increasingly_longer";
        case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
            return "random_line_random_len";
        case TrainingMethod::RANDOM_LINE_FULL:
            return "random_line_full";
        case TrainingMethod::WINDOW:
            return "window";
        case TrainingMethod::REVERSE_WINDOW:
            return "reverse_window";
        }
        return "UNKNOWN";
    }
} // namespace rllm

namespace rllm
{
    static constexpr float NAN_FINDING_HIDDEN_ABS_BOUND = 10000.0f;

    static int mtp_input_len_for_sequence(int seq_len)
    {
        assert(seq_len >= 2);
        const int max_heads = static_cast<int>(MultiTokenPredictionIndex::MAX);
        return std::max(1, seq_len - max_heads);
    }

    static MultiTokenPredictionIndex mtp_valid_head_count_for_sequence(int seq_len, int input_len)
    {
        assert(seq_len >= 2);
        assert(input_len >= 1);
        assert(input_len < seq_len);
        const int max_heads = static_cast<int>(MultiTokenPredictionIndex::MAX);
        const int valid_heads = std::min(max_heads, seq_len - input_len);
        assert(valid_heads >= 1);
        return static_cast<MultiTokenPredictionIndex>(valid_heads);
    }

    static TokenID mtp_target_for_head(const CpuInputLine& line, int input_len, MultiTokenPredictionIndex head)
    {
        const int target_index = input_len + static_cast<int>(head);
        assert(target_index < static_cast<int>(line.size()));
        if (target_index < static_cast<int>(line.size()))
            return line[static_cast<PositionIndex>(target_index)];
        return TokenID::INVALID;
    }

    static void scatter_dh_last_to_row(
        // OFFLOAD_PARAMETERS(dh_last, dh, last_pos)
        const fixed_size_vector<float, EmbeddingDimension>& dh_last,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        PositionIndex last_pos
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, d, enum_iterator1D<EmbeddingDimension>(), (dh_last, dh, last_pos))
        dh[last_pos, d] = dh_last[d];
        ENDFOR
    }

    static fixed_size_vector<int, PositionIndex> make_neural_network_nan_scan_flag(VulkanQueue& queue)
    {
        cpu_fixed_vector<int, PositionIndex> cpu_flag;
        cpu_flag.set_size(PositionIndex::MAX);
        cpu_flag.zero();

        fixed_size_vector<int, PositionIndex> flag;
        flag.copy_from_cpu(queue, cpu_flag);
        return flag;
    }

    static bool neural_network_nan_scan_failed(VulkanQueue& queue, fixed_size_vector<int, PositionIndex>& flag)
    {
        cpu_fixed_vector<int, PositionIndex> cpu_flag;
        flag.copy_to_cpu(queue, cpu_flag);
        return cpu_flag[PositionIndex::START] != 0;
    }

    static void scan_hidden_matrix(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(h, flag, rows, lower_bound, upper_bound)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
        fixed_size_vector<int, PositionIndex>& flag,
        PositionIndex rows,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (h, flag, rows, lower_bound, upper_bound))
        const float value = h[row, col];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_embedding_vector(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(values, flag, lower_bound, upper_bound)
        const fixed_size_vector<float, EmbeddingDimension>& values,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<EmbeddingDimension>(), (values, flag, lower_bound, upper_bound))
        const float value = values[i];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void scan_token_vector(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(values, flag, lower_bound, upper_bound)
        const fixed_size_vector<float, TokenID>& values,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TokenID>(), (values, flag, lower_bound, upper_bound))
        const float value = values[i];
        if (value == value)
        {
            if (value < lower_bound)
                flag[PositionIndex::START] = 1;
            if (value > upper_bound)
                flag[PositionIndex::START] = 1;
        }
        else
            flag[PositionIndex::START] = 1;
        ENDFOR
    }

    static void check_hidden_nan_finding_mode(VulkanQueue& queue, const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h, PositionIndex rows, const char* phase)
    {
        if (!nan_finding_mode_enabled())
            return;

        auto flag = make_neural_network_nan_scan_flag(queue);
        scan_hidden_matrix(queue, h, flag, rows, -NAN_FINDING_HIDDEN_ABS_BOUND, NAN_FINDING_HIDDEN_ABS_BOUND);
        if (neural_network_nan_scan_failed(queue, flag))
        {
            cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> cpu_h;
            cpu_h.set_rows(rows);
            h.copy_to_cpu(queue, cpu_h);

            float offending_value = std::nanf("");
            int offending_row = -1;
            int offending_col = -1;
            for (const auto r : enum_iterator1D<PositionIndex>(rows))
            {
                for (const auto c : enum_iterator1D<EmbeddingDimension>())
                {
                    const float v = cpu_h[r, c];
                    if (v != v || v < -NAN_FINDING_HIDDEN_ABS_BOUND || v > NAN_FINDING_HIDDEN_ABS_BOUND)
                    {
                        offending_value = v;
                        offending_row = static_cast<int>(static_cast<size_t>(r));
                        offending_col = static_cast<int>(static_cast<size_t>(c));
                        break;
                    }
                }
                if (offending_row >= 0)
                    break;
            }

            std::fprintf(
                stderr,
                "NAN_FINDING_MODE: TextTrainer hidden state invalid during %s; expected finite values in [%g, %g], "
                "actual value=%g at row=%d col=%d\n",
                phase,
                static_cast<double>(-NAN_FINDING_HIDDEN_ABS_BOUND),
                static_cast<double>(NAN_FINDING_HIDDEN_ABS_BOUND),
                static_cast<double>(offending_value),
                offending_row,
                offending_col
            );
            std::abort();
        }
    }

    static void check_hidden_vector_nan_finding_mode(VulkanQueue& queue, const fixed_size_vector<float, EmbeddingDimension>& values, const char* phase)
    {
        if (!nan_finding_mode_enabled())
            return;

        auto flag = make_neural_network_nan_scan_flag(queue);
        scan_embedding_vector(queue, values, flag, -NAN_FINDING_HIDDEN_ABS_BOUND, NAN_FINDING_HIDDEN_ABS_BOUND);
        if (neural_network_nan_scan_failed(queue, flag))
        {
            std::fprintf(stderr, "NAN_FINDING_MODE: TextTrainer hidden vector invalid during %s; expected finite values in [%g, %g]\n", phase, static_cast<double>(-NAN_FINDING_HIDDEN_ABS_BOUND), static_cast<double>(NAN_FINDING_HIDDEN_ABS_BOUND));
            std::abort();
        }
    }

    static void check_token_vector_nan_finding_mode(VulkanQueue& queue, const fixed_size_vector<float, TokenID>& values, float bound, const char* name, const char* phase)
    {
        if (!nan_finding_mode_enabled())
            return;

        auto flag = make_neural_network_nan_scan_flag(queue);
        scan_token_vector(queue, values, flag, -bound, bound);
        if (neural_network_nan_scan_failed(queue, flag))
        {
            std::fprintf(stderr, "NAN_FINDING_MODE: TextTrainer %s invalid during %s; expected finite values in [%g, %g]\n", name, phase, static_cast<double>(-bound), static_cast<double>(bound));
            std::abort();
        }
    }

    // Number of gradient-update passes over all layers per training example per epoch.
    static size_t training_steps_per_example(size_t num_layers, size_t learn_depth)
    {
        (void) num_layers;
        // Keep a stable update budget across model depth. Scaling linearly with
        // layer count made deeper models (e.g. 4 layers) overfit tiny corpora
        // and collapse to near one-hot predictions in just a few epochs.
        return learn_depth;
    }

    static void print_parallel_statistics_for_epoch(size_t epoch)
    {
        std::println("Parallel statistics after epoch {}:", epoch);
        parallel::statistics.print_statistics();
    }

    static size_t lines_per_epoch(size_t total_lines, std::optional<size_t> epoch_size)
    {
        assert(total_lines > 0);
        if (!epoch_size.has_value())
            return total_lines;
        return std::min(epoch_size.value(), total_lines);
    }

    // Layers

    void TextTrainer::propagate_forward()
    {
        assert(!m_transformer_blocks.empty());

        m_seq_len = m_last_input.size();
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_forward_workspace->reset(queue, m_seq_len);

        auto& ws = *m_forward_workspace;
        ws.h.set_rows(m_seq_len);

        // Embed tokens + sinusoidal positional encoding → h[T × D_MODEL]
        m_input_layer.propagate_forward(m_last_input, ws.h);
        check_hidden_nan_finding_mode(queue, ws.h, m_seq_len, "after input layer forward");

        // Pass through each transformer block in order
        for (size_t block_index = 0; block_index < m_transformer_blocks.size(); ++block_index)
        {
            auto& block = m_transformer_blocks[block_index];
            block.forward(ws.h, m_seq_len, ws.transformer_workspaces[block_index]);
            char phase[96];
            std::snprintf(phase, sizeof(phase), "after transformer block %zu forward", block_index);
            check_hidden_nan_finding_mode(queue, ws.h, m_seq_len, phase);
        }

        // Project the last-position hidden state to vocabulary logits.
        // Given a string of N tokens, the model learns to predict the N+1'th token,
        // so the final output is based on the hidden state at the last input position.
        const auto last_pos = dec(m_seq_len);
        copy_hidden_row_to_vector(ws.h, last_pos, ws.h_last);
        queue.wait("TextTrainer h_last ready for output layer forward");
        check_hidden_vector_nan_finding_mode(queue, ws.h_last, "after h_last copy");

        for (const auto ix : enum_iterator1D<MultiTokenPredictionIndex>())
        {
            m_output_layers[ix].forward_from_hidden(ws.h_last, queue);
            queue.wait("TextTrainer output layer forward");
        }
    }


    // TextTrainer

    static constexpr std::chrono::seconds MIN_TIME_BETWEEN_CHECKPOINTS{30};

    class ScopedDeviceBufferAllocationFreeze
    {
      public:
        ScopedDeviceBufferAllocationFreeze()
            : m_previous(rllm::vulkan_runtime::device_buffer_allocations_allowed())
        {
            rllm::vulkan_runtime::set_device_buffer_allocations_allowed(false);
        }

        ~ScopedDeviceBufferAllocationFreeze()
        {
            rllm::vulkan_runtime::set_device_buffer_allocations_allowed(m_previous);
        }

        ScopedDeviceBufferAllocationFreeze(const ScopedDeviceBufferAllocationFreeze&) = delete;
        ScopedDeviceBufferAllocationFreeze& operator=(const ScopedDeviceBufferAllocationFreeze&) = delete;

      private:
        bool m_previous = true;
    };

    static bool timed_checkpoint_due(const std::optional<std::chrono::seconds>& checkpointing_interval, std::chrono::steady_clock::time_point& last_checkpoint_at)
    {
        if (!checkpointing_interval.has_value())
            return false;

        const auto now = std::chrono::steady_clock::now();
        if ((now - last_checkpoint_at) < checkpointing_interval.value())
            return false;

        last_checkpoint_at = now;
        return true;
    }

    // Save a checkpoint, but skip it if less than MIN_TIME_BETWEEN_CHECKPOINTS
    // has elapsed since the last one (avoids rapid-fire saves at epoch boundaries).
    static void epoch_checkpoint(std::chrono::steady_clock::time_point& last_checkpoint_at, size_t epoch, const std::function<void()>& do_checkpoint)
    {
        const auto now = std::chrono::steady_clock::now();
        if ((now - last_checkpoint_at) < MIN_TIME_BETWEEN_CHECKPOINTS)
        {
            LOG_INFO("Skipping end-of-epoch checkpoint at epoch {} (too soon since last checkpoint)", epoch);
            return;
        }
        last_checkpoint_at = now;
        do_checkpoint();
    }

    // Returns top-K tokens selected by logit, with probabilities from the full
    // vocabulary softmax.
    std::vector<OutputToken> TextTrainer::get_best_output_token_ids(size_t top_k, MultiTokenPredictionIndex head) const
    {
        assert(!m_transformer_blocks.empty());
        assert(top_k > 0);

        // First, keep only top-K by raw logit (preserves correct rank order).
        std::vector<OutputToken> top_k_pairs = m_output_layers[head].get_top_k_by_logit(top_k);

        if (top_k_pairs.empty())
            return top_k_pairs;

        double best_logit = -std::numeric_limits<double>::infinity();
        for (const auto tok : enum_iterator1D<TokenID>())
            best_logit = std::max(best_logit, static_cast<double>(m_output_layers[head].m_inputs_cpu[tok]));

        // Stable softmax denominator over the full vocabulary. Normalizing only
        // over top-K makes untrained tied logits misleadingly display as 1/K.
        double sum_exp = 0.0;
        for (const auto tok : enum_iterator1D<TokenID>())
            sum_exp += std::exp(static_cast<double>(m_output_layers[head].m_inputs_cpu[tok]) - best_logit);

        for (auto& entry : top_k_pairs)
        {
            const double p = std::exp(static_cast<double>(entry.activation) - best_logit) / sum_exp;
            entry.activation = static_cast<float>(p);
        }

        return top_k_pairs;
    }

    cpu_fixed_vector<float, EmbeddingDimension> TextTrainer::get_last_hidden_mean(VulkanQueue& queue) const
    {
        const size_t seq_len = static_cast<size_t>(m_seq_len);
        if (seq_len == 0)
        {
            LOG_ERROR("seq-len == 0");
            std::abort();
        }

        cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> tmp;
        const_cast<flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>&>(m_last_hidden).copy_to_cpu(queue, tmp);

        cpu_fixed_vector<float, EmbeddingDimension> result;
        for (const auto d : enum_iterator1D<EmbeddingDimension>())
        {
            float sum = 0.0f;
            for (const auto pos : enum_iterator1D<PositionIndex>(m_seq_len))
                sum += static_cast<float>(tmp[pos, d]);
            result.push_back(static_cast<float>(sum / static_cast<float>(seq_len)));
        }
        return result;
    }


    static double elapsed_ms(const std::chrono::steady_clock::time_point& started_at)
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
    }

    void TextTrainer::initialize_training_progress_log()
    {
        m_training_progress_entries = std::make_unique<nlohmann::json>(nlohmann::json::array());
    }

    void TextTrainer::log_training_progress(
        const char* item_type, size_t epoch, size_t num_epochs,
        size_t range_start, size_t range_end, size_t total_items,
        size_t batch_size, size_t iterations, double batch_ms,
        const BatchTrainingTiming& timing)
    {
        const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        nlohmann::json entry{
            {"timestamp_ms", timestamp_ms},
            {"method", training_method_to_string(m_training_method)},
            {"item_type", item_type},
            {"epoch", epoch},
            {"epoch_percent", static_cast<double>(epoch) / static_cast<double>(num_epochs) * 100.0},
            {"range_start", range_start},
            {"range_end", range_end},
            {"total_items", total_items},
            {"progress_percent", static_cast<double>(range_end) / static_cast<double>(total_items) * 100.0},
            {"micro_batch_size", batch_size},
            {"training_loss", timing.average_primary_loss()},
            {"average_ms_per_item", batch_ms / static_cast<double>(batch_size)},
            {"iterations_total", iterations},
            {"average_iterations_per_item", static_cast<double>(iterations) / static_cast<double>(batch_size)},
            {"rounds", timing.rounds},
            {"batch_ms", batch_ms},
            {"forward_ms", timing.forward_ms},
            {"backward_ms", timing.backward_ms},
            {"apply_ms", timing.apply_ms},
            {"backward_output_ms", timing.backward_output_ms},
            {"backward_transformer_ms", timing.backward_transformer_ms},
            {"backward_input_ms", timing.backward_input_ms}
        };

        if (!m_training_progress_entries)
            initialize_training_progress_log();
        m_training_progress_entries->push_back(std::move(entry));
    }

    void TextTrainer::flush_training_progress_log() const
    {
        if (!m_training_progress_entries)
            return;
        std::ofstream file{"train.json", std::ios::trunc};
        file << m_training_progress_entries->dump(2) << '\n';
    }

    void TextTrainer::reset_workspaces()
    {
        m_forward_workspace = std::make_unique<TextTrainerForwardWorkspace>(PositionIndex::START, m_transformer_blocks.size());
        m_backward_workspace = std::make_unique<BackwardPropWorkspace>(PositionIndex::START);
        m_gradient_accumulation_workspace = std::make_unique<GradientAccumulationWorkspace>(m_transformer_blocks.size());
        m_gpu_packed_batch = std::make_unique<GpuPackedBatchInput>();
        m_batched_output_workspace = std::make_unique<BatchedOutputWorkspace>();
    }

    void TextTrainer::set_micro_batch_size(size_t n)
    {
        assert(n > 0 && n <= static_cast<size_t>(BatchIndex::MAX));
        m_micro_batch_size = n;
    }

    void TextTrainer::reset_gradient_accumulators()
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_gradient_accumulation_workspace->reset(queue);
        queue.wait("TextTrainer gradient accumulator reset");
    }

    void TextTrainer::apply_accumulated_gradients(float learning_rate_scale)
    {
        ++m_optimizer_step;
        if (!m_learning_rate_provider)
            m_learning_rate_provider = std::make_unique<ConstantLearningRate>(m_learning_rate);
        const float scheduled_rate = m_learning_rate_provider->get_rate();
        const float schedule_scale = scheduled_rate / m_learning_rate;
        // AdamW normalizes and updates every parameter tensor independently.
        // Dividing by the transformer-block count made the meaning of the CLI
        // rate depth-dependent and caused deeper models to learn too slowly.
        const float learning_rate = scheduled_rate * learning_rate_scale;
        const float peak_effective_rate = m_learning_rate * learning_rate_scale;
        const float log_delta = peak_effective_rate * 0.01f;
        const size_t warmup_steps = m_learning_rate_schedule_steps == 0
            ? 0
            : std::max<size_t>(1, m_learning_rate_schedule_steps / 20 +
                (m_learning_rate_schedule_steps % 20 != 0 ? 1 : 0));
        const bool phase_boundary = m_optimizer_step == warmup_steps ||
            m_optimizer_step == m_learning_rate_schedule_steps;
        if (!std::isfinite(m_last_logged_learning_rate) ||
            std::abs(learning_rate - m_last_logged_learning_rate) >= log_delta || phase_boundary)
        {
            const char* phase = "constant";
            if (m_learning_rate_schedule == LearningRateSchedule::Lowering)
                phase = m_optimizer_step <= warmup_steps ? "warmup" : "cosine-decay";
            else if (m_learning_rate_schedule == LearningRateSchedule::SimulatedAnnealing)
                phase = "simulated_annealing";
            LOG_INFO(
                "Learning rate changed: effective {:.10f}, base {:.10f}, schedule scale {:.6f}, phase {}, optimizer step {}/{}",
                learning_rate,
                m_learning_rate,
                schedule_scale,
                phase,
                m_optimizer_step,
                m_learning_rate_schedule_steps);
            m_last_logged_learning_rate = learning_rate;
        }
        const float bias_correction1 = 1.0f - std::pow(TransformerBlock::ADAM_BETA1, static_cast<float>(m_optimizer_step));
        const float bias_correction2 = 1.0f - std::pow(TransformerBlock::ADAM_BETA2, static_cast<float>(m_optimizer_step));

        // Treat embeddings, transformer blocks, and the LM heads as a sequence
        // of learned stages. A symmetric depth profile keeps the mean multiplier
        // at 1.0 while giving stages nearer the output a modestly higher rate.
        const size_t learned_stage_count = m_transformer_blocks.size() + 2;
        const float input_learning_rate = learning_rate *
            depth_learning_rate_multiplier(0, learned_stage_count, m_layer_learning_rate_multiplier);
        const float output_learning_rate = learning_rate *
            depth_learning_rate_multiplier(learned_stage_count - 1, learned_stage_count, m_layer_learning_rate_multiplier);

        const bool collect_diagnostics = m_optimizer_diagnostics_pending;
        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
        {
            if (collect_diagnostics)
                begin_optimizer_diagnostics();
            m_output_layers[oi].apply_accumulated_update(
                m_gradient_accumulation_workspace->output_layers[oi], output_learning_rate,
                bias_correction1, bias_correction2);
            if (collect_diagnostics)
                log_optimizer_diagnostics(std::format("output head {}", static_cast<size_t>(oi)));
        }

        for (size_t i = 0; i < m_transformer_blocks.size(); ++i)
        {
            const float layer_learning_rate = learning_rate *
                depth_learning_rate_multiplier(i + 1, learned_stage_count, m_layer_learning_rate_multiplier);
            if (collect_diagnostics)
                begin_optimizer_diagnostics();
            m_transformer_blocks[i].apply_accumulated_update(
                m_gradient_accumulation_workspace->transformer_blocks[i], layer_learning_rate,
                bias_correction1, bias_correction2);
            if (collect_diagnostics)
                log_optimizer_diagnostics(std::format("transformer layer {}", i));
        }

        m_input_layer.apply_accumulated_update(
            m_gradient_accumulation_workspace->embeddings, input_learning_rate,
            bias_correction1, bias_correction2, collect_diagnostics);
        m_optimizer_diagnostics_pending = false;
    }

    TextTrainer::TextTrainer(size_t num_layers, Corpus& corpus, Statistics& stats)
        : m_corpus(corpus)
        , m_stats(stats)
        , m_input_layer()
    {
        assert(static_cast<size_t>(TokenID::MAX) > 1);
        for (size_t i = 0; i < num_layers; ++i)
            m_transformer_blocks.emplace_back();

        reset_workspaces();
    }

    TextTrainer::~TextTrainer() = default;

    void TextTrainer::propagate_backward_mtp(const fixed_size_obj_vector<Score, MultiTokenPredictionIndex>& scores, MultiTokenPredictionIndex num_valid, TrainingStepTiming* timing)
    {
        assert(m_seq_len > PositionIndex::START);

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_backward_workspace->reset(queue, m_seq_len);

        auto& ws = *m_backward_workspace;
        ws.dh.set_rows(m_seq_len);
        ws.din.set_rows(m_seq_len);

        const auto last_pos = dec(m_seq_len);
        copy_hidden_row_to_vector(m_forward_workspace->h, last_pos, ws.h_last_vec);

        // Accumulate dh_last contributions from every valid output head.
        ws.dh_last.zero(queue);
        const auto output_started_at = std::chrono::steady_clock::now();
        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>(num_valid))
        {
            {
                ws.output_layer_delta.set_size(scores[oi].values.size());
                const auto bytes = static_cast<VkDeviceSize>(static_cast<size_t>(scores[oi].values.size()) * sizeof(float));
                if (bytes > 0)
                    ws.output_layer_delta.device_buffer().copy_from(queue, scores[oi].values.device_buffer(), bytes);
            }
            {
                const auto phase = std::format("output head {} backward delta", static_cast<size_t>(oi));
                check_token_vector_nan_finding_mode(queue, ws.output_layer_delta, 10.0f, "output delta", phase.c_str());
            }
            m_output_layers[oi].backward_accumulate(ws.output_layer_delta, ws.h_last_vec, ws.dh_last, m_gradient_accumulation_workspace->output_layers[oi]);
            {
                const auto phase = std::format("after output head {} backward", static_cast<size_t>(oi));
                check_hidden_vector_nan_finding_mode(queue, ws.dh_last, phase.c_str());
            }
        }
        if (timing)
            timing->backward_output_ms += elapsed_ms(output_started_at);

        // Propagate the summed gradient through the transformer blocks.
        ws.dh.zero(queue);
        ws.din.zero(queue);
        scatter_dh_last_to_row(ws.dh_last, ws.dh, last_pos);
        check_hidden_nan_finding_mode(queue, ws.dh, m_seq_len, "after dh_last scatter");

        auto* p_dh = &ws.dh;
        auto* p_din = &ws.din;
        const auto transformer_started_at = std::chrono::steady_clock::now();
        for (int i = static_cast<int>(m_transformer_blocks.size()) - 1; i >= 0; --i)
        {
            {
                const auto phase = std::format("before transformer block {} backward", i);
                check_hidden_nan_finding_mode(queue, *p_dh, m_seq_len, phase.c_str());
            }
            m_transformer_blocks[i].backward(*p_dh, *p_din, ws.transformer_block, m_forward_workspace->transformer_workspaces[static_cast<size_t>(i)]);
            m_transformer_blocks[i].accumulate_gradients(ws.transformer_block, m_gradient_accumulation_workspace->transformer_blocks[static_cast<size_t>(i)]);
            {
                const auto phase = std::format("after transformer block {} backward", i);
                check_hidden_nan_finding_mode(queue, *p_din, m_seq_len, phase.c_str());
            }
            std::swap(p_dh, p_din);
        }
        if (timing)
            timing->backward_transformer_ms += elapsed_ms(transformer_started_at);

        const auto input_started_at = std::chrono::steady_clock::now();
        check_hidden_nan_finding_mode(queue, *p_dh, m_seq_len, "before input layer backward");
        m_input_layer.accumulate_backward(m_last_input, *p_dh, m_gradient_accumulation_workspace->embeddings);
        if (timing)
            timing->backward_input_ms += elapsed_ms(input_started_at);
    }


    void TextTrainer::set_random_weights_and_connections()
    {
        LOG_INFO(
            "Initializers: weights={}, FFN={}, embeddings={}",
            initializer_name(m_weight_initializer),
            initializer_name(m_ffn_initializer),
            initializer_name(m_embedding_initializer));
        m_input_layer.set_random_embeddings(m_embedding_initializer);

        for (auto& block : m_transformer_blocks)
            block.randomize(m_weight_initializer, m_ffn_initializer);

        for (const auto output_index : enum_iterator1D<MultiTokenPredictionIndex>())
            m_output_layers[output_index].set_random_weights(m_weight_initializer);
    }


    void TextTrainer::dump_top_predictions()
    {
        int prediction_index = 0;
        const auto predicted_token_id_lists = get_best_output_token_ids(5, MultiTokenPredictionIndex::START);
        for (const auto& entry : predicted_token_id_lists)
        {
            const auto predicted_token = m_corpus.get_token_from_id(entry.token_id);
            LOG_INFO("\t prediction[{} of {}] / pred:'{}' (id: '{}'), {}", prediction_index, predicted_token_id_lists.size(), predicted_token, entry.token_id, entry.activation);
            prediction_index++;
        }
    }


    TextTrainer::EvaluationMetrics TextTrainer::evaluate(const std::vector<CpuInputLine>& evaluation_lines)
    {
        std::vector<WindowExample> evaluation_windows;
        evaluation_windows.reserve(evaluation_lines.size());
        for (const auto& line : evaluation_lines)
            evaluation_windows.push_back({
                .line = line,
                .context_length = static_cast<PositionIndex>(
                    mtp_input_len_for_sequence(static_cast<int>(line.size())))
            });
        return evaluate(evaluation_windows);
    }

    TextTrainer::EvaluationMetrics TextTrainer::evaluate(
        const std::vector<WindowExample>& evaluation_windows,
        bool report_worst_predictions)
    {
        if (evaluation_windows.empty())
        {
            LOG_ERROR("evaluation lines are empty!!");
            std::abort();
            return {
                .average_loss = std::numeric_limits<float>::quiet_NaN(),
                .perplexity = std::numeric_limits<double>::quiet_NaN(),
                .average_correct_token_probability = std::numeric_limits<double>::quiet_NaN()
            };
        }

        Score& score = m_evaluation_score;
        auto& queue = rllm::vulkan_runtime::get_queue(0);

        constexpr size_t MTP_HEAD_COUNT = static_cast<size_t>(MultiTokenPredictionIndex::MAX);
        std::array<double, MTP_HEAD_COUNT> per_head_loss_sum{};
        std::array<double, MTP_HEAD_COUNT> per_head_probability_sum{};
        std::array<size_t, MTP_HEAD_COUNT> per_head_count{};
        struct WorstPrediction
        {
            float loss;
            float expected_probability;
            float predicted_probability;
            MultiTokenPredictionIndex head;
            TokenID expected;
            TokenID predicted;
            std::string context;
        };
        std::vector<WorstPrediction> worst_predictions;
        constexpr size_t WORST_PREDICTIONS_TO_REPORT = 5;
        for (const auto& window : evaluation_windows)
        {
            const auto& example = window.line;
            const int seq_len = static_cast<int>(example.size());
            assert(seq_len >= 2);
            const int input_len = static_cast<int>(window.context_length);
            const auto num_valid_heads = mtp_valid_head_count_for_sequence(seq_len, input_len);

            example.sub_array(get_last_input(), static_cast<PositionIndex>(input_len));

            propagate_forward();

            for (const auto head : enum_iterator1D<MultiTokenPredictionIndex>(num_valid_heads))
            {
                const auto target = mtp_target_for_head(example, input_len, head);
                score.reset(queue);
                const float loss = m_output_layers[head].compute_score(score, target);
                if (!std::isfinite(loss) || loss < -1e-4f)
                {
                    std::println(
                        "Invalid evaluation loss detected for head {}, target '{}' ({}), input_len {}, seq_len {}, "
                        "loss {}.",
                        static_cast<size_t>(head),
                        m_corpus.get_token_from_id(target),
                        target,
                        input_len,
                        seq_len,
                        loss
                    );
                    std::abort();
                }
                const size_t head_index = static_cast<size_t>(head);
                per_head_loss_sum[head_index] += static_cast<double>(loss);
                per_head_probability_sum[head_index] += std::exp(-static_cast<double>(loss));
                ++per_head_count[head_index];
                if (report_worst_predictions)
                {
                    const auto top = m_output_layers[head].get_top_k_by_logit(1).front();
                    const float max_logit = score.temp_values_cpu[TempStorage::START];
                    const float sum_exp = score.temp_values_cpu[TempStorage::ONE];
                    const float predicted_probability = std::exp(top.activation - max_logit) / sum_exp;
                    auto context = m_corpus.get_line(get_last_input()).value_or("<unable to decode>");
                    worst_predictions.push_back({
                        .loss = loss,
                        .expected_probability = std::exp(-loss),
                        .predicted_probability = predicted_probability,
                        .head = head,
                        .expected = target,
                        .predicted = top.token_id,
                        .context = escape_whitespace_for_log(context)
                    });
                    std::ranges::sort(worst_predictions, {}, &WorstPrediction::loss);
                    if (worst_predictions.size() > WORST_PREDICTIONS_TO_REPORT)
                        worst_predictions.erase(worst_predictions.begin());
                }
            }
        }

        std::array<double, MTP_HEAD_COUNT> per_head_loss{};
        double mtp_loss_sum = 0.0;
        double mtp_probability_sum = 0.0;
        size_t mtp_prediction_count = 0;
        for (size_t head = 0; head < MTP_HEAD_COUNT; ++head)
        {
            per_head_loss[head] = per_head_count[head] == 0
                ? std::numeric_limits<double>::quiet_NaN()
                : per_head_loss_sum[head] / static_cast<double>(per_head_count[head]);
            mtp_loss_sum += per_head_loss_sum[head];
            mtp_probability_sum += per_head_probability_sum[head];
            mtp_prediction_count += per_head_count[head];
        }
        assert(per_head_count[0] == evaluation_windows.size());
        assert(mtp_prediction_count > 0);
        const double primary_count = static_cast<double>(per_head_count[0]);
        const double average_loss = per_head_loss[0];
        const double average_probability = per_head_probability_sum[0] / primary_count;
        const double mtp_average_loss = mtp_loss_sum / static_cast<double>(mtp_prediction_count);
        const double mtp_average_probability = mtp_probability_sum / static_cast<double>(mtp_prediction_count);
        if (report_worst_predictions)
        {
            for (size_t head = 0; head < MTP_HEAD_COUNT; ++head)
            {
                if (per_head_count[head] != 0)
                    LOG_INFO(
                        "validation MTP head {}: loss {:.6f} across {} predictions",
                        head, per_head_loss[head], per_head_count[head]);
            }
            std::ranges::sort(worst_predictions, std::greater{}, &WorstPrediction::loss);
            for (size_t rank = 0; rank < worst_predictions.size(); ++rank)
            {
                const auto& prediction = worst_predictions[rank];
                LOG_INFO(
                    "validation worst #{}: loss {:.6f}, expected '{}' ({:.6f}%), predicted '{}' ({:.3f}%), MTP head {}, context '{}'",
                    rank + 1,
                    prediction.loss,
                    escape_whitespace_for_log(m_corpus.get_token_from_id(prediction.expected)),
                    prediction.expected_probability * 100.0f,
                    escape_whitespace_for_log(m_corpus.get_token_from_id(prediction.predicted)),
                    prediction.predicted_probability * 100.0f,
                    static_cast<size_t>(prediction.head),
                    prediction.context);
            }
        }
        return {
            .average_loss = static_cast<float>(average_loss),
            .perplexity = std::exp(average_loss),
            .average_correct_token_probability = average_probability,
            .mtp_average_loss = mtp_average_loss,
            .mtp_average_correct_token_probability = mtp_average_probability,
            .per_head_loss = per_head_loss,
            .per_head_count = per_head_count
        };
    }

    void TextTrainer::train_with_increasingly_longer_sequences(const CpuInputLine& line_of_file, bool verbose, size_t max_iterations)
    {
        CpuInputLine line;
        for (const auto& line_substring_length : enum_iterator1D<PositionIndex>(line_of_file.size()))
        {
            line_of_file.sub_array(line, line_substring_length);
            if (line.empty())
                continue; // skip empty lines that can't be used for training

            const auto full_string_opt = m_corpus.get_line(line);
            assert(full_string_opt.has_value());
            const auto full_string = escape_whitespace_for_log(*full_string_opt);

            if (static_cast<int>(line.size()) < 2)
            {
                continue; // skip too-short lines that can't be used for training
            }

            LOG_INFO("Training on line[{}]: '{}'", (int) line_substring_length, full_string);

            do_training(line, verbose, max_iterations);
        }
    }

    void TextTrainer::train_with_up_to_N(const CpuInputLine& line_of_file, bool verbose, size_t max_iterations, int num_tokens)
    {
        assert(num_tokens >= 2);

        // If the line is too short for num_tokens, fall back to fewer tokens
        // (minimum 2 so there is always at least one input token and one target).
        if (static_cast<int>(line_of_file.size()) < num_tokens)
        {
            if (num_tokens > 2)
                train_with_up_to_N(line_of_file, verbose, max_iterations, num_tokens - 1);
            return;
        }

        CpuInputLine train_input;
        line_of_file.sub_array(train_input, static_cast<PositionIndex>(num_tokens));

        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());
        const auto full_string = escape_whitespace_for_log(*full_string_opt);

        LOG_INFO("Training on line[{}]: '{}'", num_tokens, full_string);

        do_training(train_input, verbose, max_iterations);
    }

    void TextTrainer::train_with_random_len_from_start(const CpuInputLine& line_of_file, bool verbose, size_t max_iterations, std::mt19937& rng, float learning_rate_scale, bool manage_accumulator)
    {
        const int line_len = static_cast<int>(line_of_file.size());
        if (line_len < 2) {
            return;
        }

        // Include short prefixes (len=2) so one-token contexts can learn their next token
        // (e.g. "# -> define/include").
        const int min_len = 2;
        std::uniform_int_distribution<int> len_dist(min_len, line_len);
        const int random_len = len_dist(rng);

        CpuInputLine train_input;
        line_of_file.sub_array(train_input, static_cast<PositionIndex>(random_len));
        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());

        LOG_INFO("Training on random line prefix ({} toks): '{}'", random_len, escape_whitespace_for_log(*full_string_opt));

        do_training(train_input, verbose, max_iterations, learning_rate_scale, manage_accumulator);
    }


    void TextTrainer::train_random_line_random_len_epoch(size_t epoch, const std::vector<CpuInputLine>& training_lines, bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval, std::chrono::steady_clock::time_point& last_checkpoint_at, std::mt19937& rng, std::optional<size_t> epoch_size)
    {
        batch_train(epoch, training_lines, verbose, num_epochs, checkpointing_interval, last_checkpoint_at, rng, epoch_size);
    }


    void TextTrainer::do_line_based_training(bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval, std::optional<size_t> epoch_size)
    {
        auto split = m_corpus.get_deterministic_training_split(VALIDATION_PERCENT);
        std::vector<CpuInputLine> training_lines = std::move(split.training_lines);
        std::vector<CpuInputLine> validation_lines = std::move(split.validation_lines);

        // With tiny corpora, a 20% split can leave only one held-out line.
        // That makes early stopping and best-checkpoint restoration extremely noisy
        // and can collapse the final model to whichever single line is validated.
        if (validation_lines.size() < 2)
        {
            training_lines.insert(training_lines.end(), std::make_move_iterator(validation_lines.begin()), std::make_move_iterator(validation_lines.end()));
            validation_lines.clear();
            LOG_INFO("Validation disabled for tiny corpus (fewer than 2 held-out lines)");
        }

        const auto total_lines = training_lines.size();
        const auto epoch_lines = lines_per_epoch(total_lines, epoch_size);

        LOG_INFO("Using {} training lines and {} validation lines ({}% target validation split); {} line(s) per epoch", training_lines.size(), validation_lines.size(), VALIDATION_PERCENT, epoch_lines);

        assert(!training_lines.empty());

        // Multi-epoch training with shuffling prevents catastrophic forgetting:
        // each example only gets 8*num_layers gradient updates per
        // pass, so no single example can overwrite all the others.
        std::mt19937 rng{42};
        auto last_checkpoint_at = std::chrono::steady_clock::now();
        float best_validation_loss = std::numeric_limits<float>::infinity();
        size_t epochs_without_improvement = 0;
        bool has_best_checkpoint = false;
        static constexpr const char* BEST_CHECKPOINT_FILENAME = "models/checkpoint-best.st";

        // Allocate both the device buffer and its pinned host transfer buffers
        // before batch training freezes steady-state allocations.
        begin_optimizer_diagnostics();
        log_optimizer_diagnostics("initialization");
        auto device_buffer_allocation_freeze = std::make_unique<ScopedDeviceBufferAllocationFreeze>();
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            m_optimizer_diagnostics_pending = true;
            m_backward_diagnostics_pending = true;
            bool should_stop = false;

            batch_train(epoch, training_lines, verbose, num_epochs, checkpointing_interval, last_checkpoint_at, rng, epoch_size);

            epoch_checkpoint(last_checkpoint_at, epoch, [&] {
                std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                checkpoint();
            });

            if (!validation_lines.empty())
            {
                const auto validation = evaluate(validation_lines);
                const float validation_loss = validation.average_loss;
                LOG_INFO(
                    "epoch {} validation: loss {:.6f}, perplexity {:.2f} (effective number of equally likely next-token choices; lower is better), average correct-token probability {:.3f}% across {} held-out lines",
                    epoch,
                    validation_loss,
                    validation.perplexity,
                    validation.average_correct_token_probability * 100.0,
                    validation_lines.size()
                );

                if ((validation_loss + VALIDATION_IMPROVEMENT_EPSILON) < best_validation_loss)
                {
                    best_validation_loss = validation_loss;
                    epochs_without_improvement = 0;
                    save(BEST_CHECKPOINT_FILENAME);
                    has_best_checkpoint = true;
                    LOG_INFO("Saved new best checkpoint '{}' with validation loss {:.6f}", BEST_CHECKPOINT_FILENAME, validation_loss);
                }
                else
                {
                    epochs_without_improvement++;
                    LOG_INFO("Validation loss did not improve for {} epoch(s); best remains {:.6f}", epochs_without_improvement, best_validation_loss);
                    if (m_early_stopping_enabled && epochs_without_improvement >= EARLY_STOPPING_PATIENCE)
                    {
                        LOG_INFO("early stopping at epoch {} after {} epoch(s) without validation improvement", epoch, epochs_without_improvement);
                        should_stop = true;
                    }
                }
            }

            print_parallel_statistics_for_epoch(epoch);
            if (m_learning_rate_provider)
                m_learning_rate_provider->advance_epoch();

            if (should_stop)
                break;
        }
        device_buffer_allocation_freeze.reset();

        if (has_best_checkpoint)
        {
            if (load(BEST_CHECKPOINT_FILENAME))
            {
                LOG_INFO("restored best checkpoint '{}' with validation loss {:.6f}", BEST_CHECKPOINT_FILENAME, best_validation_loss);
            }
            else
            {
                LOG_INFO("failed to restore best checkpoint '{}'", BEST_CHECKPOINT_FILENAME);
            }
        }
    }

    void TextTrainer::train_with_window(int window_size, bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval)
    {
        assert(window_size >= 2);

        auto split = m_corpus.get_deterministic_training_split(VALIDATION_PERCENT);
        if (split.validation_lines.size() < 2)
        {
            split.training_lines.insert(
                split.training_lines.end(),
                std::make_move_iterator(split.validation_lines.begin()),
                std::make_move_iterator(split.validation_lines.end()));
            split.validation_lines.clear();
            LOG_INFO("Window validation disabled for tiny corpus (fewer than 2 held-out lines)");
        }

        const bool reverse = m_training_method == TrainingMethod::REVERSE_WINDOW;
        auto training_windows = make_line_windows(
            split.training_lines, static_cast<size_t>(window_size), m_window_stride, reverse);
        auto validation_windows = make_line_windows(
            split.validation_lines, static_cast<size_t>(window_size), m_window_stride, reverse);

        if (validation_windows.size() < 2)
        {
            split.training_lines.insert(
                split.training_lines.end(),
                std::make_move_iterator(split.validation_lines.begin()),
                std::make_move_iterator(split.validation_lines.end()));
            split.validation_lines.clear();
            training_windows = make_line_windows(
                split.training_lines, static_cast<size_t>(window_size), m_window_stride, reverse);
            validation_windows.clear();
            LOG_INFO("Window validation disabled for tiny corpus (fewer than 2 held-out windows)");
        }

        if (training_windows.empty())
            return;

        LOG_INFO(
            "Window split: {} training lines producing {} boundary-preserving windows, {} validation lines producing {} comparable windows; validation interval 5 minutes",
            split.training_lines.size(),
            training_windows.size(),
            split.validation_lines.size(),
            validation_windows.size());

        const size_t steps_per_window = training_steps_per_example(m_transformer_blocks.size(), m_learn_depth);
        const size_t windows_per_epoch = training_windows.size();
        m_learning_rate_schedule_steps = 0;
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            const size_t batches = windows_per_epoch / m_micro_batch_size +
                (windows_per_epoch % m_micro_batch_size != 0 ? 1 : 0);
            m_learning_rate_schedule_steps += batches * steps_per_window;
        }
        if (m_learning_rate_schedule == LearningRateSchedule::Constant)
        {
            m_learning_rate_provider = std::make_unique<ConstantLearningRate>(m_learning_rate);
            LOG_INFO("Window training: maximum size {}, per-line stride {}; constant learning rate {:.8f}",
                window_size, m_window_stride, m_learning_rate);
        }
        else if (m_learning_rate_schedule == LearningRateSchedule::Lowering)
        {
            m_learning_rate_provider = std::make_unique<LoweringLearningRate>(m_learning_rate, m_learning_rate_schedule_steps);
            LOG_INFO(
                "Window training: maximum size {}, per-line stride {}; learning-rate schedule: {} planned optimizer steps, {:.1f}% warmup, cosine decay from {:.8f} to {:.8f}",
                window_size, m_window_stride, m_learning_rate_schedule_steps,
                LoweringLearningRate::WARMUP_FRACTION * 100.0f, m_learning_rate,
                m_learning_rate * LoweringLearningRate::MIN_SCALE);
        }
        else
        {
            m_learning_rate_provider = std::make_unique<SimulatedAnnealingLearningRate>(
                m_learning_rate, m_simulated_annealing_decay_factor,
                m_simulated_annealing_initial_multiplier, m_simulated_annealing_decay_epochs,
                m_simulated_annealing_min_multiplier);
            LOG_INFO(
                "Window training: maximum size {}, per-line stride {}; simulated_annealing learning rate multiplies by {:.4f} every {} epochs from {:.8f} to a floor of {:.8f}",
                window_size, m_window_stride,
                m_simulated_annealing_decay_factor,
                m_simulated_annealing_decay_epochs,
                std::max(
                    m_learning_rate * m_simulated_annealing_initial_multiplier,
                    m_learning_rate * m_simulated_annealing_min_multiplier),
                m_learning_rate * m_simulated_annealing_min_multiplier);
        }
        m_last_logged_learning_rate = std::numeric_limits<float>::quiet_NaN();
        std::mt19937 rng{42};
        size_t total_windows_trained = 0;
        auto last_checkpoint_at = std::chrono::steady_clock::now();
        auto next_validation_at = std::chrono::steady_clock::now() + std::chrono::minutes{5};
        float best_validation_loss = std::numeric_limits<float>::infinity();
        size_t epochs_without_improvement = 0;
        bool has_best_checkpoint = false;
        const char* best_checkpoint_filename = m_training_method == TrainingMethod::REVERSE_WINDOW ?
            "models/checkpoint-best-reverse-window.st" : "models/checkpoint-best-window.st";
        if (!validation_windows.empty())
        {
            const auto baseline = evaluate(validation_windows, true);
            best_validation_loss = baseline.average_loss;
            save(best_checkpoint_filename);
            has_best_checkpoint = true;
            LOG_INFO(
                "window validation baseline before training: head-0 loss {:.6f}, head-0 perplexity {:.2f}, head-0 average correct-token probability {:.3f}%, all-MTP loss {:.6f} across {} held-out windows",
                baseline.average_loss,
                baseline.perplexity,
                baseline.average_correct_token_probability * 100.0,
                baseline.mtp_average_loss,
                validation_windows.size());
        }
        // Optimizer update kernels always bind this buffer, even on updates for
        // which diagnostics collection is disabled. Allocate it and both pinned
        // transfer buffers before the window-training allocation freeze.
        begin_optimizer_diagnostics();
        log_optimizer_diagnostics("initialization");
        auto device_buffer_allocation_freeze = std::make_unique<ScopedDeviceBufferAllocationFreeze>();
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            m_optimizer_diagnostics_pending = true;
            m_backward_diagnostics_pending = true;
            bool should_stop = false;
            const auto epoch_started_at = std::chrono::steady_clock::now();
            LOG_INFO("Epoch[{}%]: {:0.2f}% done", epoch / static_cast<float>(num_epochs) * 100.0f, 0.0f);

            const size_t num_windows = training_windows.size();
            std::vector<size_t> indices(num_windows);
            std::iota(indices.begin(), indices.end(), 0);
            if (m_training_method == TrainingMethod::WINDOW)
                std::shuffle(indices.begin(), indices.end(), rng);

            double epoch_training_loss_sum = 0.0;
            size_t epoch_training_loss_count = 0;

            for (size_t start = 0; start < num_windows; start += m_micro_batch_size)
            {
                const size_t count = std::min(m_micro_batch_size, num_windows - start);
                std::vector<BatchTrainingItem> batch;
                batch.reserve(count);
                for (size_t offset = 0; offset < count; ++offset)
                {
                    const auto& window = training_windows[indices[start + offset]];
                    batch.push_back({window.line, false, window.context_length});
                }
                total_windows_trained += batch.size();

                if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                {
                    LOG_INFO("creating timed checkpoint at epoch {}, window {}, total windows trained {}", epoch, start, total_windows_trained);
                    checkpoint();
                }

                const auto started = std::chrono::steady_clock::now();
                BatchTrainingTiming timing;
                const size_t iterations = train_batch_items(batch, steps_per_window, timing);
                epoch_training_loss_sum += timing.primary_loss_sum;
                epoch_training_loss_count += timing.primary_loss_count;
                const double total_ms = elapsed_ms(started);
                const size_t completed_windows = start + count;
                LOG_INFO(
                    "Epoch[{}%] window[{}..{}]: {:0.2f}% done (micro-batch {}, training loss {:.6f}, avg {:.2f} ms/window, iterations total {}, avg {:.2f}/window, rounds {}, batch {:.2f} ms)",
                    epoch / static_cast<float>(num_epochs) * 100.0f,
                    start + 1,
                    completed_windows,
                    static_cast<float>(completed_windows) / static_cast<float>(num_windows) * 100.0f,
                    batch.size(),
                    timing.average_primary_loss(),
                    total_ms / batch.size(),
                    iterations,
                    static_cast<double>(iterations) / batch.size(),
                    timing.rounds,
                    total_ms);
                log_training_progress(
                    "window", epoch, num_epochs, start + 1, completed_windows,
                    num_windows, batch.size(), iterations, total_ms, timing);

                const size_t batch_number = start / m_micro_batch_size + 1;
                if (batch_number % 10 == 0)
                {
                    const double elapsed_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - epoch_started_at).count();
                    const size_t completed = start + count;
                    const double remaining_seconds = elapsed_seconds *
                        static_cast<double>(num_windows - completed) / static_cast<double>(completed);
                    const double epoch_fraction = static_cast<double>(completed) / static_cast<double>(num_windows);
                    const double estimated_epoch_seconds = elapsed_seconds / epoch_fraction;
                    const double all_epochs_remaining_seconds = estimated_epoch_seconds *
                        (static_cast<double>(num_epochs - epoch) - epoch_fraction);
                    LOG_INFO(
                        "Epoch {} batch {} ETA until epoch finishes: {}; ETA until all planned epochs finish: {}",
                        epoch, batch_number, format_eta_for_log(remaining_seconds),
                        format_eta_for_log(all_epochs_remaining_seconds));
                }

                if (!validation_windows.empty() && std::chrono::steady_clock::now() >= next_validation_at)
                {
                    const auto validation_started = std::chrono::steady_clock::now();
                    const auto validation = evaluate(validation_windows);
                    LOG_INFO(
                        "window validation at epoch {}, {:.2f}%: head-0 loss {:.6f}, head-0 perplexity {:.2f}, head-0 average correct-token probability {:.3f}%, all-MTP loss {:.6f} across {} held-out windows (evaluation {:.2f} s)",
                        epoch,
                        static_cast<float>(start + count) / static_cast<float>(num_windows) * 100.0f,
                        validation.average_loss,
                        validation.perplexity,
                        validation.average_correct_token_probability * 100.0,
                        validation.mtp_average_loss,
                        validation_windows.size(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - validation_started).count());
                    next_validation_at = std::chrono::steady_clock::now() + std::chrono::minutes{5};
                }
            }

            const double epoch_training_loss = epoch_training_loss_count == 0
                ? std::numeric_limits<double>::quiet_NaN()
                : epoch_training_loss_sum / static_cast<double>(epoch_training_loss_count);
            LOG_INFO(
                "window training at epoch {} end: loss {:.6f} across {} optimizer examples",
                epoch, epoch_training_loss, epoch_training_loss_count);

            if (!validation_windows.empty())
            {
                const auto validation_started = std::chrono::steady_clock::now();
                const auto validation = evaluate(validation_windows, true);
                LOG_INFO(
                    "window validation at epoch {} end: head-0 loss {:.6f}, head-0 perplexity {:.2f}, head-0 average correct-token probability {:.3f}%, all-MTP loss {:.6f} across {} held-out windows (evaluation {:.2f} s)",
                    epoch,
                    validation.average_loss,
                    validation.perplexity,
                    validation.average_correct_token_probability * 100.0,
                    validation.mtp_average_loss,
                    validation_windows.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - validation_started).count());
                next_validation_at = std::chrono::steady_clock::now() + std::chrono::minutes{5};

                if ((validation.average_loss + VALIDATION_IMPROVEMENT_EPSILON) < best_validation_loss)
                {
                    best_validation_loss = validation.average_loss;
                    epochs_without_improvement = 0;
                    save(best_checkpoint_filename);
                    has_best_checkpoint = true;
                    LOG_INFO(
                        "Saved new best window checkpoint '{}' with head-0 validation loss {:.6f}",
                        best_checkpoint_filename, best_validation_loss);
                }
                else
                {
                    ++epochs_without_improvement;
                    LOG_INFO(
                        "Window head-0 validation loss did not improve for {} epoch(s); best remains {:.6f}",
                        epochs_without_improvement, best_validation_loss);
                    if (m_early_stopping_enabled && epochs_without_improvement >= EARLY_STOPPING_PATIENCE)
                    {
                        LOG_INFO(
                            "early stopping window training at epoch {} after {} epoch(s) without validation improvement",
                            epoch, epochs_without_improvement);
                        should_stop = true;
                    }
                }
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            checkpoint();
            print_parallel_statistics_for_epoch(epoch);
            if (m_learning_rate_provider)
                m_learning_rate_provider->advance_epoch();
            if (should_stop)
                break;
        }

        // Loading recreates transformer blocks and their workspaces, so GPU buffer
        // allocation must be enabled again before restoring the best checkpoint.
        device_buffer_allocation_freeze.reset();
        if (has_best_checkpoint)
        {
            if (load(best_checkpoint_filename))
            {
                LOG_INFO(
                    "restored best window checkpoint '{}' with head-0 validation loss {:.6f}",
                    best_checkpoint_filename, best_validation_loss);
            }
            else
            {
                LOG_INFO("failed to restore best window checkpoint '{}'", best_checkpoint_filename);
            }
        }
    }

    void TextTrainer::do_whole_corpus_window_based_training(bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval)
    {
        // Window methods use independent per-line sequences so no example crosses a boundary.
        assert(m_training_method == TrainingMethod::WINDOW || m_training_method == TrainingMethod::REVERSE_WINDOW);
        train_with_window(m_window_size, verbose, num_epochs, checkpointing_interval);
    }


    void TextTrainer::train(bool verbose, size_t num_epochs, const std::optional<std::string>& input_filename, const std::optional<std::chrono::seconds>& checkpointing_interval, std::optional<size_t> epoch_size)
    {
        initialize_training_progress_log();
        Statistics::TotalLearnRecorderScope total_learn_recorder_scope(m_stats);

        if (input_filename)
        {
            if (!load(*input_filename))
            {
                std::println("Failed to load model from '{}'", *input_filename);
                std::exit(1);
            }
            LOG_INFO("Loaded model from '{}'", *input_filename);
        }
        else
        {
            LOG_INFO("No input model specified, starting with random weights.");
            set_random_weights_and_connections();
        }

        const size_t vocab = static_cast<size_t>(TokenID::MAX);
        const size_t corpus_size = m_corpus.count_num_lines();
        const size_t d_model = static_cast<size_t>(EmbeddingDimension::MAX);
        const size_t d_ff = static_cast<size_t>(FFDimension::MAX);
        const size_t n_layers = m_transformer_blocks.size();
        const size_t params_embed = vocab * d_model; // token embeddings
        const size_t params_lm_head = static_cast<size_t>(MultiTokenPredictionIndex::MAX) * vocab * d_model;
        const size_t params_attn = 4 * d_model * d_model; // W_q, W_k, W_v, W_o per block
        const size_t params_ffn = 3 * d_model * d_ff; // W_gate, W_up, W_down per block (2*d_ff*d_model + d_model*d_ff)
        const size_t params_per_block = params_attn + params_ffn;
        const size_t total_params = params_embed + params_lm_head + n_layers * params_per_block;
        const float total_params_mib = static_cast<float>(total_params * sizeof(float)) / (1024.0f * 1024.0f);
        size_t transformer_gpu_bytes = 0;
        if (n_layers > 0)
        {
            const size_t per_layer_gpu_bytes = m_transformer_blocks.front().memory_usage_bytes(
                m_forward_workspace->transformer_workspaces.front(),
                m_backward_workspace->transformer_block,
                m_gradient_accumulation_workspace->transformer_blocks.front(),
                n_layers);
            transformer_gpu_bytes = per_layer_gpu_bytes * n_layers;
            LOG_INFO(
                "Estimated transformer GPU memory: {:.2f} MB per-layer share x {} layers = {:.2f} MB "
                "(weights, optimizer state, forward/backward workspaces, and gradient accumulators)",
                static_cast<double>(per_layer_gpu_bytes) / 1'000'000.0,
                n_layers,
                static_cast<double>(transformer_gpu_bytes) / 1'000'000.0);
        }

        const size_t batch_slot_gpu_bytes =
            2 * static_cast<size_t>(EmbeddingDimension::MAX) * sizeof(float) + // h_last, dh_last
            2 * vocab * sizeof(float) +                                       // logits, delta
            static_cast<size_t>(TempStorage::MAX) * sizeof(float) +           // softmax_temp
            2 * sizeof(int) + sizeof(float) +                                 // expected token, active flag, loss
            3 * sizeof(int);                                                   // packed row begin, end, last
        const size_t batch_gpu_bytes = batch_slot_gpu_bytes * static_cast<size_t>(BatchIndex::MAX);
        LOG_INFO(
            "Estimated batch-slot GPU memory: {:.2f} KB per slot x {} slots = {:.2f} MB "
            "(batched output workspace and per-example packed-row metadata)",
            static_cast<double>(batch_slot_gpu_bytes) / 1'000.0,
            static_cast<size_t>(BatchIndex::MAX),
            static_cast<double>(batch_gpu_bytes) / 1'000'000.0);

        static constexpr size_t CORPUS_SIZE_WARNING_THRESHOLD = 100000;
        if (corpus_size > CORPUS_SIZE_WARNING_THRESHOLD)
        {
            LOG_INFO(
                "WARNING: corpus is large ({} lines > {} threshold). Training may take significantly longer; "
                "consider narrower --filter values, fewer --epochs, or window training.",
                corpus_size,
                CORPUS_SIZE_WARNING_THRESHOLD
            );
        }

        LOG_INFO(
            "Training the neural network...\n"
            "\t $num_layers: {}\n"
            "\t $corpus_size: {}\n"
            "\t $total_params: {} ({:.2f}M, {:.2f} MiB)  [embed:{} lm_head:{} blocks:{}x{}]\n"
            "\t convergence threshold: {:.6f}\n"
            "\t probability of correct token:  {:.6f}\n"
            "\t steps per example per epoch: {}\n"
            "\t micro-batch size: {}\n"
            "\t num epochs: {}\n"
            "\t epoch size: {}\n"
            "\t training method: {}\n",
            n_layers,
            corpus_size,
            total_params,
            static_cast<float>(total_params) / 1e6f,
            total_params_mib,
            params_embed,
            params_lm_head,
            n_layers,
            params_per_block,
            CONVERGENCE_THRESHOLD,
            PROBABILITY_OF_CORRECT_TOKEN,
            training_steps_per_example(n_layers, m_learn_depth),
            m_micro_batch_size,
            num_epochs,
            epoch_size.has_value() ? std::format("{} line(s)", epoch_size.value()) : std::string{"all training lines"},
            training_method_to_string(m_training_method)
        );
        LOG_INFO(
            "Depth-adaptive learning rates: embeddings {:.3f}x, transformer layers linearly increasing by depth, output heads {:.3f}x (mean 1.000x)",
            2.0f - m_layer_learning_rate_multiplier,
            m_layer_learning_rate_multiplier);

        if (m_learning_rate_schedule == LearningRateSchedule::Constant)
            m_learning_rate_provider = std::make_unique<ConstantLearningRate>(m_learning_rate);
        else if (m_learning_rate_schedule == LearningRateSchedule::SimulatedAnnealing)
            m_learning_rate_provider = std::make_unique<SimulatedAnnealingLearningRate>(
                m_learning_rate, m_simulated_annealing_decay_factor,
                m_simulated_annealing_initial_multiplier, m_simulated_annealing_decay_epochs,
                m_simulated_annealing_min_multiplier);

        if (training_method_is_line_based())
        {
            do_line_based_training(verbose, num_epochs, checkpointing_interval, epoch_size);
        }
        else
        {
            do_whole_corpus_window_based_training(verbose, num_epochs, checkpointing_interval);
        }
    }


    TrainingStepOutcome TextTrainer::do_training_step(const CpuInputLine& train_output, bool verbose, size_t iteration_index, float learning_rate_scale, bool manage_accumulator, TrainingStepTiming* timing)
    {
        // Multi-token prediction (MTP): given a sequence [t0,t1,...,tN-1] we
        // use the first max(1, N - max_heads) tokens as context and train each
        // real future head k to predict the token at position (context_len + k).
        assert(static_cast<int>(train_output.size()) >= 2);
        const int _seq_len = static_cast<int>(train_output.size());
        const int _input_len = mtp_input_len_for_sequence(_seq_len);
        CpuInputLine train_input;
        train_output.sub_array(train_input, static_cast<PositionIndex>(_input_len));
        const auto num_valid_heads = mtp_valid_head_count_for_sequence(_seq_len, _input_len);
        const auto expected_output_token = mtp_target_for_head(train_output, _input_len, MultiTokenPredictionIndex::START);

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto full_string = escape_whitespace_for_log(*full_string_opt);

        get_last_input() = train_input;

        if (manage_accumulator)
            reset_gradient_accumulators();

        const auto forward_started_at = std::chrono::steady_clock::now();
        propagate_forward();
        if (timing)
            timing->forward_ms += elapsed_ms(forward_started_at);

        m_training_scores.set_size(num_valid_heads);
        float loss = 0.0f;
        for (const auto _k : enum_iterator1D<MultiTokenPredictionIndex>(num_valid_heads))
        {
            Score& s = m_training_scores[_k];
            const auto _target = mtp_target_for_head(train_output, _input_len, _k);
            const float _k_loss = m_output_layers[_k].compute_score(s, _target);
            if (std::isnan(_k_loss))
            {
                LOG_ERROR("loss became NaN!");
            }
            if (!std::isfinite(_k_loss))
            {
                LOG_INFO("Non-finite loss detected for head {}, target '{}' ({}), full string: '{}', input size: {}.", static_cast<size_t>(_k), m_corpus.get_token_from_id(_target), _target, full_string, static_cast<size_t>(train_input.size()));
                m_stats.record_learning_failure();
                return TrainingStepOutcome::Failed;
            }
            if (_k == MultiTokenPredictionIndex::START)
                loss = _k_loss;
        }

        const auto backward_started_at = std::chrono::steady_clock::now();
        propagate_backward_mtp(m_training_scores, num_valid_heads, timing);
        if (timing)
            timing->backward_ms += elapsed_ms(backward_started_at);
        if (manage_accumulator)
        {
            const auto apply_started_at = std::chrono::steady_clock::now();
            apply_accumulated_gradients(learning_rate_scale);
            if (timing)
                timing->apply_ms += elapsed_ms(apply_started_at);
        }

        if (loss < CONVERGENCE_THRESHOLD)
        {
            LOG_INFO("Convergence reached after {} steps for expected '{}', full string: '{}', input size: {}", iteration_index + 1, escape_whitespace_for_log(m_corpus.get_token_from_id(expected_output_token)), full_string, static_cast<size_t>(train_input.size()));
            m_stats.record_learning_success();
            return TrainingStepOutcome::Converged;
        }

        return TrainingStepOutcome::Continue;
    }

    size_t TextTrainer::do_training(const CpuInputLine& train_output, bool verbose, size_t max_iterations, float learning_rate_scale, bool manage_accumulator)
    {
        const int _seq_len = static_cast<int>(train_output.size());
        const int _input_len = mtp_input_len_for_sequence(_seq_len);
        const auto expected_output_token = mtp_target_for_head(train_output, _input_len, MultiTokenPredictionIndex::START);
        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto full_string = escape_whitespace_for_log(*full_string_opt);

        bool print_loss_verbose = false;
        static int count = 0;
        if (count++ % 100 == 0)
        {
            print_loss_verbose = true;
        }

        size_t iterations = 0;
        for (size_t i = 0; i < max_iterations; ++i)
        {
            const auto outcome = do_training_step(train_output, verbose, i, learning_rate_scale, manage_accumulator);
            ++iterations;
            if (outcome == TrainingStepOutcome::Converged || outcome == TrainingStepOutcome::Failed)
                return iterations;

            if (print_loss_verbose)
            {
                LOG_INFO("Training iteration[{}], wanted: '{}' ({}), full string: '{}'", i, escape_whitespace_for_log(m_corpus.get_token_from_id(expected_output_token)), expected_output_token, full_string);
                LOG_INFO("  Loss: {:.6f}", 0.0f);
                dump_top_predictions();
            }
        }

        if (manage_accumulator || max_iterations != 1)
        {
            LOG_INFO("Steps exhausted ({}) for this line. threshold = {:.6f}, expected token: '{}' ({}), full string: '{}', input size: {}.", max_iterations, CONVERGENCE_THRESHOLD, escape_whitespace_for_log(m_corpus.get_token_from_id(expected_output_token)), expected_output_token, full_string, static_cast<size_t>(_input_len));
            m_stats.record_learning_failure();
        }

        return max_iterations;
    }


} // namespace rllm
