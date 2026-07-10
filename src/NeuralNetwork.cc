#include <NeuralNetwork.hpp>
#include <RuntimeConfig.hpp>
#include <TokenIDFormatter.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <enum_iterator2D.hpp>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <parallel.hpp>
#include <random>
#include <rllm_vulkan_runtime.hpp>
#include <set>
#include <vecmath.hpp>

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
                "NAN_FINDING_MODE: NeuralNetwork hidden state invalid during %s; expected finite values in [%g, %g], "
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
            std::fprintf(stderr, "NAN_FINDING_MODE: NeuralNetwork hidden vector invalid during %s; expected finite values in [%g, %g]\n", phase, static_cast<double>(-NAN_FINDING_HIDDEN_ABS_BOUND), static_cast<double>(NAN_FINDING_HIDDEN_ABS_BOUND));
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
            std::fprintf(stderr, "NAN_FINDING_MODE: NeuralNetwork %s invalid during %s; expected finite values in [%g, %g]\n", name, phase, static_cast<double>(-bound), static_cast<double>(bound));
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

    struct NeuralNetworkForwardWorkspace
    {
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> h;
        fixed_size_vector<float, EmbeddingDimension> h_last;

        std::vector<ForwardWorkspace> transformer_workspaces;

        NeuralNetworkForwardWorkspace(PositionIndex seq_len, size_t num_transformer_blocks)
            : h(seq_len)
            , transformer_workspaces()
        {
            transformer_workspaces.reserve(num_transformer_blocks);
            for (size_t i = 0; i < num_transformer_blocks; ++i)
                transformer_workspaces.emplace_back(seq_len);
            h_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(VulkanQueue& queue, PositionIndex seq_len)
        {
            h.set_rows(seq_len);
            for (auto& workspace : transformer_workspaces)
                workspace.reset(queue, seq_len);
        }
    };

    void NeuralNetwork::propagate_forward()
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
        queue.wait("NeuralNetwork h_last ready for output layer forward");
        check_hidden_vector_nan_finding_mode(queue, ws.h_last, "after h_last copy");

        for (const auto ix : enum_iterator1D<MultiTokenPredictionIndex>())
        {
            m_output_layers[ix].forward_from_hidden(ws.h_last, queue);
            queue.wait("NeuralNetwork output layer forward");
        }
    }


    // NeuralNetwork

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
    std::vector<OutputToken> NeuralNetwork::get_best_output_token_ids(size_t top_k, MultiTokenPredictionIndex head) const
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

    cpu_fixed_vector<float, EmbeddingDimension> NeuralNetwork::get_last_hidden_mean(VulkanQueue& queue) const
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


    struct BackwardPropWorkspace
    {
        fixed_size_vector<float, TokenID> output_layer_delta;
        fixed_size_vector<float, EmbeddingDimension> h_last_vec;
        fixed_size_vector<float, EmbeddingDimension> dh_last;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> dh;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> din;
        BackwardWorkspace transformer_block;

        explicit BackwardPropWorkspace(PositionIndex seq_len)
            : dh(seq_len)
            , din(seq_len)
            , transformer_block(seq_len)
        {
            output_layer_delta.set_size(TokenID::MAX);
            h_last_vec.set_size(EmbeddingDimension::MAX);
            dh_last.set_size(EmbeddingDimension::MAX);
        }

        void reset(VulkanQueue& queue, PositionIndex seq_len)
        {
            dh.set_rows(seq_len);
            din.set_rows(seq_len);
            transformer_block.reset(queue, seq_len);
            output_layer_delta.zero(queue);
            h_last_vec.zero(queue);
            dh_last.zero(queue);
        }
    };

    struct GradientAccumulationWorkspace
    {
        fixed_size_obj_vector<OutputLayerGradientAccumulator, MultiTokenPredictionIndex> output_layers;
        std::vector<TransformerGradientAccumulator> transformer_blocks;
        EmbeddingGradientAccumulator embeddings;

        explicit GradientAccumulationWorkspace(size_t num_transformer_blocks)
            : transformer_blocks(num_transformer_blocks)
        {
            output_layers.set_size(MultiTokenPredictionIndex::MAX);
        }

        void reset(VulkanQueue& queue)
        {
            for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
                output_layers[oi].reset(queue);
            for (auto& block : transformer_blocks)
                block.reset(queue);
            embeddings.reset();
        }
    };

    void NeuralNetwork::reset_workspaces()
    {
        m_forward_workspace = std::make_unique<NeuralNetworkForwardWorkspace>(PositionIndex::START, m_transformer_blocks.size());
        m_backward_workspace = std::make_unique<BackwardPropWorkspace>(PositionIndex::START);
        m_gradient_accumulation_workspace = std::make_unique<GradientAccumulationWorkspace>(m_transformer_blocks.size());
    }

    void NeuralNetwork::set_micro_batch_size(size_t n)
    {
        assert(n > 0);
        m_micro_batch_size = n;
    }

    void NeuralNetwork::reset_gradient_accumulators()
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_gradient_accumulation_workspace->reset(queue);
        queue.wait("NeuralNetwork gradient accumulator reset");
    }

    void NeuralNetwork::apply_accumulated_gradients(float learning_rate_scale)
    {
        const float learning_rate = (m_learning_rate * learning_rate_scale) /
            static_cast<float>(std::max<size_t>(1, m_transformer_blocks.size()));

        for (const auto oi : enum_iterator1D<MultiTokenPredictionIndex>())
            m_output_layers[oi].apply_accumulated_update(m_gradient_accumulation_workspace->output_layers[oi], learning_rate);

        for (size_t i = 0; i < m_transformer_blocks.size(); ++i)
            m_transformer_blocks[i].apply_accumulated_update(m_gradient_accumulation_workspace->transformer_blocks[i], learning_rate);

        m_input_layer.apply_accumulated_update(m_gradient_accumulation_workspace->embeddings, learning_rate);
    }

    NeuralNetwork::NeuralNetwork(size_t num_layers, Corpus& corpus, Statistics& stats)
        : m_corpus(corpus)
        , m_stats(stats)
        , m_input_layer()
        , m_fires_nothing_ce_loss(std::log(static_cast<float>(TokenID::MAX)))
        , m_convergence_threshold(m_fires_nothing_ce_loss / k_convergence_divisor)
    {
        assert(static_cast<size_t>(TokenID::MAX) > 1);
        for (size_t i = 0; i < num_layers; ++i)
            m_transformer_blocks.emplace_back();

        reset_workspaces();
    }

    NeuralNetwork::~NeuralNetwork() = default;

    void NeuralNetwork::propagate_backward_mtp(
        const fixed_size_obj_vector<Score, MultiTokenPredictionIndex>& scores,
        MultiTokenPredictionIndex num_valid
    )
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
            m_output_layers[oi].backward_accumulate(
                ws.output_layer_delta,
                ws.h_last_vec,
                ws.dh_last,
                m_gradient_accumulation_workspace->output_layers[oi]
            );
            {
                const auto phase = std::format("after output head {} backward", static_cast<size_t>(oi));
                check_hidden_vector_nan_finding_mode(queue, ws.dh_last, phase.c_str());
            }
        }

        // Propagate the summed gradient through the transformer blocks.
        ws.dh.zero(queue);
        ws.din.zero(queue);
        scatter_dh_last_to_row(ws.dh_last, ws.dh, last_pos);
        check_hidden_nan_finding_mode(queue, ws.dh, m_seq_len, "after dh_last scatter");

        auto* p_dh = &ws.dh;
        auto* p_din = &ws.din;
        for (int i = static_cast<int>(m_transformer_blocks.size()) - 1; i >= 0; --i)
        {
            {
                const auto phase = std::format("before transformer block {} backward", i);
                check_hidden_nan_finding_mode(queue, *p_dh, m_seq_len, phase.c_str());
            }
            m_transformer_blocks[i].backward(
                *p_dh,
                *p_din,
                ws.transformer_block,
                m_forward_workspace->transformer_workspaces[static_cast<size_t>(i)]
            );
            m_transformer_blocks[i].accumulate_gradients(
                ws.transformer_block,
                m_gradient_accumulation_workspace->transformer_blocks[static_cast<size_t>(i)]
            );
            {
                const auto phase = std::format("after transformer block {} backward", i);
                check_hidden_nan_finding_mode(queue, *p_din, m_seq_len, phase.c_str());
            }
            std::swap(p_dh, p_din);
        }

        check_hidden_nan_finding_mode(queue, *p_dh, m_seq_len, "before input layer backward");
        m_input_layer.accumulate_backward(m_last_input, *p_dh, m_gradient_accumulation_workspace->embeddings);
    }


    void NeuralNetwork::set_random_weights_and_connections()
    {
        m_input_layer.set_random_embeddings();

        for (auto& block : m_transformer_blocks)
            block.randomize();

        for (const auto output_index : enum_iterator1D<MultiTokenPredictionIndex>())
            m_output_layers[output_index].set_random_weights();
    }


    void NeuralNetwork::dump_top_predictions()
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


    float NeuralNetwork::evaluate_average_loss(const std::vector<CpuInputLine>& evaluation_lines)
    {
        if (evaluation_lines.empty())
        {
            LOG_ERROR("evaluation lines are empty!!");
            std::abort();
            return std::numeric_limits<float>::quiet_NaN();
        }

        Score& score = m_evaluation_score;
        auto& queue = rllm::vulkan_runtime::get_queue(0);

        double total_loss = 0.0;
        for (const auto& example : evaluation_lines)
        {
            const int seq_len = static_cast<int>(example.size());
            assert(seq_len >= 2);
            const int input_len = mtp_input_len_for_sequence(seq_len);
            const auto num_valid_heads = mtp_valid_head_count_for_sequence(seq_len, input_len);

            example.sub_array(get_last_input(), static_cast<PositionIndex>(input_len));

            propagate_forward();

            double example_loss = 0.0;
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
                example_loss += static_cast<double>(loss);
            }
            total_loss += example_loss / static_cast<double>(static_cast<size_t>(num_valid_heads));
        }

        return static_cast<float>(total_loss / static_cast<double>(evaluation_lines.size()));
    }

    void NeuralNetwork::train_with_increasingly_longer_sequences(const CpuInputLine& line_of_file, bool verbose, size_t max_iterations)
    {
        CpuInputLine line;
        for (const auto& line_substring_length : enum_iterator1D<PositionIndex>(line_of_file.size()))
        {
            line_of_file.sub_array(line, line_substring_length);
            if (line.empty())
                continue; // skip empty lines that can't be used for training

            const auto full_string_opt = m_corpus.get_line(line);
            assert(full_string_opt.has_value());
            const auto& full_string = *full_string_opt;

            if (static_cast<int>(line.size()) < 2)
            {
                continue; // skip too-short lines that can't be used for training
            }

            LOG_INFO("Training on line[{}]: '{}'", (int) line_substring_length, full_string);

            do_training(line, verbose, max_iterations);
        }
    }

    void NeuralNetwork::train_with_up_to_N(const CpuInputLine& line_of_file, bool verbose, size_t max_iterations, int num_tokens)
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
        const auto& full_string = *full_string_opt;

        LOG_INFO("Training on line[{}]: '{}'", num_tokens, full_string);

        do_training(train_input, verbose, max_iterations);
    }

    void NeuralNetwork::train_with_random_len_from_start(
        const CpuInputLine& line_of_file,
        bool verbose,
        size_t max_iterations,
        std::mt19937& rng,
        float learning_rate_scale,
        bool manage_accumulator
    )
    {
        const int line_len = static_cast<int>(line_of_file.size());
        if (line_len < 2)
            return;

        // Include short prefixes (len=2) so one-token contexts can learn their next token
        // (e.g. "# -> define/include").
        const int min_len = 2;
        std::uniform_int_distribution<int> len_dist(min_len, line_len);
        const int random_len = len_dist(rng);

        CpuInputLine train_input;
        line_of_file.sub_array(train_input, static_cast<PositionIndex>(random_len));
        const auto full_string_opt = m_corpus.get_line(train_input);
        assert(full_string_opt.has_value());

        LOG_INFO("Training on random line prefix ({} toks): '{}'", random_len, *full_string_opt);

        do_training(train_input, verbose, max_iterations, learning_rate_scale, manage_accumulator);
    }

    void NeuralNetwork::batch_train(
        size_t epoch,
        const std::vector<CpuInputLine>& training_lines,
        bool verbose,
        size_t num_epochs,
        const std::optional<std::chrono::seconds>& checkpointing_interval,
        std::chrono::steady_clock::time_point& last_checkpoint_at,
        std::mt19937& rng,
        std::optional<size_t> epoch_size
    )
    {
        const size_t total_lines = training_lines.size();
        assert(total_lines > 0);
        const size_t epoch_lines = lines_per_epoch(total_lines, epoch_size);
        std::vector<size_t> line_indices(total_lines);
        std::iota(line_indices.begin(), line_indices.end(), 0);
        std::shuffle(line_indices.begin(), line_indices.end(), rng);

        for (size_t batch_start = 0; batch_start < epoch_lines; batch_start += m_micro_batch_size)
        {
            const size_t selected_count = std::min(m_micro_batch_size, epoch_lines - batch_start);
            const float progress = static_cast<float>(batch_start + selected_count) / static_cast<float>(epoch_lines);

            if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
            {
                LOG_INFO("Creating timed checkpoint at epoch: {}, line: {}, epoch lines todo: {}", epoch, batch_start + 1, epoch_lines);
                checkpoint();
            }

            std::vector<CpuInputLine> batch;
            batch.reserve(selected_count);
            for (size_t offset = 0; offset < selected_count; ++offset)
            {
                const auto& random_line = training_lines[line_indices[batch_start + offset]];
                if (static_cast<int>(random_line.size()) < 2)
                    continue;

                if (m_training_method == TrainingMethod::RANDOM_LINE_FULL)
                {
                    batch.push_back(random_line);
                    continue;
                }

                const int line_len = static_cast<int>(random_line.size());
                std::uniform_int_distribution<int> len_dist(2, line_len);
                CpuInputLine train_input;
                random_line.sub_array(train_input, static_cast<PositionIndex>(len_dist(rng)));
                batch.push_back(train_input);
            }

            if (batch.empty())
                continue;

            const float learning_rate_scale = 1.0f / static_cast<float>(batch.size());
            const auto batch_started_at = std::chrono::steady_clock::now();

            reset_gradient_accumulators();
            for (const auto& example : batch)
                do_training(example, verbose, 1, 1.0f, false);
            apply_accumulated_gradients(learning_rate_scale);

            const double average_milliseconds_per_line =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - batch_started_at).count() /
                static_cast<double>(batch.size());
            LOG_INFO(
                "Epoch[{}%] random-line[{}..{}]: {:0.2f}% done (micro-batch {}, avg {:.2f} ms/line)",
                epoch / static_cast<float>(num_epochs) * 100.0f,
                batch_start + 1,
                batch_start + selected_count,
                progress * 100.0f,
                batch.size(),
                average_milliseconds_per_line
            );
        }
    }

    void NeuralNetwork::train_random_line_random_len_epoch(size_t epoch, const std::vector<CpuInputLine>& training_lines, bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval, std::chrono::steady_clock::time_point& last_checkpoint_at, std::mt19937& rng, std::optional<size_t> epoch_size)
    {
        if (m_micro_batch_size > 1)
        {
            batch_train(
                epoch,
                training_lines,
                verbose,
                num_epochs,
                checkpointing_interval,
                last_checkpoint_at,
                rng,
                epoch_size
            );
            return;
        }

        const auto total_lines = training_lines.size();
        assert(total_lines > 0);
        const auto epoch_lines = lines_per_epoch(total_lines, epoch_size);

        // Visit each selected line exactly once per epoch (in random order).
        // Sampling with replacement over-emphasizes some lines while skipping others,
        // which causes unstable learning and apparent forgetting.
        std::vector<size_t> line_indices(total_lines);
        std::iota(line_indices.begin(), line_indices.end(), 0);
        std::shuffle(line_indices.begin(), line_indices.end(), rng);
        const auto epoch_started_at = std::chrono::steady_clock::now();

        for (size_t lines_visited = 1; lines_visited <= epoch_lines; ++lines_visited)
        {
            const float progress = static_cast<float>(lines_visited) / static_cast<float>(epoch_lines);

            if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
            {
                LOG_INFO("Creating timed checkpoint at epoch: {}, line: {}, epoch lines todo: {}", epoch, lines_visited, epoch_lines);
                checkpoint();
            }

            const auto& random_line = training_lines[line_indices[lines_visited - 1]];
            if (m_training_method == TrainingMethod::RANDOM_LINE_FULL)
            {
                // Train on the full line — all tokens are visible, last token is target.
                if (static_cast<int>(random_line.size()) >= 2)
                {
                    do_training(random_line, verbose, training_steps_per_example(m_transformer_blocks.size(), m_learn_depth));
                }
            }
            else
            {
                train_with_random_len_from_start(random_line, verbose, training_steps_per_example(m_transformer_blocks.size(), m_learn_depth), rng);
            }

            const double average_milliseconds_per_line =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - epoch_started_at).count() /
                static_cast<double>(lines_visited);
            LOG_INFO(
                "Epoch[{}%] random-line[{}]: {:0.2f}% done (avg {:.2f} ms/line)",
                epoch / static_cast<float>(num_epochs) * 100.0f,
                lines_visited,
                progress * 100.0f,
                average_milliseconds_per_line
            );
        }
    }


    void NeuralNetwork::do_line_based_training(bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval, std::optional<size_t> epoch_size)
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

        LOG_INFO(
            "Using {} training lines and {} validation lines ({}% target validation split); {} line(s) per epoch",
            training_lines.size(),
            validation_lines.size(),
            VALIDATION_PERCENT,
            epoch_lines
        );

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

        auto device_buffer_allocation_freeze = std::make_unique<ScopedDeviceBufferAllocationFreeze>();
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            bool should_stop = false;

            if (m_training_method == TrainingMethod::RANDOM_LINE_RANDOM_LEN || m_training_method == TrainingMethod::RANDOM_LINE_FULL)
            {
                train_random_line_random_len_epoch(epoch, training_lines, verbose, num_epochs, checkpointing_interval, last_checkpoint_at, rng, epoch_size);

                epoch_checkpoint(last_checkpoint_at, epoch, [&] {
                    std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                    checkpoint();
                });
            }
            else
            {
                std::shuffle(training_lines.begin(), training_lines.end(), rng);

                size_t lines_visited = 0;
                for (const auto& line_of_file : training_lines)
                {
                    if (lines_visited >= epoch_lines)
                        break;
                    lines_visited++;
                    const float progress = static_cast<float>(lines_visited) / static_cast<float>(epoch_lines);

                    if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                    {
                        std::println("creating timed checkpoint at epoch {}, line {}, epoch lines visited {}", epoch, lines_visited, epoch_lines);
                        checkpoint();
                    }

                    LOG_INFO("Epoch[{}%] line[{}]: {:0.2f}% done", epoch / static_cast<float>(num_epochs) * 100.0f, lines_visited, progress * 100.0f);

                    switch (m_training_method)
                    {
                    case TrainingMethod::TWO_TOK:
                        train_with_up_to_N(line_of_file, verbose, training_steps_per_example(m_transformer_blocks.size(), m_learn_depth), 2);
                        break;

                    case TrainingMethod::THREE_TOK:
                        train_with_up_to_N(line_of_file, verbose, training_steps_per_example(m_transformer_blocks.size(), m_learn_depth), 3);
                        break;

                    case TrainingMethod::INCREASINGLY_LONGER_SEQUENCES:
                        train_with_increasingly_longer_sequences(line_of_file, verbose, training_steps_per_example(m_transformer_blocks.size(), m_learn_depth));
                        break;

                    case TrainingMethod::RANDOM_LINE_RANDOM_LEN:
                        // Handled in the random-line loop above.
                        assert(false);
                        break;

                    case TrainingMethod::RANDOM_LINE_FULL:
                        // Handled in the random-line loop above.
                        assert(false);
                        break;

                    case TrainingMethod::WINDOW:
                        // window methods don't use the line-based loop; handled separately below
                        assert(false);
                        break;
                    }
                }

                epoch_checkpoint(last_checkpoint_at, epoch, [&] {
                    std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
                    checkpoint();
                });
            }

            if (!validation_lines.empty())
            {
                const float validation_loss = evaluate_average_loss(validation_lines);
                LOG_INFO("epoch {} validation loss: {:.6f} across {} held-out lines", epoch, validation_loss, validation_lines.size());

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
                    if (epochs_without_improvement >= EARLY_STOPPING_PATIENCE)
                    {
                        LOG_INFO("early stopping at epoch {} after {} epoch(s) without validation improvement", epoch, epochs_without_improvement);
                        should_stop = true;
                    }
                }
            }

            print_parallel_statistics_for_epoch(epoch);

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

    void NeuralNetwork::train_with_window(int window_size, bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval)
    {
        assert(window_size >= 2);

        // Collect the full flat token sequence from every corpus file.
        std::vector<TokenID> tokens;
        m_corpus.visit_flat_tokens([&](TokenID tok) {
            tokens.push_back(tok);
        });

        if (tokens.size() < static_cast<size_t>(window_size))
            return;

        // Each valid start index yields one training example.
        const size_t num_windows = tokens.size() - static_cast<size_t>(window_size) + 1;
        std::vector<size_t> indices(num_windows);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng{42};
        size_t total_windows_trained = 0;
        auto last_checkpoint_at = std::chrono::steady_clock::now();
        ScopedDeviceBufferAllocationFreeze device_buffer_allocation_freeze;
        for (size_t epoch = 0; epoch < num_epochs; ++epoch)
        {
            LOG_INFO("Epoch[{}%]: {:0.2f}% done", epoch / static_cast<float>(num_epochs) * 100.0f, 0.0f);

            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t j = 0; j < num_windows; ++j)
            {
                const float progress = static_cast<float>(j) / static_cast<float>(num_windows);

                CpuInputLine window;
                int current_try_len = 2 + random_int(0, window_size - 2); // random length between 2 and window_size
                for (int k = 0; k < current_try_len; ++k)
                {
                    window.push_back(tokens[indices[j] + static_cast<size_t>(k)]);
                }
                assert(window.size() == static_cast<PositionIndex>(current_try_len));

                total_windows_trained++;

                if (timed_checkpoint_due(checkpointing_interval, last_checkpoint_at))
                {
                    LOG_INFO("creating timed checkpoint at epoch {}, window {}, total windows trained {}", epoch, j, total_windows_trained);
                    checkpoint();
                }


                if (total_windows_trained % 100 == 0)
                {
                    const auto line_opt = m_corpus.get_line(window);
                    LOG_INFO("Epoch[{}%] window[{}]: {:0.2f}% done for '{}', successes: {}, failures: {}", epoch / static_cast<float>(num_epochs) * 100.0f, j, progress * 100.0f, line_opt.has_value() ? line_opt->c_str() : "unknown", m_stats.num_learning_successes(), m_stats.num_learning_failures());
                }

                do_training(window, verbose, training_steps_per_example(m_transformer_blocks.size(), m_learn_depth));
            }

            std::println("creating end-of-epoch checkpoint at epoch {}", epoch);
            checkpoint();
            print_parallel_statistics_for_epoch(epoch);
        }
    }

    void NeuralNetwork::do_whole_corpus_window_based_training(bool verbose, size_t num_epochs, const std::optional<std::chrono::seconds>& checkpointing_interval)
    {
        // Window methods operate on the flat token stream rather than per-line.
        assert(m_training_method == TrainingMethod::WINDOW);
        train_with_window(m_window_size, verbose, num_epochs, checkpointing_interval);
    }


    void NeuralNetwork::train(bool verbose, size_t num_epochs, const std::optional<std::string>& input_filename, const std::optional<std::chrono::seconds>& checkpointing_interval, std::optional<size_t> epoch_size)
    {
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
            "\t fires nothing CE loss:  {:.6f}\n"
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
            m_convergence_threshold,
            m_fires_nothing_ce_loss,
            training_steps_per_example(n_layers, m_learn_depth),
            m_micro_batch_size,
            num_epochs,
            epoch_size.has_value() ? std::format("{} line(s)", epoch_size.value()) : std::string{"all training lines"},
            training_method_to_string(m_training_method)
        );

        if (training_method_is_line_based())
        {
            do_line_based_training(verbose, num_epochs, checkpointing_interval, epoch_size);
        }
        else
        {
            do_whole_corpus_window_based_training(verbose, num_epochs, checkpointing_interval);
        }
    }


    void NeuralNetwork::do_training(
        const CpuInputLine& train_output,
        bool verbose,
        size_t max_iterations,
        float learning_rate_scale,
        bool manage_accumulator
    )
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
        // Head-0 target is used for logging and convergence checks.
        const auto expected_output_token = mtp_target_for_head(train_output, _input_len, MultiTokenPredictionIndex::START);

        bool print_loss_verbose = false;
        static int count = 0;
        if (count++ % 100 == 0)
        {
            print_loss_verbose = true;
        }

        const auto full_string_opt = m_corpus.get_line(train_output);
        assert(full_string_opt.has_value());
        const auto& full_string = *full_string_opt;

        get_last_input() = train_input; // set the input to the train input for tracing

        // In multi-epoch training each call gets a small fixed budget (max_iterations).
        // We allow an early return once this example reaches a reasonable confidence target.
        float loss = 0.0f;
        for (size_t i = 0; i < max_iterations; ++i)
        {
            if (manage_accumulator)
                reset_gradient_accumulators();

            propagate_forward();

            m_training_scores.set_size(num_valid_heads);
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
                    return;
                }
                if (_k == MultiTokenPredictionIndex::START)
                    loss = _k_loss;
            }

            propagate_backward_mtp(m_training_scores, num_valid_heads);
            if (manage_accumulator)
                apply_accumulated_gradients(learning_rate_scale);

            if (loss < m_convergence_threshold)
            {
                LOG_INFO("Convergence reached after {} steps for expected '{}', full string: '{}', input size: {}", i + 1, m_corpus.get_token_from_id(expected_output_token), full_string, static_cast<size_t>(train_input.size()));
                m_stats.record_learning_success();
                return;
            }

            if (print_loss_verbose)
            {
                const auto expected_token = m_corpus.get_token_from_id(expected_output_token);
                LOG_INFO("Training iteration[{}], wanted: '{}' ({}), full string: '{}'", i, expected_token, expected_output_token, full_string);
                LOG_INFO("  Loss: {:.6f}", loss);
                dump_top_predictions();
            }
        }

        if (manage_accumulator || max_iterations != 1)
        {
            LOG_INFO(
                "Steps exhausted ({}) for this line. loss = {:.6f}, threshold = {:.6f}, expected token: '{}' ({}), full "
                "string: '{}', input size: {}.",
                max_iterations,
                loss,
                m_convergence_threshold,
                m_corpus.get_token_from_id(expected_output_token),
                expected_output_token,
                full_string,
                static_cast<size_t>(train_input.size())
            );
            m_stats.record_learning_failure();
        }
    }


} // namespace rllm
