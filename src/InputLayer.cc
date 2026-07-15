#include <InputLayer.hpp>
#include <WeightInitialization.hpp>
#include <RandomHelpers.hpp>
#include <RuntimeConfig.hpp>
#include <TransformerBlock.hpp>
#include <cpu/cpu_flex_rows_matrix.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace rllm
{
    static constexpr size_t embedding_gradient_index(TokenID tok, EmbeddingDimension dim)
    {
        return static_cast<size_t>(tok) * static_cast<size_t>(EmbeddingDimension::MAX) + static_cast<size_t>(dim);
    }

    EmbeddingGradientAccumulator::EmbeddingGradientAccumulator()
        : gradients(static_cast<size_t>(TokenID::MAX) * static_cast<size_t>(EmbeddingDimension::MAX), 0.0f)
        , touched(static_cast<size_t>(TokenID::MAX), 0)
    {
    }

    void EmbeddingGradientAccumulator::reset()
    {
        std::fill(gradients.begin(), gradients.end(), 0.0f);
        std::fill(touched.begin(), touched.end(), 0);
    }

    void EmbeddingGradientAccumulator::add(TokenID tok, EmbeddingDimension dim, float value)
    {
        gradients[embedding_gradient_index(tok, dim)] += value;
        touched[static_cast<size_t>(tok)] = 1;
    }

    bool EmbeddingGradientAccumulator::is_touched(TokenID tok) const
    {
        return touched[static_cast<size_t>(tok)] != 0;
    }

    float EmbeddingGradientAccumulator::get(TokenID tok, EmbeddingDimension dim) const
    {
        return gradients[embedding_gradient_index(tok, dim)];
    }

    static constexpr float NAN_FINDING_INPUT_HIDDEN_ABS_BOUND = 2.0f;
    static constexpr float NAN_FINDING_INPUT_GRADIENT_ABS_BOUND = 10000.0f;

    static fixed_size_vector<int, PositionIndex> make_input_layer_nan_scan_flag(VulkanQueue& queue)
    {
        cpu_fixed_vector<int, PositionIndex> cpu_flag;
        cpu_flag.set_size(PositionIndex::MAX);
        cpu_flag.zero();

        fixed_size_vector<int, PositionIndex> flag;
        flag.copy_from_cpu(queue, cpu_flag);
        return flag;
    }

    static bool input_layer_nan_scan_failed(VulkanQueue& queue, fixed_size_vector<int, PositionIndex>& flag)
    {
        cpu_fixed_vector<int, PositionIndex> cpu_flag;
        flag.copy_to_cpu(queue, cpu_flag);
        return cpu_flag[PositionIndex::START] != 0;
    }

    static void scan_input_embedding_matrix(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(matrix, flag, lower_bound, upper_bound)
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, lower_bound, upper_bound))
        const float value = matrix[row, col];
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

    static void scan_input_float_matrix(
        VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(matrix, flag, rows, lower_bound, upper_bound)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& matrix,
        fixed_size_vector<int, PositionIndex>& flag,
        PositionIndex rows,
        float lower_bound,
        float upper_bound
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, col, grid, (matrix, flag, rows, lower_bound, upper_bound))
        const float value = matrix[row, col];
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

    void InputLayer::check_nan_finding_mode_embeddings(const char* phase) const
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_input_layer_nan_scan_flag(queue);
        scan_input_embedding_matrix(queue, m_embeddings, flag, -1.0f, 1.0f);
        if (input_layer_nan_scan_failed(queue, flag))
        {
            std::fprintf(stderr, "NAN_FINDING_MODE: InputLayer embeddings invalid during %s\n", phase);
            std::abort();
        }
    }

    void InputLayer::check_nan_finding_mode_matrix(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& values,
        PositionIndex rows,
        const char* name,
        const char* phase
    ) const
    {
        if (!nan_finding_mode_enabled())
            return;

        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto flag = make_input_layer_nan_scan_flag(queue);
        const float bound = (std::string_view(name) == "hidden state")
            ? NAN_FINDING_INPUT_HIDDEN_ABS_BOUND
            : NAN_FINDING_INPUT_GRADIENT_ABS_BOUND;
        scan_input_float_matrix(queue, values, flag, rows, -bound, bound);
        if (input_layer_nan_scan_failed(queue, flag))
        {
            std::fprintf(
                stderr,
                "NAN_FINDING_MODE: InputLayer %s invalid during %s; expected finite values in [%g, %g]\n",
                name,
                phase,
                static_cast<double>(-bound),
                static_cast<double>(bound)
            );
            std::abort();
        }
    }

    static void fill_embeddings_with_positional_encoding(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(tokens, embeddings, h, model_dim)
        const GpuInputLine& tokens,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& embeddings,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
        float model_dim
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(tokens.size());

        OFFLOAD_PARFOR_2D_PARAM(queue, pos, di, grid, (tokens, embeddings, h, model_dim))
        const int tok = static_cast<int>(tokens[pos]);
        const int di_int = static_cast<int>(di);
        const float emb_val = embeddings[tok, di];
        const float freq = (1.0f / std::pow(10000.0f, (static_cast<float>((di_int & ~1)) / model_dim)));
        const float pos_f = static_cast<float>(pos);
        const float pe = ((di_int % 2) == 0) ? std::sin((pos_f * freq)) : std::cos((pos_f * freq));
        h[pos, di] = static_cast<float>((emb_val + pe));
        ENDFOR
    }

    static void fill_packed_embeddings_with_positional_encoding(VulkanQueue& queue,
        // OFFLOAD_PARAMETERS(tokens, local_positions, embeddings, h, model_dim, packed_rows)
        const GpuInputLine& tokens,
        const fixed_size_vector<int, PositionIndex>& local_positions,
        const fixed_size_matrix<float16, TokenID, EmbeddingDimension>& embeddings,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h,
        float model_dim,
        PositionIndex packed_rows
        // END_OFFLOAD_PARAMETERS
    )
    {
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(packed_rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, di, grid, (tokens, local_positions, embeddings, h, model_dim))
        const int tok = static_cast<int>(tokens[row]);
        const int di_int = static_cast<int>(di);
        const float emb_val = embeddings[tok, di];
        const float freq = (1.0f / std::pow(10000.0f, (static_cast<float>((di_int & ~1)) / model_dim)));
        const float pos_f = static_cast<float>(local_positions[row]);
        const float pe = ((di_int % 2) == 0) ? std::sin((pos_f * freq)) : std::cos((pos_f * freq));
        h[row, di] = static_cast<float>((emb_val + pe));
        ENDFOR
    }

    void InputLayer::reset_embeddings()
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_embeddings.zero(queue);
        m_embeddings_cpu.zero();
        m_adam_first.zero();
        m_adam_second.zero();
    }

    void InputLayer::set_random_embeddings(EmbeddingInitializerType type)
    {
        auto initializer = make_embedding_initializer(type, static_cast<size_t>(EmbeddingDimension::MAX));
        for (const auto tok : enum_iterator1D<TokenID>())
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                m_embeddings_cpu.set(tok, d, static_cast<float16>(initializer->getNextValue()));
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        m_embeddings.copy_from_cpu(queue, m_embeddings_cpu);
    }

    // Fill h[T × D_MODEL] = token_embedding + sinusoidal positional encoding.
    // PE[pos, 2i]   = sin(pos / 10000^(2i / D_MODEL))
    // PE[pos, 2i+1] = cos(pos / 10000^(2i / D_MODEL))
    void InputLayer::propagate_forward(
        const CpuInputLine& input,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        auto& gpu_input = const_cast<GpuInputLine&>(m_gpu_input);
        propagate_forward(input, gpu_input, h);
    }

    void InputLayer::propagate_forward(
        const CpuInputLine& input,
        GpuInputLine& gpu_input,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        h.set_rows(static_cast<PositionIndex>(input.size()));

        // Sync CPU input to GPU before OFFLOAD region
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        gpu_input.sync_to_device(queue, input);

        check_nan_finding_mode_embeddings("forward:start");
        fill_embeddings_with_positional_encoding(queue, gpu_input, m_embeddings, h, static_cast<float>(EmbeddingDimension::MAX));
        check_nan_finding_mode_matrix(h, static_cast<PositionIndex>(input.size()), "hidden state", "forward:end");
    }

    void InputLayer::propagate_forward(
        const PackedBatchInput& input,
        GpuPackedBatchInput& gpu_input,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h
    ) const
    {
        h.set_rows(input.packed_rows());
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        gpu_input.sync_to_device(queue, input);
        check_nan_finding_mode_embeddings("batched-forward:start");
        fill_packed_embeddings_with_positional_encoding(
            queue,
            gpu_input.tokens,
            gpu_input.local_position,
            m_embeddings,
            h,
            static_cast<float>(EmbeddingDimension::MAX),
            input.packed_rows()
        );
        check_nan_finding_mode_matrix(h, input.packed_rows(), "batched hidden state", "batched-forward:end");
    }


    // Update token embeddings.  dh[T × D_MODEL] = ∂L/∂h.
    // Positional encodings are fixed, so only the embedding contribution is updated.
    void InputLayer::propagate_backward(
        const CpuInputLine& input,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        float learning_rate
    )
    {
        // D2H: download the gradient matrix to CPU before element access
        cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> dh_cpu;
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        check_nan_finding_mode_matrix(dh, static_cast<PositionIndex>(input.size()), "gradient", "backward:dh");
        check_nan_finding_mode_embeddings("backward:start");
        const_cast<flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>&>(dh).copy_to_cpu(queue, dh_cpu);

        // Use per-instance state (not static) to avoid data race with PARFOR_2D.
        // Previously: static local variables shared across threads → heap corruption.
        m_updated_tokens.zero();
        m_conflicts.clear();

        size_t duplicate_count = 0;
        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
        {
            const auto tok = input.get(pos);
            m_updated_tokens[tok]++;;

            if (m_updated_tokens[tok] > 1)
            {
                assert(duplicate_count < static_cast<size_t>(ConflictIndex::MAX));
                m_conflicts.push_back(ConflictingToken{.tok = tok, .pos = pos});
                duplicate_count++;
            }
        }

        // Handle duplicates sequentially to avoid race conditions on the same embedding vector
        for (size_t i = 0; i < duplicate_count; i++)
        {
            for (const auto di : enum_iterator1D<EmbeddingDimension>())
            {
                const auto& conflict_tok = m_conflicts[i].tok;
                const auto& conflict_pos = m_conflicts[i].pos;
                const auto rate = learning_rate / static_cast<float>(m_updated_tokens[conflict_tok]);
                m_embeddings_cpu.set(conflict_tok, di,
                    math::clamp(
                        static_cast<float>(m_embeddings_cpu.get(conflict_tok, di)) + rate * dh_cpu.get(conflict_pos, di),
                        RLMM_NEG_ONE, RLMM_ONE));
            }
        }

        // Tokens appearing exactly once can be updated in parallel safely
        PARFOR_2D(pos, di, enum_iterator2D<PositionIndex, EmbeddingDimension>(input.size()))
        {
            const auto tok = input.get(pos);
            const auto count = m_updated_tokens[tok];
            assert(count > 0);
            if (count > 1)
                continue;

            assert(count == 1);

            m_embeddings_cpu.set(tok, di,
                math::clamp(
                    static_cast<float>(m_embeddings_cpu.get(tok, di)) + learning_rate * dh_cpu.get(pos, di),
                    RLMM_NEG_ONE, RLMM_ONE));
        }
        ENDFOR

        // Clean up per-token counters and copy modified rows to offload buffer
        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
        {
            const auto tok = input.get(pos);
            if (m_updated_tokens[tok] == 0)
                continue;
            m_embeddings.copy_row_to_offload_buffer(queue, tok, m_embeddings_cpu);
            m_updated_tokens[tok] = 0;
        }
        check_nan_finding_mode_embeddings("backward:end");
    }

    void InputLayer::accumulate_backward_packed(
        const PackedBatchInput& input,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        EmbeddingGradientAccumulator& accumulator
    )
    {
        cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> dh_cpu;
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const_cast<flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>&>(dh).copy_to_cpu(queue, dh_cpu);
        for (const auto row : enum_iterator1D<PositionIndex>(input.packed_rows()))
        {
            const auto tok = input.tokens()[row];
            for (const auto d : enum_iterator1D<EmbeddingDimension>())
                accumulator.add(tok, d, dh_cpu[row, d]);
        }
    }

    void InputLayer::accumulate_backward(
        const CpuInputLine& input,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
        EmbeddingGradientAccumulator& accumulator
    )
    {
        cpu_flex_rows_matrix<float, PositionIndex, EmbeddingDimension> dh_cpu;
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        check_nan_finding_mode_matrix(dh, static_cast<PositionIndex>(input.size()), "gradient", "accumulate_backward:dh");
        const_cast<flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>&>(dh).copy_to_cpu(queue, dh_cpu);

        m_updated_tokens.zero();
        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
            m_updated_tokens[input.get(pos)]++;

        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
        {
            const auto tok = input.get(pos);
            const auto count = m_updated_tokens[tok];
            assert(count > 0);
            const float scale = 1.0f / static_cast<float>(count);
            for (const auto di : enum_iterator1D<EmbeddingDimension>())
                accumulator.add(tok, di, scale * dh_cpu.get(pos, di));
        }

        for (const auto pos : enum_iterator1D<PositionIndex>(input.size()))
            m_updated_tokens[input.get(pos)] = 0;
    }

    void InputLayer::apply_accumulated_update(EmbeddingGradientAccumulator& accumulator, float learning_rate, float bias_correction1, float bias_correction2)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        check_nan_finding_mode_embeddings("apply_accumulated_update:start");
        for (const auto tok : enum_iterator1D<TokenID>())
        {
            if (!accumulator.is_touched(tok))
                continue;
            for (const auto di : enum_iterator1D<EmbeddingDimension>())
            {
                const float gradient = accumulator.get(tok, di);
                const float first = (TransformerBlock::ADAM_BETA1 * m_adam_first.get(tok, di)) + ((1.0f - TransformerBlock::ADAM_BETA1) * gradient);
                const float second = (TransformerBlock::ADAM_BETA2 * m_adam_second.get(tok, di)) + ((1.0f - TransformerBlock::ADAM_BETA2) * gradient * gradient);
                m_adam_first.set(tok, di, first);
                m_adam_second.set(tok, di, second);
                const float update = (first / bias_correction1) / (std::sqrt(second / bias_correction2) + TransformerBlock::ADAM_EPSILON);
                const float decayed = static_cast<float>(m_embeddings_cpu.get(tok, di)) * (1.0f - learning_rate * TransformerBlock::WEIGHT_DECAY);
                m_embeddings_cpu.set(tok, di,
                    math::clamp(
                        decayed + learning_rate * update,
                        RLMM_NEG_ONE,
                        RLMM_ONE));
            }
            m_embeddings.copy_row_to_offload_buffer(queue, tok, m_embeddings_cpu);
        }
        check_nan_finding_mode_embeddings("apply_accumulated_update:end");
    }


    void InputLayer::get_embedding(TokenID tok, embedding_row_t& out) const
    {
        fixed_size_matrix<float16, TokenID, EmbeddingDimension>::export_row(tok, m_embeddings_cpu, out);
    }

} // namespace rllm
