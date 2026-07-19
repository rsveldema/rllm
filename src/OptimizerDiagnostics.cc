#include <OptimizerDiagnostics.hpp>

#include <cmath>

#include <cpu/cpu_fixed_vector.hpp>
#include <logging.hpp>
#include <rllm_vulkan_runtime.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <parallel.hpp>

namespace rllm
{
    namespace
    {
        bool enabled = false;

        cpu_fixed_vector<float, TempStorage>& zero_host_buffer()
        {
            static cpu_fixed_vector<float, TempStorage> zero;
            zero.set_size(TempStorage::MAX);
            for (size_t i = 0; i < static_cast<size_t>(TempStorage::MAX); ++i)
                zero[static_cast<TempStorage>(i)] = 0.0f;
            return zero;
        }

        cpu_fixed_vector<float, TempStorage>& read_host_buffer()
        {
            static cpu_fixed_vector<float, TempStorage> values;
            return values;
        }
    }

    fixed_size_vector<float, TempStorage>& optimizer_diagnostics_buffer()
    {
        static fixed_size_vector<float, TempStorage> values;
        return values;
    }

    bool optimizer_diagnostics_enabled() { return enabled; }

    void prepare_optimizer_gradient_clip(
        // OFFLOAD_PARAMETERS(values)
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TempStorage>(), (values))
        if (i == TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM)
            values[i] = 0.0f;
        else if (i == TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE)
            values[i] = 1.0f;
        ENDFOR
    }

    void accumulate_optimizer_gradient_norm(
        // OFFLOAD_PARAMETERS(gradient, values)
        const fixed_size_matrix<float, TokenID, EmbeddingDimension>& gradient,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (gradient, values))
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (gradient[r, c] * gradient[r, c]));
        ENDFOR
    }

    void accumulate_optimizer_gradient_norm(
        // OFFLOAD_PARAMETERS(gradient, values)
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (gradient, values))
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (gradient[r, c] * gradient[r, c]));
        ENDFOR
    }

    void accumulate_optimizer_gradient_norm(
        // OFFLOAD_PARAMETERS(gradient, values)
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (gradient, values))
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (gradient[r, c] * gradient[r, c]));
        ENDFOR
    }

    void accumulate_optimizer_gradient_norm(
        // OFFLOAD_PARAMETERS(gradient, values)
        const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (gradient, values))
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (gradient[r, c] * gradient[r, c]));
        ENDFOR
    }

    void finalize_optimizer_gradient_clip(
        // OFFLOAD_PARAMETERS(values)
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, i, enum_iterator1D<TempStorage>(static_cast<TempStorage>(1)), (values))
        const float norm = sqrt(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]);
        values[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE] = ((norm > 1.0f) ? (1.0f / norm) : 1.0f);
        ENDFOR
    }

    static void accumulate_hidden_gradient_stats(
        // OFFLOAD_PARAMETERS(gradient, rows, values)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& gradient,
        PositionIndex rows,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (gradient, rows, values))
        const float value = gradient[r, c];
        atomicMax(values[TempStorage::OPTIMIZER_GRADIENT_MAX], abs(value));
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (value * value));
        ENDFOR
    }

    static void accumulate_ffn_gradient_stats(
        // OFFLOAD_PARAMETERS(gradient, rows, values)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gradient,
        PositionIndex rows,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, r, c, grid, (gradient, rows, values))
        const float value = gradient[r, c];
        atomicMax(values[TempStorage::OPTIMIZER_GRADIENT_MAX], abs(value));
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (value * value));
        ENDFOR
    }

    static void accumulate_attention_matrix_gradient_stats(
        // OFFLOAD_PARAMETERS(gradient, rows, values)
        const fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>& gradient,
        PositionIndex rows,
        fixed_size_vector<float, TempStorage>& values
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, r, c, rows, (gradient, rows, values))
        const float value = gradient[r, c];
        atomicMax(values[TempStorage::OPTIMIZER_GRADIENT_MAX], abs(value));
        atomicAdd(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM], (value * value));
        ENDFOR
    }

    void log_hidden_gradient_diagnostics(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& gradient,
        PositionIndex rows,
        std::string_view label)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto& values = optimizer_diagnostics_buffer();
        values.copy_from_cpu(queue, zero_host_buffer());
        accumulate_hidden_gradient_stats(gradient, rows, values);

        auto& host = read_host_buffer();
        values.copy_to_cpu(queue, host);
        queue.wait("hidden gradient diagnostics");
        const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(EmbeddingDimension::MAX);
        const double rms = std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM] / static_cast<double>(count));
        LOG_INFO("Backward gradient [{}]: RMS {:.7g}, max {:.7g}, L2 norm {:.7g} ({} values)",
            label, rms, host[TempStorage::OPTIMIZER_GRADIENT_MAX],
            std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]), count);
    }

    void log_hidden_activation_diagnostics(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& activation,
        PositionIndex rows,
        std::string_view label)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto& values = optimizer_diagnostics_buffer();
        values.copy_from_cpu(queue, zero_host_buffer());
        accumulate_hidden_gradient_stats(activation, rows, values);

        auto& host = read_host_buffer();
        values.copy_to_cpu(queue, host);
        queue.wait("hidden activation diagnostics");
        const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(EmbeddingDimension::MAX);
        const double rms = std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM] / static_cast<double>(count));
        LOG_INFO("Forward activation [{}]: RMS {:.7g}, max {:.7g}, L2 norm {:.7g} ({} values)",
            label, rms, host[TempStorage::OPTIMIZER_GRADIENT_MAX],
            std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]), count);
    }

    void log_ffn_gradient_diagnostics(
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gradient,
        PositionIndex rows,
        std::string_view label)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto& values = optimizer_diagnostics_buffer();
        values.copy_from_cpu(queue, zero_host_buffer());
        accumulate_ffn_gradient_stats(gradient, rows, values);

        auto& host = read_host_buffer();
        values.copy_to_cpu(queue, host);
        queue.wait("FFN gradient diagnostics");
        const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(FFDimension::MAX);
        const double rms = std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM] / static_cast<double>(count));
        LOG_INFO("Backward gradient [{}]: RMS {:.7g}, max {:.7g}, L2 norm {:.7g} ({} values)",
            label, rms, host[TempStorage::OPTIMIZER_GRADIENT_MAX],
            std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]), count);
    }

    void log_attention_matrix_gradient_diagnostics(
        const fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& gradients,
        PositionIndex rows,
        std::string_view label,
        bool log_per_head)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto& values = optimizer_diagnostics_buffer();
        values.copy_from_cpu(queue, zero_host_buffer());
        for (const auto head : enum_iterator1D<HeadsIndex>())
            accumulate_attention_matrix_gradient_stats(gradients[head], rows, values);

        auto& host = read_host_buffer();
        values.copy_to_cpu(queue, host);
        queue.wait("attention matrix gradient diagnostics");
        const size_t triangular_count = static_cast<size_t>(rows) * (static_cast<size_t>(rows) + 1) / 2;
        const size_t count = static_cast<size_t>(HeadsIndex::MAX) * triangular_count;
        const double rms = std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM] / static_cast<double>(count));
        LOG_INFO("Backward gradient [{}]: RMS {:.7g}, max {:.7g}, L2 norm {:.7g} ({} values across {} heads)",
            label, rms, host[TempStorage::OPTIMIZER_GRADIENT_MAX],
            std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]), count,
            static_cast<size_t>(HeadsIndex::MAX));

        if (!log_per_head)
            return;
        for (const auto head : enum_iterator1D<HeadsIndex>())
        {
            values.copy_from_cpu(queue, zero_host_buffer());
            accumulate_attention_matrix_gradient_stats(gradients[head], rows, values);
            values.copy_to_cpu(queue, host);
            queue.wait("per-head attention matrix gradient diagnostics");
            const double head_rms = std::sqrt(
                host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM] / static_cast<double>(triangular_count));
            LOG_INFO("Backward gradient [{} head {}]: RMS {:.7g}, max {:.7g}, L2 norm {:.7g} ({} values)",
                label, static_cast<size_t>(head), head_rms,
                host[TempStorage::OPTIMIZER_GRADIENT_MAX],
                std::sqrt(host[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]), triangular_count);
        }
    }

    void begin_optimizer_diagnostics()
    {
        optimizer_diagnostics_buffer().copy_from_cpu(
            rllm::vulkan_runtime::get_queue(0), zero_host_buffer());
        enabled = true;
    }

    void log_optimizer_diagnostics(std::string_view parameter_group)
    {
        auto& queue = rllm::vulkan_runtime::get_queue(0);
        auto& values = read_host_buffer();
        optimizer_diagnostics_buffer().copy_to_cpu(queue, values);
        queue.wait("optimizer diagnostics");
        enabled = false;
        const float count = values[TempStorage::OPTIMIZER_PARAMETER_COUNT];
        if (count == 0.0f)
            return;
        LOG_INFO(
            "Optimizer diagnostics [{}]: pre-clip gradient max {:.7g}, global norm {:.7g}, global clip scale {:.7g}, element-clipped {:.4f}%, Adam update RMS {:.7g} max {:.7g}, weight update RMS {:.7g} max {:.7g} ({} parameters)",
            parameter_group,
            values[TempStorage::OPTIMIZER_GRADIENT_MAX],
            std::sqrt(values[TempStorage::OPTIMIZER_GRADIENT_SQUARE_SUM]),
            values[TempStorage::OPTIMIZER_GLOBAL_CLIP_SCALE],
            100.0f * values[TempStorage::OPTIMIZER_CLIPPED_COUNT] / count,
            std::sqrt(values[TempStorage::OPTIMIZER_ADAM_UPDATE_SQUARE_SUM] / count),
            values[TempStorage::OPTIMIZER_ADAM_UPDATE_MAX],
            std::sqrt(values[TempStorage::OPTIMIZER_WEIGHT_UPDATE_SQUARE_SUM] / count),
            values[TempStorage::OPTIMIZER_WEIGHT_UPDATE_MAX],
            static_cast<size_t>(count));
    }
}
