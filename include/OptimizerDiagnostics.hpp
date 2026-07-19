#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include <fixed_size_vector.hpp>
#include <fixed_size_obj_vector.hpp>
#include <fixed_size_triangular_matrix.hpp>
#include <LayerPrimitives.hpp>

namespace rllm
{
    struct OptimizerDiagnosticMetrics
    {
        std::string parameter_group;
        float global_gradient_norm = 0.0f;
        float global_clip_scale = 1.0f;

        float normalized_global_gradient_norm() const
        {
            return global_gradient_norm * global_clip_scale;
        }
    };

    fixed_size_vector<float, TempStorage>& optimizer_diagnostics_buffer();
    bool optimizer_diagnostics_enabled();
    void begin_optimizer_diagnostics();
    std::optional<OptimizerDiagnosticMetrics> log_optimizer_diagnostics(std::string_view parameter_group);
    void prepare_optimizer_gradient_clip(fixed_size_vector<float, TempStorage>& values);
    void finalize_optimizer_gradient_clip(fixed_size_vector<float, TempStorage>& values);
    void accumulate_optimizer_gradient_norm(const fixed_size_matrix<float, TokenID, EmbeddingDimension>& gradient, fixed_size_vector<float, TempStorage>& values);
    void accumulate_optimizer_gradient_norm(const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient, fixed_size_vector<float, TempStorage>& values);
    void accumulate_optimizer_gradient_norm(const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient, fixed_size_vector<float, TempStorage>& values);
    void accumulate_optimizer_gradient_norm(const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient, fixed_size_vector<float, TempStorage>& values);
    void log_hidden_gradient_diagnostics(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& gradient,
        PositionIndex rows,
        std::string_view label);
    void log_hidden_activation_diagnostics(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& activation,
        PositionIndex rows,
        std::string_view label);
    void log_ffn_gradient_diagnostics(
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& gradient,
        PositionIndex rows,
        std::string_view label);
    void log_attention_matrix_gradient_diagnostics(
        const fixed_size_obj_vector<fixed_size_triangular_matrix<float, PositionIndex, PositionIndex>, HeadsIndex>& gradients,
        PositionIndex rows,
        std::string_view label,
        bool log_per_head = false);
}
