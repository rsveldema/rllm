#pragma once

#include <LayerPrimitives.hpp>
#include <fixed_size_matrix.hpp>
#include <flexible_rows_matrix.hpp>

namespace rllm
{
    /** Token-conditioned routing over earlier transformer residuals.
     *
     * A token owns one logit per source layer. Destination layer L normalizes
     * the prefix [0, L), so the same compact token x layer table is shared by
     * every destination. Logits increase with source depth at initialization,
     * making the immediate predecessor effectively one-hot.
     */
    class DynamicSkipAttention
    {
      public:
        static constexpr float INITIAL_LOGIT_STEP = 20.0f;

        DynamicSkipAttention();
        void initialize();

        void begin_mix(
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& output,
            PositionIndex rows) const;
        void add_source(
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& output,
            const GpuInputLine& tokens,
            int rows,
            int source_layer,
            int destination_layer) const;
        void backward_source(
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& mixed_gradient,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& mixed,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& source_gradient,
            const GpuInputLine& tokens,
            int rows,
            int source_layer,
            int destination_layer,
            fixed_size_matrix<float, TokenID, HeadsIndex>& logit_gradient) const;
        void apply_update(
            fixed_size_matrix<float, TokenID, HeadsIndex>& gradient,
            float learning_rate, float bias_correction1, float bias_correction2);

        const fixed_size_matrix<float, TokenID, HeadsIndex>& logits() const
        { return m_logits; }
        fixed_size_matrix<float, TokenID, HeadsIndex>& logits()
        { return m_logits; }
        const fixed_size_matrix<float, TokenID, HeadsIndex>& adam_first() const { return m_adam_first; }
        const fixed_size_matrix<float, TokenID, HeadsIndex>& adam_second() const { return m_adam_second; }
        fixed_size_matrix<float, TokenID, HeadsIndex>& adam_first() { return m_adam_first; }
        fixed_size_matrix<float, TokenID, HeadsIndex>& adam_second() { return m_adam_second; }

      private:
        fixed_size_matrix<float, TokenID, HeadsIndex> m_logits;
        fixed_size_matrix<float, TokenID, HeadsIndex> m_adam_first;
        fixed_size_matrix<float, TokenID, HeadsIndex> m_adam_second;
    };
}
