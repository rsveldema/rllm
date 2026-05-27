#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <vector>

namespace rllm
{
    // InputLayer converts an InputLine (sequence of token IDs) into a
    // flat hidden-state vector h[seq_len × EmbeddingDimension::MAX].
    // Each position receives its learned token embedding plus a
    // fixed sinusoidal positional encoding.
    class InputLayer
    {
      public:
        InputLayer()
        {
            reset_embeddings();
        }
        ~InputLayer() = default;
        InputLayer(const InputLayer&) = delete;
        InputLayer& operator=(const InputLayer&) = delete;

        // Fill h[seq_len × D_MODEL] with (token_embedding + positional_encoding).
        void propagate_forward(const InputLine& input,
                flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h) const;

        // Update token embeddings using dh[seq_len × D_MODEL] = ∂L/∂h.
        // Positional encodings are fixed (sinusoidal), so only embeddings change.
        void propagate_backward(
            const InputLine& input,
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dh,
            float learning_rate
        );

        void set_random_embeddings();

        // Returns the raw learned embedding for a single token (without positional encoding).
        const fixed_size_vector<rlmm_float_small, EmbeddingDimension>& get_embedding(TokenID tok) const
        {
            return m_embeddings[tok];
        }

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        // m_embeddings[token_id][d] — learned embedding for dimension d of token_id.
        fixed_size_vector<fixed_size_vector<rlmm_float_small, EmbeddingDimension>, TokenID> m_embeddings;

        void reset_embeddings();
    };

} // namespace rllm
