#pragma once

#include <LayerPrimitives.hpp>
#include <cpu/cpu_fixed_vector.hpp>
#include <cpu/cpu_fixed_matrix.hpp>
#include <safetensors.hh>

#include <nlohmann/json_fwd.hpp>
#include <vector>
#include <string>

namespace rllm
{
    /** the embedding for a given TokenID */    
    using embedding_row_t = std::array<float16, static_cast<size_t>(EmbeddingDimension::MAX)>;

    struct EmbeddingGradientAccumulator
    {
        std::vector<float> gradients;
        std::vector<uint8_t> touched;

        EmbeddingGradientAccumulator();
        void reset();
        void add(TokenID tok, EmbeddingDimension dim, float value);
        bool is_touched(TokenID tok) const;
        float get(TokenID tok, EmbeddingDimension dim) const;
    };

      /**  InputLayer converts an CpuInputLine (sequence of token IDs) into a
    * flat hidden-state vector h[seq_len × EmbeddingDimension::MAX].
    * Each position receives its learned token embedding plus a
    * fixed sinusoidal positional encoding.
    */

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
        void propagate_forward(const CpuInputLine& input,
                flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h) const;
        void propagate_forward(
            const CpuInputLine& input,
            GpuInputLine& gpu_input,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h
        ) const;
        void propagate_forward(
            const PackedBatchInput& input,
            GpuPackedBatchInput& gpu_input,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h
        ) const;

        // Update token embeddings using dh[seq_len × D_MODEL] = ∂L/∂h.
        // Positional encodings are fixed (sinusoidal), so only embeddings change.
        void propagate_backward(
            const CpuInputLine& input,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            float learning_rate
        );

        void accumulate_backward(
            const CpuInputLine& input,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            EmbeddingGradientAccumulator& accumulator
        );
        void accumulate_backward_packed(
            const PackedBatchInput& input,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dh,
            EmbeddingGradientAccumulator& accumulator
        );

        void apply_accumulated_update(EmbeddingGradientAccumulator& accumulator, float learning_rate, float bias_correction1, float bias_correction2);

        void set_random_embeddings();

        // Returns the raw learned embedding for a single token (without positional encoding).
        void get_embedding(TokenID tok, embedding_row_t& out) const;

        void load(const nlohmann::json& j);
        nlohmann::json save() const;
        void load_from_safetensors(const std::string& filename, std::string* err = nullptr);
        void save_to_safetensors(const std::string& filename,
                                         std::string* warn = nullptr,
                                         std::string* err = nullptr) const;

    friend class TextTrainer;

    private:
        // m_embeddings[token_id][d] — learned embedding for dimension d of token_id.
        fixed_size_matrix<float16, TokenID, EmbeddingDimension> m_embeddings;
        // CPU-side copy used for gradient updates and serialization.
        cpu_fixed_matrix<float16, TokenID, EmbeddingDimension> m_embeddings_cpu;
        cpu_fixed_matrix<float, TokenID, EmbeddingDimension> m_adam_first;
        cpu_fixed_matrix<float, TokenID, EmbeddingDimension> m_adam_second;

        void reset_embeddings();

        // Per-call state for propagate_backward (moved out of the parallel loop to fix data race)
        // These are class members so each InputLayer instance has its own copy — no static shared state.
        cpu_fixed_vector<uint16_t, TokenID> m_updated_tokens;
        cpu_fixed_vector<ConflictingToken, ConflictIndex> m_conflicts;

        // GPU-side copy of the input line, synced from CpuInputLine before OFFLOAD regions.
        GpuInputLine m_gpu_input;

        void check_nan_finding_mode_embeddings(const char* phase) const;
        void check_nan_finding_mode_matrix(
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& values,
            PositionIndex rows,
            const char* name,
            const char* phase
        ) const;
    };

} // namespace rllm
