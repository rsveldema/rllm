#pragma once

#include <LayerPrimitives.hpp>
#include <matmul.hpp>
#include <nlohmann/json_fwd.hpp>
#include <vector>

namespace rllm
{
    // A single transformer decoder block:
    //   Pre-RMSNorm → causal multi-head self-attention → residual
    //   Pre-RMSNorm → SwiGLU feed-forward network       → residual
    //
    // Hyperparameters (compile-time):
    //   D_MODEL   = EmbeddingDimension::MAX  = 512
    //   NUM_HEADS = static_cast<int>(HeadsIndex::MAX) = 8   →   HeadDimension::MAX = 64
    //   static_cast<int>(FFDimension::MAX)      = D_MODEL * 4              = 2048
    //
    // All weight matrices are stored [out × in] row-major on the heap.
    // Optimizer: SGD + momentum (β=0.9), gradient clip ±1, vel clip ±0.1, weight clamp ±2.
    class TransformerBlock
    {
      public:
        TransformerBlock() = default;
        ~TransformerBlock() = default;
        TransformerBlock(TransformerBlock&&) = default;
        TransformerBlock& operator=(TransformerBlock&&) = default;
        TransformerBlock(const TransformerBlock&) = delete;
        TransformerBlock& operator=(const TransformerBlock&) = delete;

        // Forward pass.  h[seq_len × D_MODEL] is modified in-place.
        // Caches intermediate activations for the backward pass.
        void forward(flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len);

        // Backward pass.  dout[seq_len × D_MODEL] = dL/dh_out.
        // Writes dL/dh_in into din (same shape) and updates all weights.
        void backward(
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dout,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& din,
            float learning_rate
        );

        void randomize();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        // Optimizer hyper-parameters
        static constexpr float MOMENTUM_BETA = 0.9f;
        static constexpr float GRAD_CLIP = 1.0f;
        static constexpr float VEL_CLIP = 0.1f;
        static constexpr float WEIGHT_CLAMP = 2.0f;

        // Attention weights [D_MODEL × D_MODEL] (out_dim × in_dim), row-major
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> W_q, W_k, W_v, W_o;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> V_q, V_k, V_v, V_o;

        // SwiGLU FFN:
        //   gate, up:  [static_cast<int>(FFDimension::MAX)    × D_MODEL]  (out × in)
        //   down:      [D_MODEL × static_cast<int>(FFDimension::MAX)   ]  (out × in)
        fixed_size_matrix<float, FFDimension, EmbeddingDimension> W_gate, W_up;
        fixed_size_matrix<float, EmbeddingDimension, FFDimension> W_down;
        fixed_size_matrix<float, FFDimension, EmbeddingDimension> V_gate, V_up;
        fixed_size_matrix<float, EmbeddingDimension, FFDimension> V_down;

        // Activations cached during forward() for use in backward().
        PositionIndex m_seq_len{PositionIndex::START};
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_h_in; // [T × D] input to this block
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_h_norm_attn; // [T × D] after 1st RMSNorm
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_Q, m_K,
            m_V; // [T × D] projected queries/keys/values
        // Per-head softmax weight matrices; each matrix is [PositionIndex::MAX × PositionIndex::MAX].
        // Only the top-left [T × T] block is live; columns are accessed with stride PositionIndex::MAX.
        template_vector<flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>, HeadsIndex> m_attn_w;
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>
            m_attn_concat; // [T × D] concatenated per-head outputs
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_h_mid; // [T × D] after attention residual
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> m_h_norm_ff; // [T × D] after 2nd RMSNorm
        flexible_rows_matrix<float, PositionIndex, FFDimension>
            m_gate_pre; // [T × static_cast<int>(FFDimension::MAX)] pre-activation gate branch
        flexible_rows_matrix<float, PositionIndex, FFDimension>
            m_up_pre; // [T × static_cast<int>(FFDimension::MAX)] pre-activation up branch
        flexible_rows_matrix<float, PositionIndex, FFDimension>
            m_ffn_act; // [T × static_cast<int>(FFDimension::MAX)] silu(gate_pre) * up_pre

        // RMSNorm:  for each row t → y_t = x_t / rms(x_t)
        static void rms_norm(
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& y
        );

        // RMSNorm backward: dx += ∂L/∂x  given dy = ∂L/∂y and the original x
        static void rms_norm_backward(
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dy,
            const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& dx
        );

        // In-place causal softmax over the active [T × T] block of x.
        static void causal_softmax(flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& x,
          PositionIndex T);

        // Accumulates softmax backward into dscores (stride T).
        // dp is the per-head d_scores matrix; p is the cached per-head softmax matrix.
        static void softmax_backward(
            const flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dp,
            const flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& p,
            flexible_rows_cols_matrix<float, PositionIndex, PositionIndex>& dscores,
            PositionIndex T
        );

        // SwiGLU backward: computes d_gate_pre and d_up_pre from d_ffn_act.
        static void swiglu_backward(
            PositionIndex seq,
            const flexible_rows_matrix<float, PositionIndex, FFDimension>& gate_pre,
            const flexible_rows_matrix<float, PositionIndex, FFDimension>& up_pre,
            const flexible_rows_matrix<float, PositionIndex, FFDimension>& d_ffn_act,
            flexible_rows_matrix<float, PositionIndex, FFDimension>& d_gate_pre,
            flexible_rows_matrix<float, PositionIndex, FFDimension>& d_up_pre
        );

        // SGD + momentum update: clips gradients, clips velocity, clamps weights.
        template <typename R_enum, typename C_enum>
        static void sgd_update(
            fixed_size_matrix<float, R_enum, C_enum>& W,
            fixed_size_matrix<float, R_enum, C_enum>& vel,
            const fixed_size_matrix<float, R_enum, C_enum>& grad,
            float lr
        )
        {
#pragma omp parallel for schedule(static)
            for (const auto [r, c] : enum_iterator2D<R_enum, C_enum>())
            {
                const float g = std::clamp(grad[r, c], -GRAD_CLIP, GRAD_CLIP);
                vel[r, c] = std::clamp(MOMENTUM_BETA * vel[r, c] + lr * g, -VEL_CLIP, VEL_CLIP);
                W[r, c] = std::clamp(W[r, c] + vel[r, c], -WEIGHT_CLAMP, WEIGHT_CLAMP);
            }
        }
    };

} // namespace rllm
