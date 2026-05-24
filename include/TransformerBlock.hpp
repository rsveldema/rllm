#pragma once

#include <LayerPrimitives.hpp>
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
    //   NUM_HEADS = 8   →   HEAD_DIM = 64
    //   D_FF      = D_MODEL * 4              = 2048
    //
    // All weight matrices are stored [out × in] row-major on the heap.
    // Optimizer: SGD + momentum (β=0.9), gradient clip ±1, vel clip ±0.1, weight clamp ±2.
    class TransformerBlock
    {
      public:
        static constexpr int D_MODEL   = static_cast<int>(EmbeddingDimension::MAX); // 512
        static constexpr int NUM_HEADS = 8;
        static constexpr int HEAD_DIM  = D_MODEL / NUM_HEADS; // 64
        static constexpr int D_FF      = D_MODEL * 4;         // 2048

        TransformerBlock() = default;
        ~TransformerBlock() = default;
        TransformerBlock(TransformerBlock&&) = default;
        TransformerBlock& operator=(TransformerBlock&&) = default;
        TransformerBlock(const TransformerBlock&) = delete;
        TransformerBlock& operator=(const TransformerBlock&) = delete;

        // Forward pass.  h[seq_len × D_MODEL] is modified in-place.
        // Caches intermediate activations for the backward pass.
        void forward(std::vector<float>& h, int seq_len);

        // Backward pass.  dout[seq_len × D_MODEL] = dL/dh_out.
        // Returns dL/dh_in of the same shape and updates all weights.
        std::vector<float> backward(const std::vector<float>& dout, float learning_rate);

        void randomize();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      private:
        // Attention weights [D_MODEL × D_MODEL] (out_dim × in_dim), row-major
        template_token_matrix<float, EmbeddingDimension, EmbeddingDimension> W_q, W_k, W_v, W_o;
        template_token_matrix<float, EmbeddingDimension, EmbeddingDimension> V_q, V_k, V_v, V_o;

        // SwiGLU FFN:
        //   gate, up:  [D_FF    × D_MODEL]  (out × in)
        //   down:      [D_MODEL × D_FF   ]  (out × in)
        template_token_matrix<float, FFDimension,        EmbeddingDimension> W_gate, W_up;
        template_token_matrix<float, EmbeddingDimension, FFDimension>        W_down;
        template_token_matrix<float, FFDimension,        EmbeddingDimension> V_gate, V_up;
        template_token_matrix<float, EmbeddingDimension, FFDimension>        V_down;

        // Activations cached during forward() for use in backward().
        int                m_seq_len{0};
        std::vector<float> m_h_in;          // [T × D] input to this block
        std::vector<float> m_h_norm_attn;   // [T × D] after 1st RMSNorm
        template_token_matrix<float, PositionIndex, EmbeddingDimension> m_Q, m_K, m_V;  // [T × D] projected queries/keys/values
        std::vector<float> m_attn_w;        // [NUM_HEADS × T × T] softmax attention weights
        std::vector<float> m_attn_concat;   // [T × D] concatenated per-head outputs
        std::vector<float> m_h_mid;         // [T × D] after attention residual
        std::vector<float> m_h_norm_ff;     // [T × D] after 2nd RMSNorm
        std::vector<float> m_gate_pre;      // [T × D_FF] pre-activation gate branch
        std::vector<float> m_up_pre;        // [T × D_FF] pre-activation up branch
        std::vector<float> m_ffn_act;       // [T × D_FF] silu(gate_pre) * up_pre

        // ── linear algebra helpers ──────────────────────────────────────────
        // C[m,n]  = A[m,k] @ B[n,k]^T   (B stored row-major [n × k])
        static void matmul_ABt(const float* A, const float* B, float* C,
                               int m, int n, int k);

        // C[m,n]  = A[m,k] @ B[k,n]     (standard; B stored row-major [k × n])
        static void matmul_AB(const float* A, const float* B, float* C,
                              int m, int n, int k);

        // C[m,n] += A^T[m,k] @ B[k,n]   (A provided row-major [k × m]; accumulates into C)
        static void matmul_AtB_acc(const float* A, const float* B, float* C,
                                   int m, int n, int k);

        // RMSNorm:  for each row t → y_t = x_t / rms(x_t)
        static void rms_norm(const float* x, float* y, int T, int d);

        // RMSNorm backward: dx += ∂L/∂x  given dy = ∂L/∂y and the original x
        static void rms_norm_backward(const float* dy, const float* x,
                                      float* dx, int T, int d);

        // In-place causal softmax over a [T × T] score matrix.
        static void causal_softmax(float* x, int T);

        // Accumulates softmax backward into dscores.
        // dp[T×T]: ∂L/∂softmax_output.  p[T×T]: softmax values.
        static void softmax_backward(const float* dp, const float* p,
                                     float* dscores, int T);

        // SGD + momentum update: clips gradients, clips velocity, clamps weights.
        static void sgd_update(float* W, float* vel, const float* grad, int n, float lr);
    };

} // namespace rllm
