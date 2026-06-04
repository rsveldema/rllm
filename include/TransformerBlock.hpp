#pragma once

#include <LayerPrimitives.hpp>
#include <fixed_size_obj_vector.hpp>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <parallel.hpp>
#include <vector>

namespace rllm
{
    // Forward declaration for heap-allocated forward-pass activation workspace.
    struct ForwardWorkspace;

    // Shared scratch workspace for one TransformerBlock backward pass. Owned by
    // the training backward workspace and reused for each block in reverse order.
    struct BackwardWorkspace
    {
        // FFN backward
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_mid;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_ffn_act;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension> dW_down;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_gate_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension> d_up_pre;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_norm_ff;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> dW_gate;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> dW_up;

        // Attention backward
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_attn_concat;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_o;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_Q;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_K;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_V;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_q;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_k;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> dW_v;
        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> d_h_norm_attn;

        flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension> tmp;
        fixed_size_obj_vector<flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>, HeadsIndex>
            d_scores;
        fixed_size_obj_vector<flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>, HeadsIndex> d_raw;

        explicit BackwardWorkspace(PositionIndex seq)
            : d_h_mid(seq)
            , d_ffn_act(seq)
            , d_gate_pre(seq)
            , d_up_pre(seq)
            , d_h_norm_ff(seq)
            , d_attn_concat(seq)
            , d_Q(seq)
            , d_K(seq)
            , d_V(seq)
            , d_h_norm_attn(seq)
            , tmp(seq)
        {}

        void reset(PositionIndex seq)
        {
            d_h_mid.set_rows(seq);
            d_ffn_act.set_rows(seq);
            d_gate_pre.set_rows(seq);
            d_up_pre.set_rows(seq);
            d_h_norm_ff.set_rows(seq);
            d_attn_concat.set_rows(seq);
            d_Q.set_rows(seq);
            d_K.set_rows(seq);
            d_V.set_rows(seq);
            d_h_norm_attn.set_rows(seq);
            tmp.set_rows(seq);

            dW_down.zero();
            d_h_norm_ff.zero();
            dW_gate.zero();
            dW_up.zero();
            dW_o.zero();
            d_Q.zero();
            d_K.zero();
            d_V.zero();
            dW_q.zero();
            dW_k.zero();
            dW_v.zero();
            d_h_norm_attn.zero();
        }
    };

    // A single transformer decoder block:
    //   Pre-RMSNorm → causal multi-head self-attention → residual
    //   Pre-RMSNorm → SwiGLU feed-forward network       → residual
    //
    // Hyperparameters (compile-time):
    //   D_MODEL   = EmbeddingDimension::MAX  = 512
    //   HeadsIndex::MAX = 8   →   HeadDimension::MAX = 64
    //   FFDimension::MAX      = D_MODEL * 4              = 2048
    //
    // All weight matrices are stored [out × in] row-major on the heap.
    // Optimizer: SGD + momentum (β=0.9), gradient clip ±1, vel clip ±0.1, weight clamp ±2.
    class TransformerBlock
    {
      public:
        // Optimizer hyper-parameters
        static constexpr float MOMENTUM_BETA = 0.9f;
        static constexpr float GRAD_CLIP = 1.0f;
        static constexpr float VEL_CLIP = 0.1f;
        static constexpr float WEIGHT_CLAMP = 2.0f;

      public:
        TransformerBlock(); // defined in .cc after ForwardWorkspace is complete
        ~TransformerBlock(); // defined in .cc after ForwardWorkspace is complete
        TransformerBlock(TransformerBlock&&) noexcept; // defined in .cc
        TransformerBlock& operator=(TransformerBlock&&) noexcept; // defined in .cc
        TransformerBlock(const TransformerBlock&) = delete;
        TransformerBlock& operator=(const TransformerBlock&) = delete;

        // Forward pass.  h[seq_len × D_MODEL] is modified in-place.
        // Caches intermediate activations for the backward pass.
        void forward(flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len);

        // Backward pass.  dout[seq_len × D_MODEL] = dL/dh_out.
        // Writes dL/dh_in into din (same shape) and updates all weights.
        void backward(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dout,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& din,
            BackwardWorkspace& workspace,
            float learning_rate
        );

        void randomize();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

        // Public RMS norm: used by NeuralNetwork to apply the final pre-LM-head norm.
        static void apply_rms_norm(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& y
        )
        {
            rms_norm(x, y);
        }

        // Test helper: expose causal softmax without exposing internals broadly.
        static void
        causal_softmax_for_test(flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& x, PositionIndex T)
        {
            causal_softmax(x, T);
        }

        // Test helper: expose softmax backward Jacobian application.
        static void softmax_backward_for_test(
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dp,
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& p,
            flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dscores,
            PositionIndex T
        )
        {
            softmax_backward(dp, p, dscores, T);
        }


      private:
        // Attention weights [D_MODEL × D_MODEL] (out_dim × in_dim), row-major
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, EmbeddingDimension> W_q, W_k, W_v, W_o;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, EmbeddingDimension> V_q, V_k, V_v, V_o;

        // SwiGLU FFN:
        //   gate, up:  [FFDimension::MAX    × D_MODEL]  (out × in)
        //   down:      [D_MODEL × FFDimension::MAX   ]  (out × in)
        fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension> W_gate, W_up;
        fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> V_gate, V_up;
        fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension> W_down;
        fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension> V_down;

        // Activations cached during forward() for use in backward().
        // Heap-allocated to avoid blowing the stack (~21 MB of fixed-size arrays).
        std::unique_ptr<ForwardWorkspace> m_fwd_ws;

        void copy_weights_to_offload_buffer();

        // RMSNorm:  for each row t → y_t = x_t / rms(x_t)
        static void rms_norm(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& y
        );

        // RMSNorm backward: dx += ∂L/∂x  given dy = ∂L/∂y and the original x
        static void rms_norm_backward(
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dy,
            const flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& x,
            flexible_rows_matrix<rlmm_float, PositionIndex, EmbeddingDimension>& dx
        );

        // In-place causal softmax over the active [T × T] block of x.
        static void
        causal_softmax(flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& x, PositionIndex T);

        // Accumulates softmax backward into dscores (stride T).
        // dp is the per-head d_scores matrix; p is the cached per-head softmax matrix.
        static void softmax_backward(
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dp,
            const flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& p,
            flexible_rows_cols_matrix<rlmm_float, PositionIndex, PositionIndex>& dscores,
            PositionIndex T
        );

        // SwiGLU backward: computes d_gate_pre and d_up_pre from d_ffn_act.
        static void swiglu_backward(
            PositionIndex seq,
            const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& gate_pre,
            const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& up_pre,
            const flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_ffn_act,
            flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_gate_pre,
            flexible_rows_matrix<rlmm_float, PositionIndex, FFDimension>& d_up_pre
        );
    };

} // namespace rllm
