#include <JsonTensorHelpers.hpp>
#include <RandomHelpers.hpp>
#include <TransformerBlock.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>
#include <omp.h>

namespace rllm
{
    static_assert(
        TransformerBlock::D_MODEL == static_cast<int>(EmbeddingDimension::MAX),
        "D_MODEL must match EmbeddingDimension::MAX"
    );
    static_assert(
        TransformerBlock::D_MODEL % TransformerBlock::NUM_HEADS == 0,
        "D_MODEL must be divisible by NUM_HEADS"
    );

    static constexpr float MOMENTUM_BETA = 0.9f;
    static constexpr float GRAD_CLIP = 1.0f;
    static constexpr float VEL_CLIP = 0.1f;
    static constexpr float WEIGHT_CLAMP = 2.0f;

    // ── randomize ─────────────────────────────────────────────────────────────

    void TransformerBlock::randomize()
    {
        const float sd = 1.0f / std::sqrt(static_cast<float>(D_MODEL));
        const float sf = 1.0f / std::sqrt(static_cast<float>(D_FF));
        const int dd = D_MODEL * D_MODEL;
        const int dff_d = D_FF * D_MODEL;

        auto fill_rand = [](float* p, int n, float lo, float hi) {
            for (int i = 0; i < n; ++i)
                p[i] = get_random_value(lo, hi);
        };

        fill_rand(W_q.data(), dd, -sd, sd);
        fill_rand(W_k.data(), dd, -sd, sd);
        fill_rand(W_v.data(), dd, -sd, sd);
        fill_rand(W_o.data(), dd, -sd, sd);
        fill_rand(W_gate.data(), dff_d, -sd, sd);
        fill_rand(W_up.data(), dff_d, -sd, sd);
        fill_rand(W_down.data(), dff_d, -sf, sf);

        V_q.fill(0.f);
        V_k.fill(0.f);
        V_v.fill(0.f);
        V_o.fill(0.f);
        V_gate.fill(0.f);
        V_up.fill(0.f);
        V_down.fill(0.f);
    }

    // ── matrix helpers ────────────────────────────────────────────────────────

    // C[m, n] = A[m, k] @ B[n, k]^T   (B stored row-major [n × k])
    // C[i, j] = sum_l  A[i*k+l] * B[j*k+l]
    void TransformerBlock::matmul_ABt(const float* A, const float* B, float* C, int m, int n, int k)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.f;
#pragma omp simd reduction(+ : sum)
                for (int l = 0; l < k; ++l)
                    sum += A[i * k + l] * B[j * k + l];
                C[i * n + j] = sum;
            }
        }
    }

    // C[m, n] = A[m, k] @ B[k, n]     (standard; B stored row-major [k × n])
    // C[i, j] = sum_l  A[i*k+l] * B[l*n+j]
    void TransformerBlock::matmul_AB(const float* A, const float* B, float* C, int m, int n, int k)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.f;
                for (int l = 0; l < k; ++l)
                    sum += A[i * k + l] * B[l * n + j];
                C[i * n + j] = sum;
            }
        }
    }

    // C[m, n] += A^T @ B  where A is stored row-major [k × m], B is [k × n]
    // C[i, j] += sum_l  A[l*m+i] * B[l*n+j]
    void TransformerBlock::matmul_AtB_acc(const float* A, const float* B, float* C, int m, int n, int k)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.f;
                for (int l = 0; l < k; ++l)
                    sum += A[l * m + i] * B[l * n + j];
                C[i * n + j] += sum;
            }
        }
    }

    // ── normalisation ──────────────────────────────────────────────────────────

    // y_t = x_t / rms(x_t)  for each row t
    void TransformerBlock::rms_norm(const float* x, float* y, int T, int d)
    {
        const float eps = 1e-6f;
        for (int t = 0; t < T; ++t)
        {
            const float* xr = x + t * d;
            float* yr = y + t * d;
            float sq = 0.f;
#pragma omp simd reduction(+ : sq)
            for (int i = 0; i < d; ++i)
                sq += xr[i] * xr[i];
            const float inv = 1.0f / std::sqrt(sq / static_cast<float>(d) + eps);
#pragma omp simd
            for (int i = 0; i < d; ++i)
                yr[i] = xr[i] * inv;
        }
    }

    // dx += dL/dx  given dy = dL/dy and the original x (not the normalised y).
    // Per row:  dx_j += (1/rms) * (dy_j  -  y_j * mean(dy · y))
    void TransformerBlock::rms_norm_backward(const float* dy, const float* x, float* dx, int T, int d)
    {
        const float eps = 1e-6f;
        const float fd = static_cast<float>(d);
        for (int t = 0; t < T; ++t)
        {
            const float* xr = x + t * d;
            const float* dyr = dy + t * d;
            float* dxr = dx + t * d;

            float sq = 0.f;
#pragma omp simd reduction(+ : sq)
            for (int i = 0; i < d; ++i)
                sq += xr[i] * xr[i];
            const float inv = 1.0f / std::sqrt(sq / fd + eps);

            // dot = mean(dy · y) = (1/d) * sum_i dy[i] * x[i] * inv
            float dot = 0.f;
#pragma omp simd reduction(+ : dot)
            for (int i = 0; i < d; ++i)
                dot += dyr[i] * xr[i] * inv;
            dot /= fd;

#pragma omp simd
            for (int i = 0; i < d; ++i)
                dxr[i] += inv * (dyr[i] - xr[i] * inv * dot);
        }
    }

    // ── attention helpers ──────────────────────────────────────────────────────

    // In-place causal softmax over a [T × T] score matrix (row i only attends
    // to positions j ≤ i).  stride is the distance between rows in floats.
    void TransformerBlock::causal_softmax(flexible_size_matrix<float, PositionIndex, PositionIndex>& x, int T)
    {
        for (const auto i : enum_iterator<PositionIndex>(static_cast<PositionIndex>(T)))
        {
            float max_val = x[i, PositionIndex::START];
            for (const auto j : enum_iterator<PositionIndex>(inc(PositionIndex::START), inc(i)))
                max_val = std::max(max_val, x[i, j]);
            float sum_exp = 0.f;
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
            {
                x[i, j] = std::exp(x[i, j] - max_val);
                sum_exp += x[i, j];
            }
            const float inv = 1.0f / sum_exp;
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                x[i, j] *= inv;
            for (const auto j : enum_iterator<PositionIndex>(inc(i), static_cast<PositionIndex>(T)))
                x[i, j] = 0.f;
        }
    }

    // dscores[T×T] += ∂L/∂scores  via the softmax Jacobian.
    // dp/dscores use stride T; p uses p_stride (the cached matrix's row stride).
    void TransformerBlock::softmax_backward(
        const flexible_size_matrix<float, PositionIndex, PositionIndex>& dp,
        const flexible_size_matrix<float, PositionIndex, PositionIndex>& p,
        flexible_size_matrix<float, PositionIndex, PositionIndex>& dscores,
        int T
    )
    {
        for (const auto i : enum_iterator<PositionIndex>(static_cast<PositionIndex>(T)))
        {
            float dot = 0.f;
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                dot += dp[i, j] * p[i, j];
            for (const auto j : enum_iterator<PositionIndex>(inc(i)))
                dscores[i, j] += p[i, j] * (dp[i, j] - dot);
        }
    }

    // ── SGD + momentum ─────────────────────────────────────────────────────────

    void TransformerBlock::sgd_update(float* W, float* vel, const float* grad, int n, float lr)
    {
#pragma omp simd
        for (int i = 0; i < n; ++i)
        {
            const float g = std::clamp(grad[i], -GRAD_CLIP, GRAD_CLIP);
            vel[i] = std::clamp(MOMENTUM_BETA * vel[i] + lr * g, -VEL_CLIP, VEL_CLIP);
            W[i] = std::clamp(W[i] + vel[i], -WEIGHT_CLAMP, WEIGHT_CLAMP);
        }
    }

    // ── forward pass ──────────────────────────────────────────────────────────

    void
    TransformerBlock::forward(flexible_size_matrix<float, PositionIndex, EmbeddingDimension>& h, PositionIndex seq_len)
    {
        const int T = static_cast<int>(seq_len);
        const int D = D_MODEL;
        const int H = NUM_HEADS;
        const int Dh = HEAD_DIM;
        const int Dff = D_FF;

        m_seq_len = seq_len;
        std::copy(h.data(), h.data() + T * D, m_h_in.data()); // cache for backward

        // ── 1. Pre-norm (attention) ──────────────────────────────────────────
        rms_norm(h.data(), m_h_norm_attn.data(), T, D);

        // ── 2. Q / K / V projections ─────────────────────────────────────────
        matmul_ABt(m_h_norm_attn.data(), W_q.data(), m_Q.data(), T, D, D);
        matmul_ABt(m_h_norm_attn.data(), W_k.data(), m_K.data(), T, D, D);
        matmul_ABt(m_h_norm_attn.data(), W_v.data(), m_V.data(), T, D, D);

        // ── 3. Multi-head causal self-attention ──────────────────────────────
        const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        m_attn_concat.fill(0.f);

        for (int hi = 0; hi < H; ++hi)
        {
            auto& scores_mat = m_attn_w[static_cast<HeadsIndex>(hi)];
            scores_mat.set_size(static_cast<PositionIndex>(T), static_cast<PositionIndex>(T));

            // scores_mat[i, j] = Q[i, hi*Dh..] · K[j, hi*Dh..] * scale
            for (int i = 0; i < T; ++i)
            {
                const float* qi = m_Q.data() + i * D + hi * Dh;
                for (int j = 0; j < T; ++j)
                {
                    const float* kj = m_K.data() + j * D + hi * Dh;
                    float dot = 0.f;
#pragma omp simd reduction(+ : dot)
                    for (int d = 0; d < Dh; ++d)
                        dot += qi[d] * kj[d];
                    scores_mat.set(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j), dot * scale);
                }
            }
            causal_softmax(scores_mat, T);

            // attn_concat[i, hi*Dh..] = sum_j scores_mat[i,j] * V[j, hi*Dh..]
            for (int i = 0; i < T; ++i)
            {
                float* out = m_attn_concat.data() + i * D + hi * Dh;
                std::fill(out, out + Dh, 0.f);
                for (int j = 0; j <= i; ++j)
                {
                    const float* vj = m_V.data() + j * D + hi * Dh;
                    const float w = scores_mat[static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)];
#pragma omp simd
                    for (int d = 0; d < Dh; ++d)
                        out[d] += w * vj[d];
                }
            }
        }

        // ── 4. Output projection + residual ──────────────────────────────────
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension> attn_proj;
        matmul_ABt(m_attn_concat.data(), W_o.data(), attn_proj.data(), T, D, D);

#pragma omp simd
        for (int i = 0; i < T * D; ++i)
            m_h_mid.data()[i] = h.data()[i] + attn_proj.data()[i];

        // ── 5. Pre-norm (FFN) ─────────────────────────────────────────────────

        rms_norm(m_h_mid.data(), m_h_norm_ff.data(), T, D);

        // ── 6. SwiGLU FFN ─────────────────────────────────────────────────────
        matmul_ABt(m_h_norm_ff.data(), W_gate.data(), m_gate_pre.data(), T, Dff, D);
        matmul_ABt(m_h_norm_ff.data(), W_up.data(), m_up_pre.data(), T, Dff, D);

#pragma omp simd
        for (int i = 0; i < T * Dff; ++i)
        {
            const float g = m_gate_pre.data()[i];
            const float silu = g / (1.0f + std::exp(-g));
            m_ffn_act.data()[i] = silu * m_up_pre.data()[i];
        }

        fixed_size_matrix<float, PositionIndex, EmbeddingDimension> ffn_out;
        matmul_ABt(m_ffn_act.data(), W_down.data(), ffn_out.data(), T, D, Dff);

        // ── 7. Residual ───────────────────────────────────────────────────────
#pragma omp simd
        for (int i = 0; i < T * D; ++i)
            h.data()[i] = m_h_mid.data()[i] + ffn_out.data()[i];
    }

    // ── backward pass ─────────────────────────────────────────────────────────

    void TransformerBlock::backward(
        const flexible_size_matrix<float, PositionIndex, EmbeddingDimension>& dout,
        flexible_size_matrix<float, PositionIndex, EmbeddingDimension>& din,
        float learning_rate
    )
    {
        const int T = static_cast<int>(m_seq_len);
        const int D = D_MODEL;
        const int H = NUM_HEADS;
        const int Dh = HEAD_DIM;
        const int Dff = D_FF;

        // ── FFN backward ──────────────────────────────────────────────────────
        // h_out = h_mid + ffn_out  → d_h_mid += dout,  d_ffn_out = dout (same buffer)
        flexible_size_matrix<float, PositionIndex, EmbeddingDimension> d_h_mid(
            static_cast<PositionIndex>(T), EmbeddingDimension::MAX
        );
        std::copy(dout.data(), dout.data() + T * D, d_h_mid.data());

        // d_ffn_act = d_ffn_out @ W_down   (W_down[D × Dff])
        // C[T, Dff] = A[T, D] @ B[D, Dff]
        std::vector<float> d_ffn_act(T * Dff, 0.f);
        matmul_AB(dout.data(), W_down.data(), d_ffn_act.data(), T, Dff, D);

        // dW_down[D, Dff] += dout^T @ ffn_act
        std::vector<float> dW_down(D * Dff, 0.f);
        matmul_AtB_acc(dout.data(), m_ffn_act.data(), dW_down.data(), D, Dff, T);

        // SwiGLU backward: d(silu(g)*u) / dg, du
        std::vector<float> d_gate_pre(T * Dff);
        std::vector<float> d_up_pre(T * Dff);
#pragma omp simd
        for (int i = 0; i < T * Dff; ++i)
        {
            const float g = m_gate_pre.data()[i];
            const float sg = 1.0f / (1.0f + std::exp(-g)); // sigma(g), 0 when g << 0
            const float silu = g * sg;
            // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
            // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
            const float dsilu_dg = sg * (1.0f + g * (1.0f - sg));
            d_gate_pre[i] = d_ffn_act[i] * m_up_pre.data()[i] * dsilu_dg;
            d_up_pre[i] = d_ffn_act[i] * silu;
        }

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        fixed_size_matrix<float, PositionIndex, EmbeddingDimension> d_h_norm_ff;
        d_h_norm_ff.fill(0.f);
        {
            std::vector<float> tmp(T * D);
            matmul_AB(d_gate_pre.data(), W_gate.data(), tmp.data(), T, D, Dff);
#pragma omp simd
            for (int i = 0; i < T * D; ++i)
                d_h_norm_ff.data()[i] += tmp[i];
            matmul_AB(d_up_pre.data(), W_up.data(), tmp.data(), T, D, Dff);
#pragma omp simd
            for (int i = 0; i < T * D; ++i)
                d_h_norm_ff.data()[i] += tmp[i];
        }

        // weight gradients for gate, up
        std::vector<float> dW_gate(Dff * D, 0.f);
        std::vector<float> dW_up(Dff * D, 0.f);
        matmul_AtB_acc(d_gate_pre.data(), m_h_norm_ff.data(), dW_gate.data(), Dff, D, T);
        matmul_AtB_acc(d_up_pre.data(), m_h_norm_ff.data(), dW_up.data(), Dff, D, T);

        // RMSNorm backward for FFN: d_h_mid += rms_bwd(d_h_norm_ff, h_mid)
        rms_norm_backward(d_h_norm_ff.data(), m_h_mid.data(), d_h_mid.data(), T, D);

        // ── Attention backward ─────────────────────────────────────────────────
        // h_mid = h_in + attn_proj  → d_attn_proj = d_h_mid (passed through residual)
        //                             d_h_in (residual part) = d_h_mid (accumulated below)

        // d_attn_concat = d_attn_proj @ W_o
        std::vector<float> d_attn_concat(T * D, 0.f);
        matmul_AB(d_h_mid.data(), W_o.data(), d_attn_concat.data(), T, D, D);

        std::vector<float> dW_o(D * D, 0.f);
        matmul_AtB_acc(d_h_mid.data(), m_attn_concat.data(), dW_o.data(), D, D, T);

        // Per-head backward
        std::vector<float> d_Q(T * D, 0.f);
        std::vector<float> d_K(T * D, 0.f);
        std::vector<float> d_V(T * D, 0.f);

        for (int hi = 0; hi < H; ++hi)
        {
            const auto& scores_mat = m_attn_w[static_cast<HeadsIndex>(hi)];
            const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

            // d_V_h[j, :] += sum_i scores_mat[i,j] * d_attn_concat[i, hi*Dh..]
            for (int j = 0; j < T; ++j)
            {
                float* dvj = d_V.data() + j * D + hi * Dh;
                for (int i = j; i < T; ++i) // only non-masked rows
                {
                    const float* do_i = d_attn_concat.data() + i * D + hi * Dh;
                    const float w = scores_mat[static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)];
#pragma omp simd
                    for (int d = 0; d < Dh; ++d)
                        dvj[d] += w * do_i[d];
                }
            }

            // d_scores[i,j] = d_attn_concat[i, hi*Dh..] · V[j, hi*Dh..]
            flexible_size_matrix<float, PositionIndex, PositionIndex> d_scores(
                static_cast<PositionIndex>(T), static_cast<PositionIndex>(T)
            );
            for (int i = 0; i < T; ++i)
            {
                const float* do_i = d_attn_concat.data() + i * D + hi * Dh;
                for (int j = 0; j <= i; ++j)
                {
                    const float* vj = m_V.data() + j * D + hi * Dh;
                    float dot = 0.f;
#pragma omp simd reduction(+ : dot)
                    for (int d = 0; d < Dh; ++d)
                        dot += do_i[d] * vj[d];
                    d_scores.set(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j), dot);
                }
            }

            // Backward through causal softmax
            flexible_size_matrix<float, PositionIndex, PositionIndex> d_raw(
                static_cast<PositionIndex>(T), static_cast<PositionIndex>(T)
            );
            softmax_backward(d_scores, scores_mat, d_raw, T);

            // d_Q and d_K
            for (int i = 0; i < T; ++i)
            {
                float* dqi = d_Q.data() + i * D + hi * Dh;
                for (int j = 0; j <= i; ++j)
                {
                    const float* kj = m_K.data() + j * D + hi * Dh;
                    const float ds = d_raw.get(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)) * scale;
#pragma omp simd
                    for (int d = 0; d < Dh; ++d)
                        dqi[d] += ds * kj[d];
                }
            }
            for (int j = 0; j < T; ++j)
            {
                float* dkj = d_K.data() + j * D + hi * Dh;
                for (int i = j; i < T; ++i)
                {
                    const float* qi = m_Q.data() + i * D + hi * Dh;
                    const float ds = d_raw.get(static_cast<PositionIndex>(i), static_cast<PositionIndex>(j)) * scale;
#pragma omp simd
                    for (int d = 0; d < Dh; ++d)
                        dkj[d] += ds * qi[d];
                }
            }
        }

        // Weight gradients for W_q, W_k, W_v
        std::vector<float> dW_q(D * D, 0.f);
        std::vector<float> dW_k(D * D, 0.f);
        std::vector<float> dW_v(D * D, 0.f);
        matmul_AtB_acc(d_Q.data(), m_h_norm_attn.data(), dW_q.data(), D, D, T);
        matmul_AtB_acc(d_K.data(), m_h_norm_attn.data(), dW_k.data(), D, D, T);
        matmul_AtB_acc(d_V.data(), m_h_norm_attn.data(), dW_v.data(), D, D, T);

        // d_h_norm_attn = d_Q @ W_q  +  d_K @ W_k  +  d_V @ W_v
        std::vector<float> d_h_norm_attn(T * D, 0.f);
        {
            std::vector<float> tmp(T * D);
            matmul_AB(d_Q.data(), W_q.data(), tmp.data(), T, D, D);
#pragma omp simd
            for (int i = 0; i < T * D; ++i)
                d_h_norm_attn[i] += tmp[i];
            matmul_AB(d_K.data(), W_k.data(), tmp.data(), T, D, D);
#pragma omp simd
            for (int i = 0; i < T * D; ++i)
                d_h_norm_attn[i] += tmp[i];
            matmul_AB(d_V.data(), W_v.data(), tmp.data(), T, D, D);
#pragma omp simd
            for (int i = 0; i < T * D; ++i)
                d_h_norm_attn[i] += tmp[i];
        }

        // d_h_in: residual from d_h_mid + RMSNorm backward
        std::copy(d_h_mid.data(), d_h_mid.data() + T * D, din.data());
        rms_norm_backward(d_h_norm_attn.data(), m_h_in.data(), din.data(), T, D);

        // ── Weight updates ─────────────────────────────────────────────────────
        const int dd = D_MODEL * D_MODEL;
        const int dff_d = D_FF * D_MODEL;
        sgd_update(W_q.data(), V_q.data(), dW_q.data(), dd, learning_rate);
        sgd_update(W_k.data(), V_k.data(), dW_k.data(), dd, learning_rate);
        sgd_update(W_v.data(), V_v.data(), dW_v.data(), dd, learning_rate);
        sgd_update(W_o.data(), V_o.data(), dW_o.data(), dd, learning_rate);
        sgd_update(W_gate.data(), V_gate.data(), dW_gate.data(), dff_d, learning_rate);
        sgd_update(W_up.data(), V_up.data(), dW_up.data(), dff_d, learning_rate);
        sgd_update(W_down.data(), V_down.data(), dW_down.data(), dff_d, learning_rate);
    }

    // ── serialisation ──────────────────────────────────────────────────────────

    nlohmann::json TransformerBlock::save() const
    {
        using namespace json_helpers;
        return {
            {"W_q", serialize_matrix(W_q)},
            {"W_k", serialize_matrix(W_k)},
            {"W_v", serialize_matrix(W_v)},
            {"W_o", serialize_matrix(W_o)},
            {"W_gate", serialize_matrix(W_gate)},
            {"W_up", serialize_matrix(W_up)},
            {"W_down", serialize_matrix(W_down)},
        };
    }

    void TransformerBlock::load(const nlohmann::json& j)
    {
        using namespace json_helpers;
        deserialize_matrix(j.at("W_q"), W_q);
        deserialize_matrix(j.at("W_k"), W_k);
        deserialize_matrix(j.at("W_v"), W_v);
        deserialize_matrix(j.at("W_o"), W_o);
        deserialize_matrix(j.at("W_gate"), W_gate);
        deserialize_matrix(j.at("W_up"), W_up);
        deserialize_matrix(j.at("W_down"), W_down);

        // Reset momentum on load — do not persist transient training state
        V_q.fill(0.f);
        V_k.fill(0.f);
        V_v.fill(0.f);
        V_o.fill(0.f);
        V_gate.fill(0.f);
        V_up.fill(0.f);
        V_down.fill(0.f);
    }

} // namespace rllm
