#include <TransformerBlock.hpp>
#include <JsonTensorHelpers.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>
#include <omp.h>

namespace rllm
{
    static_assert(TransformerBlock::D_MODEL == static_cast<int>(EmbeddingDimension::MAX),
                  "D_MODEL must match EmbeddingDimension::MAX");
    static_assert(TransformerBlock::D_MODEL % TransformerBlock::NUM_HEADS == 0,
                  "D_MODEL must be divisible by NUM_HEADS");

    static constexpr float MOMENTUM_BETA = 0.9f;
    static constexpr float GRAD_CLIP     = 1.0f;
    static constexpr float VEL_CLIP      = 0.1f;
    static constexpr float WEIGHT_CLAMP  = 2.0f;

    // ── randomize ─────────────────────────────────────────────────────────────

    void TransformerBlock::randomize()
    {
        const float sd    = 1.0f / std::sqrt(static_cast<float>(D_MODEL));
        const float sf    = 1.0f / std::sqrt(static_cast<float>(D_FF));
        const int   dd    = D_MODEL * D_MODEL;
        const int   dff_d = D_FF    * D_MODEL;

        auto fill_rand = [](float* p, int n, float lo, float hi) {
            for (int i = 0; i < n; ++i) p[i] = get_random_value(lo, hi);
        };

        fill_rand(W_q.data(),    dd,    -sd, sd);
        fill_rand(W_k.data(),    dd,    -sd, sd);
        fill_rand(W_v.data(),    dd,    -sd, sd);
        fill_rand(W_o.data(),    dd,    -sd, sd);
        fill_rand(W_gate.data(), dff_d, -sd, sd);
        fill_rand(W_up.data(),   dff_d, -sd, sd);
        fill_rand(W_down.data(), dff_d, -sf, sf);

        V_q.fill(0.f);    V_k.fill(0.f);    V_v.fill(0.f);    V_o.fill(0.f);
        V_gate.fill(0.f); V_up.fill(0.f);   V_down.fill(0.f);
    }

    // ── matrix helpers ────────────────────────────────────────────────────────

    // C[m, n] = A[m, k] @ B[n, k]^T   (B stored row-major [n × k])
    // C[i, j] = sum_l  A[i*k+l] * B[j*k+l]
    void TransformerBlock::matmul_ABt(const float* A, const float* B, float* C,
                                      int m, int n, int k)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.f;
#pragma omp simd reduction(+:sum)
                for (int l = 0; l < k; ++l)
                    sum += A[i * k + l] * B[j * k + l];
                C[i * n + j] = sum;
            }
        }
    }

    // C[m, n] = A[m, k] @ B[k, n]     (standard; B stored row-major [k × n])
    // C[i, j] = sum_l  A[i*k+l] * B[l*n+j]
    void TransformerBlock::matmul_AB(const float* A, const float* B, float* C,
                                     int m, int n, int k)
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
    void TransformerBlock::matmul_AtB_acc(const float* A, const float* B, float* C,
                                          int m, int n, int k)
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
#pragma omp simd reduction(+:sq)
            for (int i = 0; i < d; ++i) sq += xr[i] * xr[i];
            const float inv = 1.0f / std::sqrt(sq / static_cast<float>(d) + eps);
#pragma omp simd
            for (int i = 0; i < d; ++i) yr[i] = xr[i] * inv;
        }
    }

    // dx += dL/dx  given dy = dL/dy and the original x (not the normalised y).
    // Per row:  dx_j += (1/rms) * (dy_j  -  y_j * mean(dy · y))
    void TransformerBlock::rms_norm_backward(const float* dy, const float* x,
                                              float* dx, int T, int d)
    {
        const float eps = 1e-6f;
        const float fd = static_cast<float>(d);
        for (int t = 0; t < T; ++t)
        {
            const float* xr  = x  + t * d;
            const float* dyr = dy + t * d;
            float* dxr = dx + t * d;

            float sq = 0.f;
#pragma omp simd reduction(+:sq)
            for (int i = 0; i < d; ++i) sq += xr[i] * xr[i];
            const float inv = 1.0f / std::sqrt(sq / fd + eps);

            // dot = mean(dy · y) = (1/d) * sum_i dy[i] * x[i] * inv
            float dot = 0.f;
#pragma omp simd reduction(+:dot)
            for (int i = 0; i < d; ++i) dot += dyr[i] * xr[i] * inv;
            dot /= fd;

#pragma omp simd
            for (int i = 0; i < d; ++i)
                dxr[i] += inv * (dyr[i] - xr[i] * inv * dot);
        }
    }

    // ── attention helpers ──────────────────────────────────────────────────────

    // In-place causal softmax over a [T × T] score matrix (row i only attends
    // to positions j ≤ i).
    void TransformerBlock::causal_softmax(float* x, int T)
    {
        for (int i = 0; i < T; ++i)
        {
            float* row = x + i * T;
            float max_val = row[0];
            for (int j = 1; j <= i; ++j) max_val = std::max(max_val, row[j]);
            float sum_exp = 0.f;
            for (int j = 0; j <= i; ++j)
            {
                row[j] = std::exp(row[j] - max_val);
                sum_exp += row[j];
            }
            const float inv = 1.0f / sum_exp;
            for (int j = 0;     j <= i; ++j) row[j] *= inv;
            for (int j = i + 1; j <  T; ++j) row[j]  = 0.f;
        }
    }

    // dscores[T×T] += ∂L/∂scores  via the softmax Jacobian.
    // p[T×T]: softmax values (0 for masked positions).
    void TransformerBlock::softmax_backward(const float* dp, const float* p,
                                            float* dscores, int T)
    {
        for (int i = 0; i < T; ++i)
        {
            const float* dp_r = dp + i * T;
            const float* p_r  = p  + i * T;
            float* ds_r = dscores + i * T;
            float dot = 0.f;
            for (int j = 0; j <= i; ++j) dot += dp_r[j] * p_r[j];
            for (int j = 0; j <= i; ++j) ds_r[j] += p_r[j] * (dp_r[j] - dot);
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
            W[i]   = std::clamp(W[i] + vel[i], -WEIGHT_CLAMP, WEIGHT_CLAMP);
        }
    }

    // ── forward pass ──────────────────────────────────────────────────────────

    void TransformerBlock::forward(std::vector<float>& h, int seq_len)
    {
        const int T   = seq_len;
        const int D   = D_MODEL;
        const int H   = NUM_HEADS;
        const int Dh  = HEAD_DIM;
        const int Dff = D_FF;

        m_seq_len = T;
        m_h_in    = h; // cache for backward

        // ── 1. Pre-norm (attention) ──────────────────────────────────────────
        m_h_norm_attn.resize(T * D);
        rms_norm(h.data(), m_h_norm_attn.data(), T, D);

        // ── 2. Q / K / V projections ─────────────────────────────────────────
        m_Q.resize(T * D);  m_K.resize(T * D);  m_V.resize(T * D);
        matmul_ABt(m_h_norm_attn.data(), W_q.data(), m_Q.data(), T, D, D);
        matmul_ABt(m_h_norm_attn.data(), W_k.data(), m_K.data(), T, D, D);
        matmul_ABt(m_h_norm_attn.data(), W_v.data(), m_V.data(), T, D, D);

        // ── 3. Multi-head causal self-attention ──────────────────────────────
        const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        m_attn_w.assign(H * T * T, 0.f);
        m_attn_concat.assign(T * D, 0.f);

        for (int hi = 0; hi < H; ++hi)
        {
            float* scores = m_attn_w.data() + hi * T * T;

            // scores[i, j] = Q[i, hi*Dh..] · K[j, hi*Dh..] * scale
            for (int i = 0; i < T; ++i)
            {
                const float* qi = m_Q.data() + i * D + hi * Dh;
                for (int j = 0; j < T; ++j)
                {
                    const float* kj = m_K.data() + j * D + hi * Dh;
                    float dot = 0.f;
#pragma omp simd reduction(+:dot)
                    for (int d = 0; d < Dh; ++d) dot += qi[d] * kj[d];
                    scores[i * T + j] = dot * scale;
                }
            }
            causal_softmax(scores, T);

            // attn_concat[i, hi*Dh..] = sum_j scores[i,j] * V[j, hi*Dh..]
            for (int i = 0; i < T; ++i)
            {
                float* out       = m_attn_concat.data() + i * D + hi * Dh;
                const float* row = scores + i * T;
                std::fill(out, out + Dh, 0.f);
                for (int j = 0; j <= i; ++j)
                {
                    const float* vj = m_V.data() + j * D + hi * Dh;
                    const float   w  = row[j];
#pragma omp simd
                    for (int d = 0; d < Dh; ++d) out[d] += w * vj[d];
                }
            }
        }

        // ── 4. Output projection + residual ──────────────────────────────────
        std::vector<float> attn_proj(T * D);
        matmul_ABt(m_attn_concat.data(), W_o.data(), attn_proj.data(), T, D, D);

        m_h_mid.resize(T * D);
#pragma omp simd
        for (int i = 0; i < T * D; ++i) m_h_mid[i] = h[i] + attn_proj[i];

        // ── 5. Pre-norm (FFN) ─────────────────────────────────────────────────
        m_h_norm_ff.resize(T * D);
        rms_norm(m_h_mid.data(), m_h_norm_ff.data(), T, D);

        // ── 6. SwiGLU FFN ─────────────────────────────────────────────────────
        m_gate_pre.resize(T * Dff);  m_up_pre.resize(T * Dff);
        matmul_ABt(m_h_norm_ff.data(), W_gate.data(), m_gate_pre.data(), T, Dff, D);
        matmul_ABt(m_h_norm_ff.data(), W_up.data(),   m_up_pre.data(),   T, Dff, D);

        m_ffn_act.resize(T * Dff);
#pragma omp simd
        for (int i = 0; i < T * Dff; ++i)
        {
            const float g    = m_gate_pre[i];
            const float silu = g / (1.0f + std::exp(-g));
            m_ffn_act[i] = silu * m_up_pre[i];
        }

        std::vector<float> ffn_out(T * D);
        matmul_ABt(m_ffn_act.data(), W_down.data(), ffn_out.data(), T, D, Dff);

        // ── 7. Residual ───────────────────────────────────────────────────────
        h.resize(T * D);
#pragma omp simd
        for (int i = 0; i < T * D; ++i) h[i] = m_h_mid[i] + ffn_out[i];
    }

    // ── backward pass ─────────────────────────────────────────────────────────

    std::vector<float> TransformerBlock::backward(const std::vector<float>& dout,
                                                   float learning_rate)
    {
        const int T   = m_seq_len;
        const int D   = D_MODEL;
        const int H   = NUM_HEADS;
        const int Dh  = HEAD_DIM;
        const int Dff = D_FF;
        const int TT  = T * T;

        // ── FFN backward ──────────────────────────────────────────────────────
        // h_out = h_mid + ffn_out  → d_h_mid += dout,  d_ffn_out = dout (same buffer)
        std::vector<float> d_h_mid(dout); // copy; we'll add more to this below

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
            const float g    = m_gate_pre[i];
            const float sg   = 1.0f / (1.0f + std::exp(-g)); // sigma(g), 0 when g << 0
            const float silu = g * sg;
            // silu'(g) = sigma(g) * (1 + g * (1 - sigma(g)))
            // Avoids exp(-g)*sigma(g) which gives inf*0=NaN for g < -88 in float32.
            const float dsilu_dg = sg * (1.0f + g * (1.0f - sg));
            d_gate_pre[i] = d_ffn_act[i] * m_up_pre[i] * dsilu_dg;
            d_up_pre[i]   = d_ffn_act[i] * silu;
        }

        // d_h_norm_ff = d_gate_pre @ W_gate  +  d_up_pre @ W_up
        std::vector<float> d_h_norm_ff(T * D, 0.f);
        {
            std::vector<float> tmp(T * D);
            matmul_AB(d_gate_pre.data(), W_gate.data(), tmp.data(), T, D, Dff);
#pragma omp simd
            for (int i = 0; i < T * D; ++i) d_h_norm_ff[i] += tmp[i];
            matmul_AB(d_up_pre.data(), W_up.data(), tmp.data(), T, D, Dff);
#pragma omp simd
            for (int i = 0; i < T * D; ++i) d_h_norm_ff[i] += tmp[i];
        }

        // weight gradients for gate, up
        std::vector<float> dW_gate(Dff * D, 0.f);
        std::vector<float> dW_up(Dff * D, 0.f);
        matmul_AtB_acc(d_gate_pre.data(), m_h_norm_ff.data(), dW_gate.data(), Dff, D, T);
        matmul_AtB_acc(d_up_pre.data(),   m_h_norm_ff.data(), dW_up.data(),   Dff, D, T);

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
            const float* scores = m_attn_w.data() + hi * TT;
            const float   scale  = 1.0f / std::sqrt(static_cast<float>(Dh));

            // d_V_h[j, :] += sum_i scores[i,j] * d_attn_concat[i, hi*Dh..]
            for (int j = 0; j < T; ++j)
            {
                float* dvj = d_V.data() + j * D + hi * Dh;
                for (int i = j; i < T; ++i)   // only non-masked rows
                {
                    const float* do_i = d_attn_concat.data() + i * D + hi * Dh;
                    const float   w    = scores[i * T + j];
#pragma omp simd
                    for (int d = 0; d < Dh; ++d) dvj[d] += w * do_i[d];
                }
            }

            // d_scores[i,j] = d_attn_concat[i, hi*Dh..] · V[j, hi*Dh..]
            std::vector<float> d_scores(TT, 0.f);
            for (int i = 0; i < T; ++i)
            {
                const float* do_i = d_attn_concat.data() + i * D + hi * Dh;
                for (int j = 0; j <= i; ++j)
                {
                    const float* vj = m_V.data() + j * D + hi * Dh;
                    float dot = 0.f;
#pragma omp simd reduction(+:dot)
                    for (int d = 0; d < Dh; ++d) dot += do_i[d] * vj[d];
                    d_scores[i * T + j] = dot;
                }
            }

            // Backward through causal softmax
            std::vector<float> d_raw(TT, 0.f);
            softmax_backward(d_scores.data(), scores, d_raw.data(), T);

            // d_Q and d_K
            for (int i = 0; i < T; ++i)
            {
                float* dqi = d_Q.data() + i * D + hi * Dh;
                for (int j = 0; j <= i; ++j)
                {
                    const float* kj = m_K.data() + j * D + hi * Dh;
                    const float   ds = d_raw[i * T + j] * scale;
#pragma omp simd
                    for (int d = 0; d < Dh; ++d) dqi[d] += ds * kj[d];
                }
            }
            for (int j = 0; j < T; ++j)
            {
                float* dkj = d_K.data() + j * D + hi * Dh;
                for (int i = j; i < T; ++i)
                {
                    const float* qi = m_Q.data() + i * D + hi * Dh;
                    const float   ds = d_raw[i * T + j] * scale;
#pragma omp simd
                    for (int d = 0; d < Dh; ++d) dkj[d] += ds * qi[d];
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
            for (int i = 0; i < T * D; ++i) d_h_norm_attn[i] += tmp[i];
            matmul_AB(d_K.data(), W_k.data(), tmp.data(), T, D, D);
#pragma omp simd
            for (int i = 0; i < T * D; ++i) d_h_norm_attn[i] += tmp[i];
            matmul_AB(d_V.data(), W_v.data(), tmp.data(), T, D, D);
#pragma omp simd
            for (int i = 0; i < T * D; ++i) d_h_norm_attn[i] += tmp[i];
        }

        // d_h_in: residual from d_h_mid + RMSNorm backward
        std::vector<float> d_h_in(d_h_mid); // residual: h_mid = h_in + attn_proj
        rms_norm_backward(d_h_norm_attn.data(), m_h_in.data(), d_h_in.data(), T, D);

        // ── Weight updates ─────────────────────────────────────────────────────
        const int dd    = D_MODEL * D_MODEL;
        const int dff_d = D_FF    * D_MODEL;
        sgd_update(W_q.data(),    V_q.data(),    dW_q.data(),    dd,    learning_rate);
        sgd_update(W_k.data(),    V_k.data(),    dW_k.data(),    dd,    learning_rate);
        sgd_update(W_v.data(),    V_v.data(),    dW_v.data(),    dd,    learning_rate);
        sgd_update(W_o.data(),    V_o.data(),    dW_o.data(),    dd,    learning_rate);
        sgd_update(W_gate.data(), V_gate.data(), dW_gate.data(), dff_d, learning_rate);
        sgd_update(W_up.data(),   V_up.data(),   dW_up.data(),   dff_d, learning_rate);
        sgd_update(W_down.data(), V_down.data(), dW_down.data(), dff_d, learning_rate);

        return d_h_in;
    }

    // ── serialisation ──────────────────────────────────────────────────────────

    nlohmann::json TransformerBlock::save() const
    {
        using namespace json_helpers;
        return {
            {"W_q",    serialize_matrix(W_q)},
            {"W_k",    serialize_matrix(W_k)},
            {"W_v",    serialize_matrix(W_v)},
            {"W_o",    serialize_matrix(W_o)},
            {"W_gate", serialize_matrix(W_gate)},
            {"W_up",   serialize_matrix(W_up)},
            {"W_down", serialize_matrix(W_down)},
        };
    }

    void TransformerBlock::load(const nlohmann::json& j)
    {
        using namespace json_helpers;
        deserialize_matrix(j.at("W_q"),    W_q);
        deserialize_matrix(j.at("W_k"),    W_k);
        deserialize_matrix(j.at("W_v"),    W_v);
        deserialize_matrix(j.at("W_o"),    W_o);
        deserialize_matrix(j.at("W_gate"), W_gate);
        deserialize_matrix(j.at("W_up"),   W_up);
        deserialize_matrix(j.at("W_down"), W_down);

        // Reset momentum on load — do not persist transient training state
        V_q.fill(0.f);    V_k.fill(0.f);    V_v.fill(0.f);    V_o.fill(0.f);
        V_gate.fill(0.f); V_up.fill(0.f);   V_down.fill(0.f);
    }

} // namespace rllm
