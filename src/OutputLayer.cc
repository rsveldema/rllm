#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

namespace rllm
{
    static constexpr float MOMENTUM_BETA = 0.9f;
    static constexpr float GRAD_CLIP     = 1.0f;
    static constexpr float VEL_CLIP      = 0.1f;
    static constexpr float WEIGHT_CLAMP  = 2.0f;

    OutputLayer::OutputLayer(const Corpus& corpus)
        : m_corpus(corpus)
    {
    }

    void OutputLayer::set_random_weights()
    {
        const int   D     = static_cast<int>(EmbeddingDimension::MAX);
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        const int   n     = static_cast<int>(W_lm_head.ROWS * W_lm_head.COLS);
        for (int i = 0; i < n; ++i) W_lm_head.data()[i] = get_random_value(-scale, scale);
        V_lm_head.fill(0.f);
    }

    // logits[v] = sum_d  h_last[d] * W_lm_head[v, d]
    void OutputLayer::forward_from_hidden(const template_token_vector<float, EmbeddingDimension>& h_last)
    {
        const int V = static_cast<int>(TokenID::MAX);
        const int D = static_cast<int>(EmbeddingDimension::MAX);

        m_inputs.fill(0.f);
#pragma omp parallel for schedule(static)
        for (int v = 0; v < V; ++v)
        {
            const float* w_row = W_lm_head.data() + v * D;
            float sum = 0.f;
#pragma omp simd reduction(+:sum)
            for (int d = 0; d < D; ++d) sum += h_last[static_cast<EmbeddingDimension>(d)] * w_row[d];
            m_inputs[static_cast<TokenID>(v)] = sum;
        }
    }

    // Returns dL/dh_last[D] and updates W_lm_head.
    template_token_vector<float, EmbeddingDimension> OutputLayer::backward_and_update(
        const template_token_vector<float, TokenID>& delta,
        const template_token_vector<float, EmbeddingDimension>& h_last,
        float                                        learning_rate
    )
    {
        const int V = static_cast<int>(TokenID::MAX);
        const int D = static_cast<int>(EmbeddingDimension::MAX);

        template_token_vector<float, EmbeddingDimension> dh;
        dh.fill(0.f);

        for (int v = 0; v < V; ++v)
        {
            const float dv    = delta[static_cast<TokenID>(v)];
            float*       w_row = W_lm_head.data() + v * D;
            float*       vel   = V_lm_head.data() + v * D;

#pragma omp simd
            for (int d = 0; d < D; ++d)
            {
                dh[static_cast<EmbeddingDimension>(d)] += dv * w_row[d];
                const float g = std::clamp(dv * h_last[static_cast<EmbeddingDimension>(d)], -GRAD_CLIP, GRAD_CLIP);
                vel[d]   = std::clamp(MOMENTUM_BETA * vel[d] + learning_rate * g, -VEL_CLIP, VEL_CLIP);
                w_row[d] = std::clamp(w_row[d] + vel[d], -WEIGHT_CLAMP, WEIGHT_CLAMP);
            }
        }
        return dh;
    }

    // ── scoring (unchanged from previous architecture) ─────────────────────────

    void OutputLayer::rms_normalize_inputs()
    {
        constexpr float eps = 1e-6f;
        const float n = static_cast<float>(TokenID::MAX);
        const int n_int = static_cast<int>(TokenID::MAX);
        float sum_sq = 0.0f;
#pragma omp simd reduction(+:sum_sq)
        for (int i = 0; i < n_int; ++i)
        {
            const auto k = m_inputs[static_cast<TokenID>(i)];
            sum_sq += k * k;
        }
        const float rms = std::sqrt(sum_sq / n + eps);
#pragma omp simd
        for (int i = 0; i < n_int; ++i)
            m_inputs[static_cast<TokenID>(i)] /= rms;
    }

    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        const int n_tok = static_cast<int>(TokenID::MAX);
        float max_val = m_inputs[TokenID::START];
#pragma omp simd reduction(max:max_val)
        for (int i = 0; i < n_tok; ++i)
            max_val = std::max(max_val, m_inputs[static_cast<TokenID>(i)]);

        float sum_exp = 0.0f;
#pragma omp simd reduction(+:sum_exp)
        for (int i = 0; i < n_tok; ++i)
        {
            score.values[static_cast<TokenID>(i)] = std::exp(m_inputs[static_cast<TokenID>(i)] - max_val);
            sum_exp += score.values[static_cast<TokenID>(i)];
        }

#pragma omp simd
        for (int i = 0; i < n_tok; ++i)
            score.values[static_cast<TokenID>(i)] = -score.values[static_cast<TokenID>(i)] / sum_exp;

        score.values[expected_output_token] += 1.0f;
    }

    void OutputLayer::compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const
    {
        deltas = score.values;
    }

} // namespace rllm
