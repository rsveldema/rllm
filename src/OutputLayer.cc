#include <OutputLayer.hpp>
#include <RandomHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace rllm
{
    //root-mean-square normalize the accumulated logits so the softmax sees reasonable magnitudes
    //regardless of the intermediate-layer fan-in.
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

    // Compute numerically-stable softmax of m_inputs and store the backprop delta
    // in score.values using the convention delta = one_hot - softmax (positive means
    // "this token should fire more").  The backprop weight update is weight += lr*delta*act,
    // so the delta must have the sign of (target - actual), NOT (actual - target).
    void OutputLayer::compute_score(Score& score, const TokenID expected_output_token)
    {
        // Numerically stable softmax: subtract max before exponentiation.
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
