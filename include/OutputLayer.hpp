#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <Corpus.hpp>

namespace rllm
{
    class OutputLayer
    {
      public:
        OutputLayer(const Corpus& corpus)
            : m_corpus(corpus)
        {}
        ~OutputLayer() = default;
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer& operator=(const OutputLayer&) = delete;

        void accumulate_input(TokenID target_idx, float value)
        {
            m_inputs.add_no_clamp(target_idx, value);
        }

        void compute_score(Score& score, const TokenID expected_output_token);

        void compute_deltas(const Score& score, template_token_vector<float, TokenID>& deltas) const;

        // RMSNorm the accumulated logits so the softmax sees reasonable magnitudes
        // regardless of the intermediate-layer fan-in.
        void rms_normalize_inputs();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

      public:
        template_token_vector<float, TokenID> m_inputs;

      private:
        const Corpus& m_corpus;
    };

} // namespace rllm
