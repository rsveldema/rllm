#pragma once

#include <LayerPrimitives.hpp>

#include <nlohmann/json_fwd.hpp>
#include <Corpus.hpp>

namespace rllm
{
    // OutputLayer holds the learned linear projection ("LM head") from the last
    // transformer block's hidden state at the final sequence position to
    // vocabulary logits, plus the scoring and serialisation logic.
    //
    // W_lm_head is [TokenID::MAX × EmbeddingDimension::MAX] (out × in).
    class OutputLayer
    {
      public:
        OutputLayer(const Corpus& corpus);
        ~OutputLayer() = default;
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer& operator=(const OutputLayer&) = delete;

        // Initialise W_lm_head with small random values.
        void set_random_weights();

        // Project h_last[D_MODEL] to vocabulary logits, storing them in m_inputs.
        void forward_from_hidden(const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last);

        // Backpropagate the output delta through W_lm_head.
        // Returns d_h_last[D_MODEL] = ∂L/∂h_last and updates W_lm_head via SGD+momentum.
        fixed_size_vector<rlmm_float, EmbeddingDimension> backward_and_update(
            const fixed_size_vector<rlmm_float, TokenID>& delta,
            const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
            float learning_rate
        );

        void compute_score(Score& score, const TokenID expected_output_token);
        void compute_deltas(const Score& score, fixed_size_vector<rlmm_float, TokenID>& deltas) const;
        void rms_normalize_inputs();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

        // Vocabulary logits computed by forward_from_hidden().
        fixed_size_vector<rlmm_float, TokenID> m_inputs;

      private:
        const Corpus& m_corpus;

        // LM head weight matrix [vocab × D_MODEL] (out × in), row-major.
        fixed_size_matrix<rlmm_float, TokenID, EmbeddingDimension> W_lm_head;
        fixed_size_matrix<rlmm_float, TokenID, EmbeddingDimension> V_lm_head;  // SGD momentum velocities
    };

} // namespace rllm
