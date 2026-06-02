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
        OutputLayer() = default;
        OutputLayer(const Corpus& corpus);
        ~OutputLayer() = default;
        OutputLayer(const OutputLayer&) = delete;
        OutputLayer& operator=(const OutputLayer&) = delete;

        // Initialise W_lm_head with small random values.
        void set_random_weights();

        // Project h_last[D_MODEL] to vocabulary logits, storing them in m_inputs.
        void forward_from_hidden(const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last);

        // Backpropagate the output delta through W_lm_head.
        // Accumulates ∂L/∂h_last into dh_last and updates W_lm_head via SGD+momentum.
        void backward_and_update(
            const fixed_size_vector<rlmm_float, TokenID>& delta,
            const fixed_size_vector<rlmm_float, EmbeddingDimension>& h_last,
          fixed_size_vector<rlmm_float, EmbeddingDimension>& dh_last,
            float learning_rate
        );

        // Computes softmax deltas (with label smoothing) into score for backprop,
        // and returns the cross-entropy loss -log(softmax[target]).
        float compute_score(Score& score, const TokenID expected_output_token);
        void rms_normalize_inputs();

        void load(const nlohmann::json& j);
        nlohmann::json save() const;

        // Returns the top-k tokens by raw logit value.
        std::vector<OutputToken> get_top_k_by_logit(size_t k) const;

      private:
        // Vocabulary logits computed by forward_from_hidden().
        fixed_size_vector<rlmm_float, TokenID> m_inputs;

        // LM head weight matrix [vocab × D_MODEL] (out × in), row-major.
        fixed_size_matrix<rlmm_float_small, TokenID, EmbeddingDimension> W_lm_head;
        fixed_size_matrix<rlmm_float, TokenID, EmbeddingDimension> V_lm_head; // SGD momentum velocities
    };

} // namespace rllm
