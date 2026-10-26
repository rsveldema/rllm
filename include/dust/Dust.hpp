#pragma once

#include <LayerPrimitives.hpp>
#include <cstddef>
#include <cstdint>

namespace rllm::dust
{
    enum class ProjectionSite : uint64_t
    {
        Query = 1,
        Key,
        Value,
        AttentionOutput,
        Gate,
        Up,
        Down,
    };

    struct Settings
    {
        size_t population = 256;
        size_t draws_per_forward = 2;
        float activation_noise = 0.2f;
        float attention_noise = 0.05f;
        float credit_decay = 0.98f;
    };

    struct Backend final
    {
        static constexpr const char* name = "dust";
    };

    /** Reusable noise buffers for one perturbed transformer forward pass. */
    struct ForwardPerturbation
    {
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension> hidden_noise;
        flexible_rows_matrix<float, PositionIndex, FFDimension> ff_noise;
        PositionIndex rows;
        uint64_t seed;
        float activation_sigma;
        float attention_sigma;

        ForwardPerturbation(PositionIndex rows_, uint64_t seed_,
            float activation_sigma_, float attention_sigma_);
        void reset(PositionIndex rows_, uint64_t seed_);
        void hidden(
            flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& activations,
            ProjectionSite site, size_t execution);
        void ff(
            flexible_rows_matrix<float, PositionIndex, FFDimension>& activations,
            ProjectionSite site, size_t execution);
        uint64_t site_seed(ProjectionSite site, size_t execution) const;
    };

    /** Add sigma-scaled activation noise in-place for one DUST draw. */
    void perturb_hidden(
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& activations,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& noise,
        PositionIndex rows,
        float sigma);

    /** Accumulate (reward / sigma) * noise as an additive optimizer descent direction. */
    void accumulate_hidden_error(
        const fixed_size_vector<float, PositionIndex>& rewards,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& noise,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& error,
        PositionIndex rows,
        float inverse_population_sigma);

    /** Form sum_t error[t]^T input[t], the same local linear gradient as DUST. */
    void accumulate_hidden_outer_product(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& error,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& layer_input,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient,
        PositionIndex rows);

    /** Deterministic host-side Gaussian draw, uploaded for the Vulkan estimator. */
    void gaussian_hidden_noise(
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& noise,
        PositionIndex rows,
        uint seed);

    /** Per-example loss improvement, clean_loss - perturbed_loss. */
    void compute_rewards(
        const fixed_size_vector<float, BatchIndex>& clean_losses,
        const fixed_size_vector<float, BatchIndex>& perturbed_losses,
        fixed_size_vector<float, BatchIndex>& rewards,
        BatchIndex count);
    void scatter_rewards(
        const fixed_size_vector<float, BatchIndex>& rewards,
        const fixed_size_vector<int, BatchIndex>& row_indices,
        fixed_size_vector<float, PositionIndex>& row_rewards,
        BatchIndex count);
    void causal_credit_rewards(
        const fixed_size_vector<float, PositionIndex>& row_rewards,
        fixed_size_vector<float, PositionIndex>& credited_rewards,
        const fixed_size_vector<int, PositionIndex>& row_batch,
        const fixed_size_vector<int, BatchIndex>& row_end,
        PositionIndex rows, float decay);

    void perturb_logits(
        fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        const fixed_size_matrix<float, BatchIndex, TokenID>& noise,
        BatchIndex count, float sigma);

    void accumulate_logit_error(
        const fixed_size_vector<float, BatchIndex>& rewards,
        const fixed_size_matrix<float, BatchIndex, TokenID>& noise,
        fixed_size_matrix<float, BatchIndex, TokenID>& error,
        BatchIndex count, float inverse_population_sigma);

    void accumulate_output_outer_product(
        const fixed_size_matrix<float, BatchIndex, TokenID>& error,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& hidden,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& gradient,
        BatchIndex count);

    void gaussian_logit_noise(
        fixed_size_matrix<float, BatchIndex, TokenID>& noise,
        BatchIndex count, uint seed);

    void perturb_batched_hidden(
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& hidden,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& noise,
        BatchIndex count, float sigma);
    void accumulate_batched_hidden_error(
        const fixed_size_vector<float, BatchIndex>& rewards,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& noise,
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& error,
        BatchIndex count, float inverse_population_sigma);
    void gaussian_batched_hidden_noise(
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& noise,
        BatchIndex count, uint64_t seed);

    void perturb_ff(
        flexible_rows_matrix<float, PositionIndex, FFDimension>& activations,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& noise,
        PositionIndex rows, float sigma);
    void gaussian_ff_noise(
        flexible_rows_matrix<float, PositionIndex, FFDimension>& noise,
        PositionIndex rows, uint seed);

    void accumulate_hidden_ff_outer_product(
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& error,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& layer_input,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient,
        PositionIndex rows);
    void accumulate_ff_error(
        const fixed_size_vector<float, PositionIndex>& rewards,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& noise,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& error,
        PositionIndex rows, float inverse_population_sigma);
    void accumulate_ff_hidden_outer_product(
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& error,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& layer_input,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient,
        PositionIndex rows);
}
