#include <dust/Dust.hpp>

#include <cpu/cpu_flex_rows_matrix.hpp>
#include <enum_iterator1D.hpp>
#include <enum_iterator2D.hpp>
#include <random>
#include <rllm_vulkan_runtime.hpp>

namespace rllm::dust
{
    static_assert(Backend::name[0] == 'd');

    ForwardPerturbation::ForwardPerturbation(
        PositionIndex rows_, uint64_t seed_, float activation_sigma_,
        float attention_sigma_)
        : hidden_noise(rows_), ff_noise(rows_), rows(rows_), seed(seed_),
          activation_sigma(activation_sigma_), attention_sigma(attention_sigma_)
    {}

    void ForwardPerturbation::reset(PositionIndex rows_, uint64_t seed_)
    {
        rows = rows_;
        seed = seed_;
        hidden_noise.set_rows(rows);
        ff_noise.set_rows(rows);
    }

    uint64_t ForwardPerturbation::site_seed(
        ProjectionSite site, size_t execution) const
    {
        // SplitMix64 finalizer: deterministic and well separated by site/layer.
        uint64_t value = seed ^
            (static_cast<uint64_t>(site) * 0x9e3779b97f4a7c15ULL) ^
            (static_cast<uint64_t>(execution + 1) * 0xbf58476d1ce4e5b9ULL);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    void ForwardPerturbation::hidden(
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& activations,
        ProjectionSite site, size_t execution)
    {
        gaussian_hidden_noise(hidden_noise, rows, site_seed(site, execution));
        const bool attention = site == ProjectionSite::Query ||
            site == ProjectionSite::Key || site == ProjectionSite::Value;
        perturb_hidden(activations, hidden_noise, rows,
            attention ? attention_sigma : activation_sigma);
    }

    void ForwardPerturbation::ff(
        flexible_rows_matrix<float, PositionIndex, FFDimension>& activations,
        ProjectionSite site, size_t execution)
    {
        gaussian_ff_noise(ff_noise, rows, site_seed(site, execution));
        perturb_ff(activations, ff_noise, rows, activation_sigma);
    }

    void perturb_hidden(
        // OFFLOAD_PARAMETERS(activations, noise, rows, sigma)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& activations,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& noise,
        PositionIndex rows,
        float sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, d, grid, (activations, noise, rows, sigma))
        activations[row, d] += (sigma * noise[row, d]);
        ENDFOR
    }

    void accumulate_hidden_error(
        // OFFLOAD_PARAMETERS(rewards, noise, error, rows, inverse_population_sigma)
        const fixed_size_vector<float, PositionIndex>& rewards,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& noise,
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& error,
        PositionIndex rows,
        float inverse_population_sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, d, grid, (rewards, noise, error, rows, inverse_population_sigma))
        error[row, d] -= ((rewards[row] * noise[row, d]) * inverse_population_sigma);
        ENDFOR
    }

    void accumulate_hidden_outer_product(
        // OFFLOAD_PARAMETERS(error, layer_input, gradient, rows)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& error,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& layer_input,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient,
        PositionIndex rows
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, output_dimension, input_dimension, grid, (error, layer_input, gradient, rows))
        float value = gradient[output_dimension, input_dimension];
        for (const auto row : enum_iterator1D<PositionIndex>(rows))
            value += (error[row, output_dimension] * layer_input[row, input_dimension]);
        gradient[output_dimension, input_dimension] = value;
        ENDFOR
    }

    void gaussian_hidden_noise(
        // OFFLOAD_PARAMETERS(noise, rows, seed)
        flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& noise,
        PositionIndex rows,
        uint seed
        // END_OFFLOAD_PARAMETERS
    )
    {
        noise.set_rows(rows);
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, EmbeddingDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, d, grid, (noise, rows, seed))
        const float row_value = float(row);
        const float dimension_value = float(d);
        const float seed_value = float(seed);
        const float state = (((row_value * 12.9898f) +
            (dimension_value * 78.233f)) + seed_value);
        const float shifted_state = (state + 19.19f);
        const float sine1 = sin(state);
        const float sine2 = sin(shifted_state);
        const float raw1 = (sine1 * 43758.5453f);
        const float raw2 = (sine2 * 24634.6345f);
        const int whole1 = int(raw1);
        const int whole2 = int(raw2);
        const float whole1_value = float(whole1);
        const float whole2_value = float(whole2);
        const float difference1 = (raw1 - whole1_value);
        const float difference2 = (raw2 - whole2_value);
        const float fraction1 = (difference1 < 0.0f) ? -difference1 : difference1;
        const float fraction2 = (difference2 < 0.0f) ? -difference2 : difference2;
        const float scaled_fraction1 = (fraction1 * 0.999999f);
        const float u1 = (scaled_fraction1 + 0.000001f);
        const float u2 = fraction2;
        const float logarithm = log(u1);
        const float radius_squared = (-2.0f * logarithm);
        const float radius = sqrt(radius_squared);
        const float angle = (6.283185307179586f * u2);
        const float cosine = cos(angle);
        noise[row, d] = (radius * cosine);
        ENDFOR
    }

    void compute_rewards(
        // OFFLOAD_PARAMETERS(clean_losses, perturbed_losses, rewards, count)
        const fixed_size_vector<float, BatchIndex>& clean_losses,
        const fixed_size_vector<float, BatchIndex>& perturbed_losses,
        fixed_size_vector<float, BatchIndex>& rewards,
        BatchIndex count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, item, enum_iterator1D<BatchIndex>(count), (clean_losses, perturbed_losses, rewards, count))
        rewards[item] = (clean_losses[item] - perturbed_losses[item]);
        ENDFOR
    }

    void scatter_rewards(
        // OFFLOAD_PARAMETERS(rewards, row_indices, row_rewards, count)
        const fixed_size_vector<float, BatchIndex>& rewards,
        const fixed_size_vector<int, BatchIndex>& row_indices,
        fixed_size_vector<float, PositionIndex>& row_rewards,
        BatchIndex count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, item, enum_iterator1D<BatchIndex>(count), (rewards, row_indices, row_rewards, count))
        row_rewards[row_indices[item]] += rewards[item];
        ENDFOR
    }

    void causal_credit_rewards(
        // OFFLOAD_PARAMETERS(row_rewards, credited_rewards, row_batch, row_end, rows, decay)
        const fixed_size_vector<float, PositionIndex>& row_rewards,
        fixed_size_vector<float, PositionIndex>& credited_rewards,
        const fixed_size_vector<int, PositionIndex>& row_batch,
        const fixed_size_vector<int, BatchIndex>& row_end,
        PositionIndex rows, float decay
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        OFFLOAD_PARFOR_1D_PARAM(queue, row, enum_iterator1D<PositionIndex>(rows), (row_rewards, credited_rewards, row_batch, row_end, rows, decay))
        const int end = row_end[row_batch[row]];
        float credit = 0.0f;
        float factor = 1.0f;
        for (const auto future : enum_iterator1D<PositionIndex>(row, static_cast<PositionIndex>(end)))
        {
            credit += (factor * row_rewards[future]);
            factor *= decay;
        }
        credited_rewards[row] = credit;
        ENDFOR
    }

    void perturb_logits(
        // OFFLOAD_PARAMETERS(logits, noise, count, sigma)
        fixed_size_matrix<float, BatchIndex, TokenID>& logits,
        const fixed_size_matrix<float, BatchIndex, TokenID>& noise,
        BatchIndex count,
        float sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(count);
        OFFLOAD_PARFOR_2D_PARAM(queue, item, token, grid, (logits, noise, count, sigma))
        logits[item, token] += (sigma * noise[item, token]);
        ENDFOR
    }

    void accumulate_logit_error(
        // OFFLOAD_PARAMETERS(rewards, noise, error, count, inverse_population_sigma)
        const fixed_size_vector<float, BatchIndex>& rewards,
        const fixed_size_matrix<float, BatchIndex, TokenID>& noise,
        fixed_size_matrix<float, BatchIndex, TokenID>& error,
        BatchIndex count,
        float inverse_population_sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(count);
        OFFLOAD_PARFOR_2D_PARAM(queue, item, token, grid, (rewards, noise, error, count, inverse_population_sigma))
        error[item, token] -= ((rewards[item] * noise[item, token]) * inverse_population_sigma);
        ENDFOR
    }

    void accumulate_output_outer_product(
        // OFFLOAD_PARAMETERS(error, hidden, gradient, count)
        const fixed_size_matrix<float, BatchIndex, TokenID>& error,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& hidden,
        fixed_size_matrix<float, TokenID, EmbeddingDimension>& gradient,
        BatchIndex count
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<TokenID, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, token, dimension, grid, (error, hidden, gradient, count))
        float value = gradient[token, dimension];
        for (const auto item : enum_iterator1D<BatchIndex>(count))
            value += (error[item, token] * hidden[item, dimension]);
        gradient[token, dimension] = value;
        ENDFOR
    }

    void gaussian_logit_noise(
        // OFFLOAD_PARAMETERS(noise, count, seed)
        fixed_size_matrix<float, BatchIndex, TokenID>& noise,
        BatchIndex count, uint seed
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, TokenID>(count);
        OFFLOAD_PARFOR_2D_PARAM(queue, item, token, grid, (noise, count, seed))
        const float item_value = float(item);
        const float token_value = float(token);
        const float seed_value = float(seed);
        const float state = (((item_value * 12.9898f) +
            (token_value * 78.233f)) + seed_value);
        const float shifted_state = (state + 19.19f);
        const float sine1 = sin(state);
        const float sine2 = sin(shifted_state);
        const float raw1 = (sine1 * 43758.5453f);
        const float raw2 = (sine2 * 24634.6345f);
        const int whole1 = int(raw1);
        const int whole2 = int(raw2);
        const float whole1_value = float(whole1);
        const float whole2_value = float(whole2);
        const float difference1 = (raw1 - whole1_value);
        const float difference2 = (raw2 - whole2_value);
        const float fraction1 = (difference1 < 0.0f) ? -difference1 : difference1;
        const float fraction2 = (difference2 < 0.0f) ? -difference2 : difference2;
        const float scaled_fraction1 = (fraction1 * 0.999999f);
        const float u1 = (scaled_fraction1 + 0.000001f);
        const float u2 = fraction2;
        const float logarithm = log(u1);
        const float radius_squared = (-2.0f * logarithm);
        const float radius = sqrt(radius_squared);
        const float angle = (6.283185307179586f * u2);
        const float cosine = cos(angle);
        noise[item, token] = (radius * cosine);
        ENDFOR
    }

    void perturb_batched_hidden(
        // OFFLOAD_PARAMETERS(hidden, noise, count, sigma)
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& hidden,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& noise,
        BatchIndex count, float sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(count);
        OFFLOAD_PARFOR_2D_PARAM(queue, item, dimension, grid, (hidden, noise, count, sigma))
        hidden[item, dimension] += (sigma * noise[item, dimension]);
        ENDFOR
    }

    void accumulate_batched_hidden_error(
        // OFFLOAD_PARAMETERS(rewards, noise, error, count, inverse_population_sigma)
        const fixed_size_vector<float, BatchIndex>& rewards,
        const fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& noise,
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& error,
        BatchIndex count, float inverse_population_sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<BatchIndex, EmbeddingDimension>(count);
        OFFLOAD_PARFOR_2D_PARAM(queue, item, dimension, grid, (rewards, noise, error, count, inverse_population_sigma))
        error[item, dimension] -= ((rewards[item] * noise[item, dimension]) * inverse_population_sigma);
        ENDFOR
    }

    void gaussian_batched_hidden_noise(
        fixed_size_matrix<float, BatchIndex, EmbeddingDimension>& noise,
        BatchIndex count, uint64_t seed)
    {
        cpu_fixed_matrix<float, BatchIndex, EmbeddingDimension> host;
        host.zero();
        std::mt19937_64 generator(seed);
        std::normal_distribution<float> normal;
        for (const auto item : enum_iterator1D<BatchIndex>(count))
            for (const auto dimension : enum_iterator1D<EmbeddingDimension>())
                host[item, dimension] = normal(generator);
        noise.copy_from_cpu(vulkan_runtime::get_queue(0), host);
    }

    void perturb_ff(
        // OFFLOAD_PARAMETERS(activations, noise, rows, sigma)
        flexible_rows_matrix<float, PositionIndex, FFDimension>& activations,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& noise,
        PositionIndex rows, float sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, dimension, grid, (activations, noise, rows, sigma))
        activations[row, dimension] += (sigma * noise[row, dimension]);
        ENDFOR
    }

    void gaussian_ff_noise(
        // OFFLOAD_PARAMETERS(noise, rows, seed)
        flexible_rows_matrix<float, PositionIndex, FFDimension>& noise,
        PositionIndex rows, uint seed
        // END_OFFLOAD_PARAMETERS
    )
    {
        noise.set_rows(rows);
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, dimension, grid, (noise, rows, seed))
        const float row_value = float(row);
        const float dimension_value = float(dimension);
        const float seed_value = float(seed);
        const float state = (((row_value * 12.9898f) +
            (dimension_value * 78.233f)) + seed_value);
        const float shifted_state = (state + 19.19f);
        const float sine1 = sin(state);
        const float sine2 = sin(shifted_state);
        const float raw1 = (sine1 * 43758.5453f);
        const float raw2 = (sine2 * 24634.6345f);
        const int whole1 = int(raw1);
        const int whole2 = int(raw2);
        const float whole1_value = float(whole1);
        const float whole2_value = float(whole2);
        const float difference1 = (raw1 - whole1_value);
        const float difference2 = (raw2 - whole2_value);
        const float fraction1 = (difference1 < 0.0f) ? -difference1 : difference1;
        const float fraction2 = (difference2 < 0.0f) ? -difference2 : difference2;
        const float scaled_fraction1 = (fraction1 * 0.999999f);
        const float u1 = (scaled_fraction1 + 0.000001f);
        const float u2 = fraction2;
        const float logarithm = log(u1);
        const float radius_squared = (-2.0f * logarithm);
        const float radius = sqrt(radius_squared);
        const float angle = (6.283185307179586f * u2);
        const float cosine = cos(angle);
        noise[row, dimension] = (radius * cosine);
        ENDFOR
    }

    void accumulate_hidden_ff_outer_product(
        // OFFLOAD_PARAMETERS(error, layer_input, gradient, rows)
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& error,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& layer_input,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient,
        PositionIndex rows
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<EmbeddingDimension, FFDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, output_dimension, input_dimension, grid, (error, layer_input, gradient, rows))
        float value = gradient[output_dimension, input_dimension];
        for (const auto row : enum_iterator1D<PositionIndex>(rows))
            value += (error[row, output_dimension] * layer_input[row, input_dimension]);
        gradient[output_dimension, input_dimension] = value;
        ENDFOR
    }

    void accumulate_ff_error(
        // OFFLOAD_PARAMETERS(rewards, noise, error, rows, inverse_population_sigma)
        const fixed_size_vector<float, PositionIndex>& rewards,
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& noise,
        flexible_rows_matrix<float, PositionIndex, FFDimension>& error,
        PositionIndex rows, float inverse_population_sigma
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<PositionIndex, FFDimension>(rows);
        OFFLOAD_PARFOR_2D_PARAM(queue, row, dimension, grid, (rewards, noise, error, rows, inverse_population_sigma))
        error[row, dimension] -= ((rewards[row] * noise[row, dimension]) * inverse_population_sigma);
        ENDFOR
    }

    void accumulate_ff_hidden_outer_product(
        // OFFLOAD_PARAMETERS(error, layer_input, gradient, rows)
        const flexible_rows_matrix<float, PositionIndex, FFDimension>& error,
        const flexible_rows_matrix<float, PositionIndex, EmbeddingDimension>& layer_input,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient,
        PositionIndex rows
        // END_OFFLOAD_PARAMETERS
    )
    {
        auto& queue = vulkan_runtime::get_queue(0);
        const auto grid = enum_iterator2D<FFDimension, EmbeddingDimension>();
        OFFLOAD_PARFOR_2D_PARAM(queue, output_dimension, input_dimension, grid, (error, layer_input, gradient, rows))
        float value = gradient[output_dimension, input_dimension];
        for (const auto row : enum_iterator1D<PositionIndex>(rows))
            value += (error[row, output_dimension] * layer_input[row, input_dimension]);
        gradient[output_dimension, input_dimension] = value;
        ENDFOR
    }
}
