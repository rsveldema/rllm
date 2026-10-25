#pragma once

#include <LayerPrimitives.hpp>

namespace rllm::backprop
{
    struct MuonWorkspace
    {
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> gram;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> gram_update;
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension> next_square;
        fixed_size_matrix<float, FFDimension, EmbeddingDimension> next_tall;
        fixed_size_matrix<float, EmbeddingDimension, FFDimension> next_wide;
        fixed_size_vector<float, TempStorage> norm;

        MuonWorkspace() { norm.set_size(TempStorage::MAX); }
    };

    void muon_update(
        fixed_size_matrix<float16, EmbeddingDimension, EmbeddingDimension>& weight,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& momentum,
        fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& update,
        const fixed_size_matrix<float, EmbeddingDimension, EmbeddingDimension>& gradient,
        MuonWorkspace& workspace, float learning_rate,
        fixed_size_vector<float, TempStorage>& diagnostics);
    void muon_update(
        fixed_size_matrix<float16, FFDimension, EmbeddingDimension>& weight,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& momentum,
        fixed_size_matrix<float, FFDimension, EmbeddingDimension>& update,
        const fixed_size_matrix<float, FFDimension, EmbeddingDimension>& gradient,
        MuonWorkspace& workspace, float learning_rate,
        fixed_size_vector<float, TempStorage>& diagnostics);
    void muon_update(
        fixed_size_matrix<float16, EmbeddingDimension, FFDimension>& weight,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& momentum,
        fixed_size_matrix<float, EmbeddingDimension, FFDimension>& update,
        const fixed_size_matrix<float, EmbeddingDimension, FFDimension>& gradient,
        MuonWorkspace& workspace, float learning_rate,
        fixed_size_vector<float, TempStorage>& diagnostics);
}
