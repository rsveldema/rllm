// === PARFOR block from src/vecmath.cc:296 ===

OFFLOAD_PARFOR_2D_PARAM(i, j, grid, (A, B, C))

// === OFFLOAD_PARAMETERS ===
const rlmm::fixed_size_matrix<rlmm_float, EmbeddingDimension, Position>& A = A;
const rlmm::fixed_size_matrix<rlmm_float, TokenID, Position>& B = B;
rlmm::fixed_size_matrix<rlmm_float, PositionID, Token>& C = C;

// === PARFOR BODY ===
float sum = 0.f;
RLLM_OMP_SIMD_REDUCTION_PLUS(sum)
for (size_t l_idx = 0; l_idx < static_cast<size_t>(EmbeddingDimension::MAX); ++l_idx)
{
    const int k = int(l_idx);
    const float term = A[i, k] * B[j, k];
    OVERFLOW_CHECK_ADD(sum, term);
    sum += term;
}
C[i, j] = static_cast<rlmm_float>(sum);
