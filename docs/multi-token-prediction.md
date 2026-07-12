# Multi-Token Prediction Targets

Training uses all `MultiTokenPredictionIndex` output heads for each example.
For a token sequence `[t0, t1, ..., tN-1]`, the input context length is
`max(1, N - MultiTokenPredictionIndex::MAX)`.

Head `k` learns the token at `context_len + k` when that position exists.
When the position is past the end of the sequence, head `k` learns the reserved
`TokenID::INVALID` target.

The tokenizer generator always reserves the text token `INVALID` and emits the
stable enum alias `TokenID::INVALID`.
