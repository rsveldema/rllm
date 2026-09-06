# Multi-Token Prediction Targets

Training currently uses one `MultiTokenPredictionIndex` output head for each
example, predicting only the immediate next token. The `ONE` and `TWO` enum
names remain reserved for checkpoint and tooling compatibility but are outside
the allocated range.
For a token sequence `[t0, t1, ..., tN-1]`, the input context length is
`max(1, N - MultiTokenPredictionIndex::MAX)`.

Head `k` learns the token at `context_len + k` when that position exists.
When the position is past the end of the sequence, head `k` learns the reserved
`TokenID::INVALID` target.

The tokenizer generator always reserves the text token `INVALID` and emits the
stable enum alias `TokenID::INVALID`.
