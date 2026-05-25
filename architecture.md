# RLLM Architecture

RLLM is a small-scale experimental language model for next-token prediction. It implements a transformer decoder stack with multi-head causal self-attention, SwiGLU feed-forward blocks, and sinusoidal positional encodings, trained with SGD + momentum.

---

## System Overview

```mermaid
graph TD
    subgraph Entry["Entry Point (main.cc)"]
        CLI["CLI Args\n--train / --prompt\n--filter / --method / --epochs"]
    end

    subgraph RLLM_["RLLM (orchestrator)"]
        TM["train_mode()"]
        PM["prompt_mode()"]
    end

    subgraph Data["Data Layer"]
        Corpus["Corpus\nTokenizer + Vocabulary\nFile Loading"]
        TokenMap["tokenizer_map\n(generated C++)"]
    end

    subgraph NN["NeuralNetwork"]
        IL["InputLayer\nToken Embeddings\n+ Sinusoidal Pos. Enc."]
        TB0["TransformerBlock 0"]
        TBN["TransformerBlock 1..N"]
        OL["OutputLayer\nLM Head (linear)"]
    end

    subgraph IO["Persistence"]
        JSON["model.json\n(weights only)"]
    end

    CLI --> RLLM_
    RLLM_ --> Data
    RLLM_ --> NN
    Data --> TokenMap
    NN --> IL --> TB0 --> TBN --> OL
    NN <-->|load / save| JSON
```

---

## Forward Pass (Inference)

```mermaid
graph LR
    Tokens["Input Token IDs\n[t₀, t₁, ..., tₙ]"]
    Embed["Token Embedding Lookup\nE[T × 512]"]
    PosEnc["+ Sinusoidal\nPositional Encoding"]
    H["Hidden State h\n[T × 512]"]
    Blocks["Transformer Blocks × N\n(stacked)"]
    HLast["h_last[512]\n(final position)"]
    LMHead["LM Head\nW_lm_head @ h_last\nLogits[vocab]"]
    Softmax["Softmax\nProbabilities[vocab]"]
    TopK["Top-K OutputTokens\n{token_id, probability}"]

    Tokens --> Embed --> PosEnc --> H --> Blocks --> HLast --> LMHead --> Softmax --> TopK
```

---

## Transformer Block (single)

```mermaid
graph TD
    Input["h_in [T × 512]"]

    subgraph Attn["Multi-Head Causal Attention  (8 heads, head_dim = 64)"]
        RN1["RMSNorm"]
        Q["Q = h_norm @ W_q"]
        K["K = h_norm @ W_k"]
        V["V = h_norm @ W_v"]
        Scores["scores = softmax(Q·Kᵀ / √64)\n(causal mask)"]
        AttnOut["attn_out = scores · V"]
        Proj["out = attn_out @ W_o"]
    end

    Res1["+ Residual"]

    subgraph FFN["SwiGLU Feed-Forward  (dim = 2048)"]
        RN2["RMSNorm"]
        Gate["gate = h_mid @ W_gate  [2048]"]
        Up["up   = h_mid @ W_up    [2048]"]
        SiLU["silu(gate) ⊙ up"]
        Down["down = ffn_act @ W_down  [512]"]
    end

    Res2["+ Residual"]
    Output["h_out [T × 512]"]

    Input --> RN1 --> Q & K & V
    Q & K --> Scores
    Scores & V --> AttnOut --> Proj --> Res1
    Input --> Res1 --> RN2 --> Gate & Up
    Gate & Up --> SiLU --> Down --> Res2
    Res1 --> Res2 --> Output
```

---

## Training Flow

```mermaid
flowchart TD
    LoadCorpus["Load Corpus from Directory\n(filter by name)"]
    InitWeights["Initialize Random Weights\n(or load existing model.json)"]
    Epoch["For Each Epoch"]
    Method{Training Method}

    LineBased["Iterate Training Lines"]
    WindowBased["Sliding Window over Flat Tokens\nwindow size = N"]

    Pair["Extract (input sequence → target token)"]
    Fwd["Forward Pass\nPropagate through all layers"]
    Loss["Cross-Entropy Loss\nvs. expected token"]
    Check{Loss ≤ threshold?}
    Bwd["Backward Pass\nUpdate W_lm_head, Transformer weights,\nToken Embeddings via SGD+momentum"]
    MaxIter{Max iterations\nreached?}
    Fail["Record Failure"]
    Success["Record Success"]
    Save["Save model.json"]

    LoadCorpus --> InitWeights --> Epoch --> Method
    Method -->|TWO_TOK / THREE_TOK\nINCREASINGLY_LONGER| LineBased --> Pair
    Method -->|WINDOW| WindowBased --> Pair
    Pair --> Fwd --> Loss --> Check
    Check -->|No| Bwd --> MaxIter
    MaxIter -->|No| Fwd
    MaxIter -->|Yes| Fail --> Save
    Check -->|Yes| Success --> Save
```

**Training methods:**

| Method | Input | Target |
|--------|-------|--------|
| `TWO_TOK` | `[t₀]` | `t₁` |
| `THREE_TOK` | `[t₀, t₁]` | `t₂` |
| `INCREASINGLY_LONGER_SEQUENCES` | `[t₀ … tₖ]`, k = 1 → line_len−1 | `tₖ₊₁` |
| `WINDOW:N` | `[tᵢ … tᵢ₊ₙ₋₁]` (sliding) | `tᵢ₊ₙ` |

---

## Backward Pass

```mermaid
graph RL
    GT["Ground Truth Token"]
    Delta["Output Delta\ndL / d(logits)"]
    OLBwd["OutputLayer backward\nUpdate W_lm_head\nReturn dL/dh_last [512]"]
    FullGrad["Full Gradient dL/dh\n[T × 512]\n(non-zero at last position)"]
    TBBwd["TransformerBlocks backward\n(reverse order)\nUpdate W_q, W_k, W_v, W_o\nW_gate, W_up, W_down"]
    ILBwd["InputLayer backward\nUpdate token embeddings\n(pos. enc. frozen)"]

    GT --> Delta --> OLBwd --> FullGrad --> TBBwd --> ILBwd
```

**Optimizer:** SGD + momentum (β = 0.9), lr = 0.01, gradient clip ±1.0, weight clamp ±2.0.

---

## Prompt Mode (Inference Loop)

```mermaid
flowchart TD
    Load["Load model.json"]
    Input["User types prompt"]
    Tokenize["Tokenize → Token ID sequence"]
    Fwd["Forward pass"]
    TopK["Get Top-5 predictions\n(softmax probabilities)"]
    Valid{Any prediction\n≥ 10% confidence?}
    Stop["Stop generation"]
    Pick["Pick next token\n(highest prio or random)"]
    Append["Append to sequence"]
    MaxTok{10 tokens\ngenerated?}
    Show["Display answer"]

    Load --> Input --> Tokenize --> Fwd --> TopK --> Valid
    Valid -->|No| Stop --> Show
    Valid -->|Yes| Pick --> Append --> MaxTok
    MaxTok -->|No| Fwd
    MaxTok -->|Yes| Show --> Input
```

---

## Key Compile-Time Dimensions

| Enum | Value | Meaning |
|------|-------|---------|
| `EmbeddingDimension::MAX` | 512 | Hidden state / model dimension |
| `PositionIndex::MAX` | 128 | Max sequence length |
| `HeadsIndex::MAX` | 8 | Number of attention heads |
| `HeadDimension::MAX` | 64 | Per-head dim (512 / 8) |
| `FFDimension::MAX` | 2048 | FFN intermediate dim (512 × 4) |

---

## File Map

```
include/
  RLLM.hpp              – Top-level orchestrator interface
  NeuralNetwork.hpp     – Core model: forward / backward / serialization
  InputLayer.hpp        – Token embeddings + positional encoding
  TransformerBlock.hpp  – Single decoder block (attention + FFN)
  OutputLayer.hpp       – LM head (linear projection to vocab logits)
  Corpus.hpp            – Tokenizer, vocabulary, training data iteration
  LayerPrimitives.hpp   – Shared enums, matrix types, Score, OutputToken

src/
  main.cc               – CLI argument parsing, entry point
  RLLM.cc               – train_mode() and prompt_mode() implementations
  NeuralNetwork.cc      – Forward pass, backward pass, serialization
  TransformerBlock.cc   – Attention + SwiGLU forward & backward
  InputLayer.cc         – Embedding lookup + sinusoidal enc.
  OutputLayer.cc        – LM head forward, loss, backward
  Corpus.cpp            – File loading, tokenization, line iteration
  serialization.cc      – JSON load/save helpers

build/generated/
  tokenizer_map.cc/.hpp – Auto-generated token↔ID mapping (from create_tokenizer_map.py)
```
