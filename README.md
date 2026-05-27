This is an experimental LLM.
Just a vehicle to experiment with.

How to Build
===============

```bash
mkdir build
cd build
cmake ..
cmake --build .
```


How to use
=============

To train:

 ./build/rllm --train --filter simple --method window3 --epochs 10

Prompt mode:

 ./build/rllm


TODOs:
==============

more accuracy:
- multi-stage
- transformer arch
- check impact of more layers and/or wider layers

more features:
- check prompt mode

be faster:
- optimistic concurrency
- CI for testing various options overnight
- OpenCL instead of OpenMP


mob-of-expert support:
    FFN networks are specialized:

        struct Expert {
            // SwiGLU FFN:
            //   gate, up:  [FFDimension::MAX    × D_MODEL]  (out × in)
            //   down:      [D_MODEL × FFDimension::MAX   ]  (out × in)
            fixed_size_matrix<rlmm_float16, FFDimension, EmbeddingDimension> W_gate, W_up;
            fixed_size_matrix<rlmm_float16, EmbeddingDimension, FFDimension> W_down;
            fixed_size_matrix<rlmm_float, FFDimension, EmbeddingDimension> V_gate, V_up;
            fixed_size_matrix<rlmm_float, EmbeddingDimension, FFDimension> V_down;
        }
        constexpr NUM_EXPERTS = 8;
        constexpr NUM_TOKEN_DISTRIBUTION_PER_EXPERT = 2; // token sent to this many 'experts'

        using FFN  = std::array<Expert, NUM_EXPERT>;

        on forward pass:
            - copy the most likely / token to expert A and B.
        on backward pass:
            - copy the values back to the most likely experts back