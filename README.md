This is an experimental LLM.
Just a vehicle to experiment with.

Build Guide
===============

Build requirements (minimum):

- CMake 3.20+
- C++23 compiler
- Python 3 (used during tokenizer code generation)

Backend/offload dependencies:

- FastFork backend: hwloc development package
    - Fedora: sudo dnf install hwloc-devel
- OpenMP backend: OpenMP runtime/dev package for your compiler
- ROCm offload mode: ROCm headers/runtime (ROCM_ROOT defaults to /opt/rocm)

Parallel backends:

- fastfork
- openmp
- sequential

Offload modes:

- none
- rocm (requires openmp backend)
- vulkan (macro mode currently uses CPU fallback loops)

Compatibility matrix:

- fastfork + none: supported
- fastfork + vulkan: supported
- openmp + none: supported
- openmp + rocm: supported
- sequential + none: supported
- sequential + vulkan: supported
- openmp + vulkan: supported

Note: rocm offload mode requires PARALLEL_BACKEND=openmp.

Quick start with CMake presets
===============

List presets:

    cmake --list-presets

Configure + build default FastFork:

    cmake --preset default-fastfork
    cmake --build --preset build-default-fastfork --parallel

Configure + build FastFork with Vulkan offload mode:

    cmake --preset fastfork-vulkan
    cmake --build --preset build-fastfork-vulkan --parallel

Configure + build OpenMP with ROCm offload mode:

    cmake --preset openmp-rocm
    cmake --build --preset build-openmp-rocm --parallel

Manual CMake configuration (without presets)
===============

Default FastFork build:

    cmake -S . -B build/default-fastfork -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=none
    cmake --build build/default-fastfork --parallel

OpenMP build:

    cmake -S . -B build/openmp -DPARALLEL_BACKEND=openmp -DOFFLOAD_BACKEND=none
    cmake --build build/openmp --parallel

OpenMP + ROCm offload build (ROCm under /opt/rocm):

    cmake -S . -B build/openmp-rocm -DPARALLEL_BACKEND=openmp -DOFFLOAD_BACKEND=rocm -DROCM_ROOT=/opt/rocm
    cmake --build build/openmp-rocm --parallel

FastFork + Vulkan offload mode build:

    cmake -S . -B build/fastfork-vulkan -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=vulkan
    cmake --build build/fastfork-vulkan --parallel

Sequential build:

    cmake -S . -B build/sequential -DPARALLEL_BACKEND=sequential -DOFFLOAD_BACKEND=none
    cmake --build build/sequential --parallel

Installation
==============

```bash
    sudo dnf install hwloc-devel
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
            fixed_size_matrix<rlmm_float_small, FFDimension, EmbeddingDimension> W_gate, W_up;
            fixed_size_matrix<rlmm_float_small, EmbeddingDimension, FFDimension> W_down;
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