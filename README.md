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
- HIP offload mode: HIP headers/runtime (HIP_ROOT defaults to /opt/rocm)

Parallel backends:

- fastfork
- openmp
- sequential

Offload modes:

- none
- hip
- vulkan

Offload dispatch note:

- In `hip`/`vulkan` offload modes, kernel launch helpers fail fast if no real backend dispatch is implemented.

Compatibility matrix:

- fastfork + none: supported
- fastfork + hip: supported
- fastfork + vulkan: supported
- openmp + none: supported
- openmp + hip: supported
- sequential + none: supported
- sequential + vulkan: supported
- openmp + vulkan: supported

Quick start with CMake presets
===============

List presets:

```bash
cmake --list-presets
```

Configure + build default FastFork:

```bash
cmake --preset default-fastfork
cmake --build --preset build-default-fastfork --parallel
```

Configure + build FastFork with Vulkan offload mode:

```bash
cmake --preset fastfork-vulkan
cmake --build --preset build-fastfork-vulkan --parallel
```

Configure + build OpenMP with HIP offload mode:

```bash
cmake --preset openmp-hip
cmake --build --preset build-openmp-hip --parallel
```

Configure + build FastFork with HIP offload mode:

```bash
cmake --preset fastfork-hip
cmake --build --preset build-fastfork-hip --parallel
```

Manual CMake configuration (without presets)
===============

Default FastFork build:

```bash
cmake -S . -B build/default-fastfork -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=none
cmake --build build/default-fastfork --parallel
```

OpenMP build:

```bash
cmake -S . -B build/openmp -DPARALLEL_BACKEND=openmp -DOFFLOAD_BACKEND=none
cmake --build build/openmp --parallel
```

OpenMP + HIP offload build (HIP under /opt/rocm):

```bash
cmake -S . -B build/openmp-hip -DPARALLEL_BACKEND=openmp -DOFFLOAD_BACKEND=hip -DHIP_ROOT=/opt/rocm
cmake --build build/openmp-hip --parallel
```

FastFork + HIP offload build (HIP under /opt/rocm):

```bash
cmake -S . -B build/fastfork-hip -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=hip -DHIP_ROOT=/opt/rocm
cmake --build build/fastfork-hip --parallel
```

FastFork + Vulkan offload mode build:

```bash
cmake -S . -B build/fastfork-vulkan -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=vulkan
cmake --build build/fastfork-vulkan --parallel
```

Vulkan device selection (optional):

```bash
# Prefer AMD Vulkan devices (vendor id 0x1002)
RLLM_VULKAN_VENDOR=amd ./build/fastfork-vulkan/rllm --help

# Select a specific enumerated compute-capable Vulkan device index
RLLM_VULKAN_DEVICE_INDEX=1 ./build/fastfork-vulkan/rllm --help

# Match by case-insensitive device name substring
RLLM_VULKAN_DEVICE_SUBSTRING=radeon ./build/fastfork-vulkan/rllm --help
```

Selection priority is: `RLLM_VULKAN_DEVICE_INDEX` first, then `RLLM_VULKAN_VENDOR` and/or
`RLLM_VULKAN_DEVICE_SUBSTRING`, then automatic best-device scoring.

Sequential build:

```bash
cmake -S . -B build/sequential -DPARALLEL_BACKEND=sequential -DOFFLOAD_BACKEND=none
cmake --build build/sequential --parallel
```

Installation
==============

```bash
    sudo dnf install hwloc-devel
```

How to use
=============

To train:

```bash
 ./build/rllm --train --filter simple --method window3 --epochs 10
```

Prompt mode:

```bash
 ./build/rllm
```

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