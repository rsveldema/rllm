This is an experimental LLM.
Just a vehicle to experiment with.

Build Guide
===============

Build requirements (minimum):

- CMake 3.20+
- C++23 compiler
- Python 3 (used during tokenizer code generation)

```bash
git submodules update --init --recursive
sudo apt install libhwloc-dev
sudo apt install glslc
```

Offload dispatch note:

- In `vulkan` offload mode, kernel launch helpers dispatch through kernel_compiler generated stubs and SPIR-V.
- Set `NAN_FINDING_MODE=1` to run extra offloaded checks around `TransformerBlock` and `OutputLayer` forward/backward passes. The checks abort if model weights or optimizer velocities become NaN or leave their clamp ranges.
- Set `FRESH_START=1` when running `train_release.sh` to ignore existing checkpoints and start from random weights.

Compatibility matrix:

- fastfork + none: supported
- fastfork + vulkan: supported
- openmp + none: supported
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

FastFork + Vulkan offload mode build:

```bash
cmake -S . -B build/fastfork-vulkan -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=vulkan
cmake --build build/fastfork-vulkan --parallel
```

Vulkan device selection (optional):

```bash
# Select the device used by rllm by a case-insensitive name substring
./build/fastfork-vulkan/rllm --vulkan-device radeon

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
            fixed_size_matrix<float16, FFDimension, EmbeddingDimension> W_gate, W_up;
            fixed_size_matrix<float16, EmbeddingDimension, FFDimension> W_down;
            fixed_size_matrix<float, FFDimension, EmbeddingDimension> V_gate, V_up;
            fixed_size_matrix<float, EmbeddingDimension, FFDimension> V_down;
        }
        constexpr NUM_EXPERTS = 8;
        constexpr NUM_TOKEN_DISTRIBUTION_PER_EXPERT = 2; // token sent to this many 'experts'

        using FFN  = std::array<Expert, NUM_EXPERT>;

        on forward pass:
            - copy the most likely / token to expert A and B.
        on backward pass:
            - copy the values back to the most likely experts back

======
valid value ranges, change types to match:

compute_score() now aborts if finite CE loss exceeds 1000, printing target logit, max logit, sum_exp, log prob, and loss.
NAN_FINDING_MODE activation scans now use sane bounds:InputLayer hidden state: [-2, 2]
input gradient: [-10000, 10000]
TextTrainer hidden states / h_last: [-10000, 10000]
OutputLayer logits: [-1000000, 1000000]

--- Training
During training, what matters is the trend:
Validation loss continues falling: training is working.
The validation loss is logged after each epoch.

Training loss falls but validation stays near 5.1: overfitting or distribution mismatch.
Validation rises for several epochs: the learning rate may be too high or training is overfitting.
Validation settles substantially below 3: predictions are becoming useful.
Below 2 would be strong for this corpus, though achievable loss depends heavily on ambiguous prefixes.