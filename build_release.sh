#!/usr/bin/env bash
set -euo pipefail

# Release builds do not need tests; keeping them off avoids Vulkan test-kernel
# generation from polluting the main release kernel umbrella header.
rm -rf build_release/generated/vulkanized/kernels

cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=vulkan -DBUILD_TESTING=OFF
cmake --build build_release --parallel
