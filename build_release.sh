#!/usr/bin/env bash
set -euo pipefail

# Keep generated kernels fresh when switching between source and test codegen.
rm -rf build_release/generated/vulkanized/kernels

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
if [[ "$PYTHON" != /* ]]; then
    PYTHON="$(pwd)/$PYTHON"
fi

cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DPARALLEL_BACKEND=sequential -DOFFLOAD_BACKEND=vulkan -DBUILD_TESTING=ON -DRLLM_BUILD_KERNEL_COMPILER=ON -DPython3_EXECUTABLE:FILEPATH="$PYTHON"
cmake --build build_release --parallel
