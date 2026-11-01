#!/usr/bin/env bash
set -euo pipefail

# Keep generated kernels fresh when switching between source and test codegen.
rm -rf build_debug/generated/hipized/kernels

cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug -DPARALLEL_BACKEND=sequential -DOFFLOAD_BACKEND=hip "$@"
cmake --build build_debug --parallel
