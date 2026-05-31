#!/usr/bin/env bash
set -euo pipefail

cmake -B build_release -DCMAKE_BUILD_TYPE=Release -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=vulkan
cmake --build build_release --parallel
