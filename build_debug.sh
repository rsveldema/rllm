#!/usr/bin/env bash
set -euo pipefail

cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug -DPARALLEL_BACKEND=fastfork -DOFFLOAD_BACKEND=none
cmake --build build_debug --parallel
