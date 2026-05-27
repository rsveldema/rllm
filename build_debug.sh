#!/usr/bin/env bash
set -euo pipefail

cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --parallel
