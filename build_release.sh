#!/usr/bin/env bash
set -euo pipefail

cmake -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel
