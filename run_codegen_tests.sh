#!/usr/bin/env bash
# Run the Python unit tests for the codegen package.
# Usage: ./run_codegen_tests.sh [--verbose] [--junitxml=path]

set -euo pipefail

cd "$(dirname "$0")"  # change to project root

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose|-v)
            EXTRA_ARGS+=(-v)
            shift
            ;;
        --junitxml)
            EXTRA_ARGS+=(--junitxml="$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1 (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

python3 -m pytest tests/test_codegen.py "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
