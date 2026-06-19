#!/usr/bin/env bash
# Run the Python unit tests for the codegen package.
# Usage: ./run_codegen_tests.sh [--verbose] [--junitxml=path]

set -euo pipefail

cd "$(dirname "$0")"  # change to project root

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

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

"$PYTHON" -m pytest kernel_compiler/tests/test_codegen.py "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
