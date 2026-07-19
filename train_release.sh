#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repository_root/scripts/train_common.sh"
train_rllm release "$@"
