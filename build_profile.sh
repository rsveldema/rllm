#!/usr/bin/env bash
set -euo pipefail

CPU_BUILD_DIR="${CPU_BUILD_DIR:-${BUILD_DIR:-build_profile_cpu}}"
VULKAN_BUILD_DIR="${VULKAN_BUILD_DIR:-build_profile_vulkan}"
TRAIN_DIR="${TRAIN_DIR:-profile_training_data}"
OUTPUT_MODEL_CPU="${OUTPUT_MODEL_CPU:-/tmp/rllm-profile-small-cpu.json}"
OUTPUT_MODEL_VULKAN="${OUTPUT_MODEL_VULKAN:-/tmp/rllm-profile-small-vulkan.json}"
LAYERS="${LAYERS:-1}"
EPOCHS="${EPOCHS:-10}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-fastfork}"

if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "Training directory '$TRAIN_DIR' does not exist. Set TRAIN_DIR to an existing folder."
    exit 1
fi

configure_build() {
    local build_dir="$1"
    local offload_backend="$2"

    echo "Configuring $offload_backend profiling build in '$build_dir'..."
    cmake -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPARALLEL_BACKEND="$PARALLEL_BACKEND" \
        -DOFFLOAD_BACKEND="$offload_backend"
}

build_profiler() {
    local build_dir="$1"

    echo "Building profiling target rllm_prof in '$build_dir'..."
    cmake --build "$build_dir" --parallel --target rllm_prof
}

run_training() {
    local label="$1"
    local build_dir="$2"
    local output_model="$3"
    local elapsed_file="$4"

    echo "Running $label profiling training batch..."
    /usr/bin/time -f '%e' -o "$elapsed_file" \
        "./$build_dir/rllm_prof" \
        --train \
        --train-dir "$TRAIN_DIR" \
        --layers "$LAYERS" \
        --epochs "$EPOCHS" \
        --checkpoint-interval 0 \
        -o "$output_model"
}

cpu_elapsed_file="$(mktemp)"
vulkan_elapsed_file="$(mktemp)"
cleanup() {
    rm -f "$cpu_elapsed_file" "$vulkan_elapsed_file"
}
trap cleanup EXIT

configure_build "$CPU_BUILD_DIR" "none"
build_profiler "$CPU_BUILD_DIR"

configure_build "$VULKAN_BUILD_DIR" "vulkan"
build_profiler "$VULKAN_BUILD_DIR"

run_training "CPU-only" "$CPU_BUILD_DIR" "$OUTPUT_MODEL_CPU" "$cpu_elapsed_file"
run_training "Vulkan" "$VULKAN_BUILD_DIR" "$OUTPUT_MODEL_VULKAN" "$vulkan_elapsed_file"

cpu_elapsed="$(<"$cpu_elapsed_file")"
vulkan_elapsed="$(<"$vulkan_elapsed_file")"
speedup="$(awk -v cpu="$cpu_elapsed" -v vulkan="$vulkan_elapsed" 'BEGIN { if (vulkan > 0) printf "%.3f", cpu / vulkan; else printf "inf" }')"

echo
echo "Training profile comparison:"
echo "  CPU-only elapsed: ${cpu_elapsed}s"
echo "  Vulkan elapsed:   ${vulkan_elapsed}s"
echo "  Vulkan speedup:   ${speedup}x"
