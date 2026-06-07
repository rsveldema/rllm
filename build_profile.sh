#!/usr/bin/env bash
set -euo pipefail

CPU_BUILD_DIR="${CPU_BUILD_DIR:-${BUILD_DIR:-build_profile_cpu}}"
VULKAN_BUILD_DIR="${VULKAN_BUILD_DIR:-build_profile_vulkan}"
VULKAN_STATS_BUILD_DIR="${VULKAN_STATS_BUILD_DIR:-build_profile_vulkan_stats}"
TRAIN_DIR="${TRAIN_DIR:-profile_training_data}"
OUTPUT_MODEL_CPU="${OUTPUT_MODEL_CPU:-/tmp/rllm-profile-small-cpu.json}"
OUTPUT_MODEL_VULKAN="${OUTPUT_MODEL_VULKAN:-/tmp/rllm-profile-small-vulkan.json}"
OUTPUT_MODEL_VULKAN_STATS="${OUTPUT_MODEL_VULKAN_STATS:-/tmp/rllm-profile-small-vulkan-stats.json}"
LAYERS="${LAYERS:-4}"
EPOCHS="${EPOCHS:-3}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-fastfork}"
ALLOW_SOFTWARE_VULKAN_PROFILE="${ALLOW_SOFTWARE_VULKAN_PROFILE:-0}"
export RLLM_HOST_POOL_BYTES="${RLLM_HOST_POOL_BYTES:-2147483648}"

if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "Training directory '$TRAIN_DIR' does not exist. Set TRAIN_DIR to an existing folder."
    exit 1
fi

configure_build() {
    local build_dir="$1"
    local offload_backend="$2"
    local statistics="$3"

    echo "Configuring $offload_backend profiling build in '$build_dir' (statistics: $statistics)..."
    cmake -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPARALLEL_BACKEND="$PARALLEL_BACKEND" \
        -DOFFLOAD_BACKEND="$offload_backend" \
        -DRLLM_ENABLE_STATISTICS="$statistics"
}

build_profiler() {
    local build_dir="$1"

    echo "Building profiling target rllm_prof in '$build_dir'..."
    cmake --build "$build_dir" --parallel --target rllm_prof
}

check_vulkan_provider() {
    local build_dir="$1"
    local provider_output

    provider_output="$("./$build_dir/rllm_prof" --help 2>&1)"
    local provider_line
    provider_line="$(awk '/Offload provider:/ { print; exit }' <<<"$provider_output")"

    if [[ -n "$provider_line" ]]; then
        echo "$provider_line"
    fi

    local provider_lower
    provider_lower="$(tr '[:upper:]' '[:lower:]' <<<"$provider_line")"
    if [[ "$ALLOW_SOFTWARE_VULKAN_PROFILE" != "1" ]] &&
       [[ "$provider_lower" == *llvmpipe* || "$provider_lower" == *lavapipe* || "$provider_lower" == *software* ]]; then
        cat <<EOF

Refusing to run the Vulkan profile on a software Vulkan provider.
This would compare the CPU backend against CPU-emulated Vulkan and is expected to be slower.

Select a hardware Vulkan device with RLLM_VULKAN_DEVICE_INDEX, RLLM_VULKAN_VENDOR, or
RLLM_VULKAN_DEVICE_SUBSTRING. To intentionally profile software Vulkan, rerun with:

  ALLOW_SOFTWARE_VULKAN_PROFILE=1 $0
EOF
        exit 1
    fi
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
vulkan_stats_elapsed_file="$(mktemp)"
cleanup() {
    rm -f "$cpu_elapsed_file" "$vulkan_elapsed_file" "$vulkan_stats_elapsed_file"
}
trap cleanup EXIT

# build version with CPU-only runtime / no stats
configure_build "$CPU_BUILD_DIR" "none" "OFF"
build_profiler "$CPU_BUILD_DIR"

# build version with vulkan / no stats
#configure_build "$VULKAN_BUILD_DIR" "vulkan" "OFF"
#build_profiler "$VULKAN_BUILD_DIR"
#check_vulkan_provider "$VULKAN_BUILD_DIR"

# build version with vulkan / stats
configure_build "$VULKAN_STATS_BUILD_DIR" "vulkan" "ON"
build_profiler "$VULKAN_STATS_BUILD_DIR"
check_vulkan_provider "$VULKAN_STATS_BUILD_DIR"

run_training "CPU-only runtime" "$CPU_BUILD_DIR" "$OUTPUT_MODEL_CPU" "$cpu_elapsed_file"
#run_training "Vulkan runtime" "$VULKAN_BUILD_DIR" "$OUTPUT_MODEL_VULKAN" "$vulkan_stats_elapsed_file"
run_training "Vulkan statistics" "$VULKAN_STATS_BUILD_DIR" "$OUTPUT_MODEL_VULKAN_STATS" "$vulkan_stats_elapsed_file"

cpu_elapsed="$(<"$cpu_elapsed_file")"
vulkan_elapsed="$(<"$vulkan_stats_elapsed_file")"
vulkan_stats_elapsed="$(<"$vulkan_stats_elapsed_file")"
speedup="$(awk -v cpu="$cpu_elapsed" -v vulkan="$vulkan_elapsed" 'BEGIN { if (vulkan > 0) printf "%.3f", cpu / vulkan; else printf "inf" }')"

echo
echo "Training profile comparison:"
echo "  CPU-only runtime elapsed:     ${cpu_elapsed}s (statistics disabled)"
echo "  Vulkan runtime elapsed:       ${vulkan_elapsed}s (statistics enabled)"
echo "  Vulkan runtime speedup:       ${speedup}x"
echo "  Vulkan statistics elapsed:    ${vulkan_stats_elapsed}s (statistics enabled)"
