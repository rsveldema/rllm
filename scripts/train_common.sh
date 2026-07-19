#!/usr/bin/env bash

# Shared implementation for train_debug.sh and train_release.sh.
# This file is meant to be sourced by a small build-specific wrapper.

train_rllm() {
    local build_type="$1"
    shift

    local repository_root
    repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$repository_root"

    local build_dir
    case "$build_type" in
        debug)
            build_dir="build_debug"
            echo "Configuring and building debug before training..."
            ./build_debug.sh -DDEBUG_ATTENTION_DIAGNOSTICS=ON
            ;;
        release)
            build_dir="build_release"
            echo "Configuring and building release before training..."
            ./build_release.sh -DDEBUG_ATTENTION_DIAGNOSTICS=ON
            ;;
        *)
            echo "Unknown training build type '$build_type' (expected debug or release)." >&2
            return 2
            ;;
    esac

    local train_dir="${TRAIN_DIR:-training_data1}"
    local runtime_tokenizer_header="$build_dir/generated/tokenizer_map.hpp"

    runtime_vocab_size() {
        python3 ./runtime_vocab_size.py "$runtime_tokenizer_header"
    }

    model_vocab_size() {
        python3 ./model_vocab_size.py "$1"
    }

    resume_arg_for_model() {
        local candidate="$1"
        local required="${2:-0}"
        local runtime_vocab
        local model_vocab

        runtime_vocab="$(runtime_vocab_size)"
        model_vocab="$(model_vocab_size "$candidate")"

        if [[ "$model_vocab" == ERROR:* ]]; then
            echo "Cannot inspect resume model '$candidate': ${model_vocab#ERROR:}" >&2
            if [[ "$required" != "0" ]]; then
                return 2
            fi
            return 1
        fi

        if [[ -n "$model_vocab" && "$model_vocab" != "$runtime_vocab" ]]; then
            echo "Skipping incompatible resume model '$candidate' (model vocab=$model_vocab, runtime vocab=$runtime_vocab)." >&2
            if [[ "$required" != "0" ]]; then
                return 2
            fi
            return 1
        fi

        return 0
    }

    delete_superseded_checkpoints() {
        local checkpoint

        shopt -s nullglob
        for checkpoint in models/checkpoint-[0-9]*.st; do
            echo "Deleting superseded checkpoint $checkpoint"
            rm -f -- "$checkpoint"
        done
        shopt -u nullglob
    }

    echo "Normalizing training_data1 with training_postprocessor.py..."
    python3 ./training_postprocessor.py --dir training_data1

    echo "Formatting training_data0/*.cpp with maximum line length..."
    if compgen -G "training_data0/*.cpp" > /dev/null; then
        clang-format -i --style='{BasedOnStyle: LLVM, ColumnLimit: 0}' training_data0/*.cpp
    else
        echo "No .cpp files found in training_data0"
    fi

    if [[ ! -d "$train_dir" ]]; then
        echo "Training directory '$train_dir' does not exist. Set TRAIN_DIR to an existing folder." >&2
        return 1
    fi

    local -a resume_args=()
    local latest_checkpoint=""
    echo "Locating checkpoint to resume from..."
    if [[ "${FRESH_START:-0}" != "0" ]]; then
        echo "FRESH_START=${FRESH_START}: ignoring existing checkpoints and starting from random weights."
    elif [[ -n "${RESUME_MODEL:-}" ]]; then
        if [[ ! -f "$RESUME_MODEL" ]]; then
            echo "Explicit resume model '$RESUME_MODEL' does not exist." >&2
            return 1
        fi
        echo "Resuming from $RESUME_MODEL"
        resume_arg_for_model "$RESUME_MODEL" 1 || return $?
        resume_args=(-i "$RESUME_MODEL")
    elif [[ -f "models/checkpoint-best-window.st" ]] &&
            resume_arg_for_model "models/checkpoint-best-window.st"; then
        echo "Resuming from models/checkpoint-best-window.st"
        resume_args=(-i "models/checkpoint-best-window.st")
        delete_superseded_checkpoints
    elif [[ -f "models/after_training.st" ]] &&
            resume_arg_for_model "models/after_training.st"; then
        echo "Resuming from models/after_training.st"
        resume_args=(-i "models/after_training.st")
    else
        shopt -s nullglob
        local checkpoint
        for checkpoint in models/checkpoint-*.st; do
            if [[ -z "$latest_checkpoint" || "$checkpoint" -nt "$latest_checkpoint" ]]; then
                latest_checkpoint="$checkpoint"
            fi
        done
        shopt -u nullglob

        if [[ -n "$latest_checkpoint" ]] && resume_arg_for_model "$latest_checkpoint"; then
            echo "Resuming from $latest_checkpoint"
            resume_args=(-i "$latest_checkpoint")
        else
            echo "No compatible checkpoint found, starting from random weights."
        fi
    fi

    echo "--- Starting $build_type training ---"
    "./$build_dir/rllm" --train "${resume_args[@]}" \
        -o models/after_training.st \
        --train-dir "$train_dir" \
        --filter simple \
        --method window:32 \
        --window-stride 1 \
        --epochs 80 \
        --disable-example-convergence \
        --layers 8 \
        --checkpoint-interval 300 \
        --learn-depth 4 \
        --learning-rate 0.00001 \
        --layer-learning-rate-multiplier 1.05 \
        --weight-initializer xavier-input-projections \
        --ffn-initializer xavier-input-projections \
        --embedding-initializer legacy-uniform \
        --learning-rate-schedule lowering \
        --simulated-annealing-initial-multiplier 9 \
        --simulated-annealing-decay-factor 0.7 \
        --simulated-annealing-decay-epochs 1 \
        --simulated-annealing-min-multiplier 0.02 \
        --micro-batch-size 256 \
        --vulkan-device R9700 \
        "$@"
}
