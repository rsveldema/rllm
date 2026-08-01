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

    local -a train_sources
    if [[ -n "${TRAIN_DIR:-}" ]]; then
        train_sources=("$TRAIN_DIR")
    else
        train_sources=(
            "training_data0:0.05"
            "curriculum/grammar:0.15"
            "curriculum/syntax:0.10"
            "curriculum/comments:0.10"
            "curriculum/systems:0.10"
            "training_data2:0.50"
        )
    fi
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

    for corpus_root in training_data0 curriculum training_data2; do
        echo "Normalizing $corpus_root with training_postprocessor.py..."
        python3 ./training_postprocessor.py --dir "$corpus_root"
    done

    local -a train_dir_args=()
    local source path
    for source in "${train_sources[@]}"; do
        path="${source%:*}"
        if [[ "$path" == "$source" ]]; then
            path="$source"
        fi
        if [[ ! -d "$path" ]]; then
            echo "Training directory '$path' does not exist." >&2
            return 1
        fi
        train_dir_args+=(--train-dir "$source")
    done

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
    # At stride 96, window:256 produces about half as many windows as
    # window:128. The packed-row cap also halves the effective micro-batch
    # (64 to 32), preserving optimizer updates and LR schedule steps.
    "./$build_dir/rllm" --train "${resume_args[@]}" \
        -o models/after_training.st \
        "${train_dir_args[@]}" \
        --method window:256 \
        --window-stride 96 \
        --epochs 160 \
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
        --max-validation-windows 4096 \
        --validation-worst-count 100 \
        --vulkan-device R9700 \
        "$@"
}
