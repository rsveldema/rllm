#!/usr/bin/env bash

# Shared implementation for train_debug.sh and train_release.sh.
# This file is meant to be sourced by a small build-specific wrapper.

train_rllm() {
    local build_type="$1"
    shift

    local strip_comments=0
    local -a forwarded_args=()
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--strip-comments" ]]; then
            strip_comments=1
        else
            forwarded_args+=("$arg")
        fi
    done
    set -- "${forwarded_args[@]}"

    local repository_root
    repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$repository_root"

    local num_layers=4
    local expect_layers_value=0
    for arg in "$@"; do
        if (( expect_layers_value )); then
            num_layers="$arg"
            expect_layers_value=0
        elif [[ "$arg" == "--layers" ]]; then
            expect_layers_value=1
        fi
    done
    if [[ ! "$num_layers" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid layer count '$num_layers'." >&2
        return 2
    fi
    local window_size=$((num_layers * 16))
    if (( window_size > 256 )); then
        window_size=256
    fi
    local window_stride=$((window_size / 2))
    if (( window_stride < 1 )); then
        window_stride=1
    fi
    local learn_depth=$((num_layers / 2))
    if (( learn_depth < 1 )); then
        learn_depth=1
    fi

    local model_dir="${RLLM_MODEL_DIR:-models-$num_layers}"
    local previous_model_dir=""
    local previous_model_layers=0
    local candidate_dir candidate_layers
    shopt -s nullglob
    for candidate_dir in models-[0-9]*; do
        [[ -d "$candidate_dir" ]] || continue
        [[ -f "$candidate_dir/checkpoint-best-window.st" ||
           -f "$candidate_dir/after_training.st" ]] || continue
        candidate_layers="${candidate_dir#models-}"
        [[ "$candidate_layers" =~ ^[1-9][0-9]*$ ]] || continue
        if (( candidate_layers < num_layers && candidate_layers > previous_model_layers )); then
            previous_model_layers="$candidate_layers"
            previous_model_dir="$candidate_dir"
        fi
    done
    shopt -u nullglob
    mkdir -p "$model_dir"
    export RLLM_MODEL_DIR="$model_dir"
    echo "Training shape: $num_layers layers, learn depth $learn_depth, window size $window_size, stride $window_stride"
    echo "Model artifacts: $model_dir/"

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
        for checkpoint in "$model_dir"/checkpoint-[0-9]*.st; do
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
    local stripped_corpus_root=""
    if (( strip_comments )); then
        stripped_corpus_root="$(mktemp -d "${TMPDIR:-/tmp}/rllm-training-no-comments.XXXXXX")"
        trap 'rm -rf -- "$stripped_corpus_root"' RETURN
        echo "Comment-free corpus copies: $stripped_corpus_root"
    fi
    local source_index=0
    for source in "${train_sources[@]}"; do
        path="${source%:*}"
        if [[ "$path" == "$source" ]]; then
            path="$source"
        fi
        if [[ ! -d "$path" ]]; then
            echo "Training directory '$path' does not exist." >&2
            return 1
        fi
        if (( strip_comments )); then
            local stripped_path="$stripped_corpus_root/source-$source_index"
            echo "Stripping comments from $path..."
            python3 ./training_postprocessor.py --dir "$path" \
                --output-dir "$stripped_path" --strip-comments
            if [[ "$path" == "$source" ]]; then
                train_dir_args+=(--train-dir "$stripped_path")
            else
                train_dir_args+=(--train-dir "$stripped_path:${source##*:}")
            fi
            source_index=$((source_index + 1))
        else
            train_dir_args+=(--train-dir "$source")
        fi
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
    elif [[ -f "$model_dir/checkpoint-best-window.st" ]] &&
            resume_arg_for_model "$model_dir/checkpoint-best-window.st"; then
        echo "Resuming from $model_dir/checkpoint-best-window.st"
        resume_args=(-i "$model_dir/checkpoint-best-window.st")
        delete_superseded_checkpoints
    elif [[ -f "$model_dir/after_training.st" ]] &&
            resume_arg_for_model "$model_dir/after_training.st"; then
        echo "Resuming from $model_dir/after_training.st"
        resume_args=(-i "$model_dir/after_training.st")
    else
        shopt -s nullglob
        local checkpoint
        for checkpoint in "$model_dir"/checkpoint-*.st; do
            if [[ -z "$latest_checkpoint" || "$checkpoint" -nt "$latest_checkpoint" ]]; then
                latest_checkpoint="$checkpoint"
            fi
        done
        shopt -u nullglob

        if [[ -n "$latest_checkpoint" ]] && resume_arg_for_model "$latest_checkpoint"; then
            echo "Resuming from $latest_checkpoint"
            resume_args=(-i "$latest_checkpoint")
        elif [[ -n "$previous_model_dir" && -f "$previous_model_dir/checkpoint-best-window.st" ]] &&
                resume_arg_for_model "$previous_model_dir/checkpoint-best-window.st"; then
            echo "Upgrading from $previous_model_dir/checkpoint-best-window.st"
            resume_args=(-i "$previous_model_dir/checkpoint-best-window.st")
        elif [[ -n "$previous_model_dir" && -f "$previous_model_dir/after_training.st" ]] &&
                resume_arg_for_model "$previous_model_dir/after_training.st"; then
            echo "Upgrading from $previous_model_dir/after_training.st"
            resume_args=(-i "$previous_model_dir/after_training.st")
        else
            echo "No compatible checkpoint found, starting from random weights."
        fi
    fi

    local -a layer_upgrade_args=()
    if (( ${#resume_args[@]} != 0 )); then
        layer_upgrade_args=(--upgrade-mode --freeze-old-blocks)
        echo "Layer upgrade defaults: increase checkpoint depth with the original blocks read-only."
    fi

    echo "--- Starting $build_type training ---"
    # Scale context and sampling stride together as model depth increases.
    "./$build_dir/rllm" --train "${resume_args[@]}" "${layer_upgrade_args[@]}" \
        -o "$model_dir/after_training.st" \
        "${train_dir_args[@]}" \
        --method "window:$window_size" \
        --window-stride "$window_stride" \
        --epochs 40 \
        --disable-example-convergence \
        --layers "$num_layers" \
        --checkpoint-interval 600 \
        --validation-interval 600 \
        --learn-depth "$learn_depth" \
        --learning-rate 0.00001 \
        --layer-learning-rate-multiplier 1.05 \
        --warmup-percent 10 \
        --weight-initializer xavier-input-projections \
        --ffn-initializer xavier-input-projections \
        --embedding-initializer legacy-uniform \
        --learning-rate-schedule lowering \
        --simulated-annealing-initial-multiplier 9 \
        --simulated-annealing-decay-factor 0.7 \
        --simulated-annealing-decay-epochs 1 \
        --simulated-annealing-min-multiplier 0.02 \
        --micro-batch-size 256 \
        --max-validation-windows 512 \
        --validation-worst-count 100 \
        --vulkan-device R9700 \
        --restart-learning-rate-schedule \
                "$@"
}


#        --all-blocks-read-write \
