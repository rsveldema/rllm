#!/usr/bin/env python3
"""Build and launch an rLLM training run.

This replaces the debug/release shell wrappers and their shared sourced script.
The selected JSON file is the sole source of build and training options.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def positive_integer(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Config field '{name}' must be a positive integer.")
    return value


def string_list(config: dict[str, object], name: str) -> tuple[str, ...]:
    value = config.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Config field '{name}' must be an array of strings.")
    return tuple(value)


def load_config(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
    except OSError as error:
        raise ValueError(f"Cannot read config file '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in config file '{path}': {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{path}' must contain a JSON object.")
    for name in ("layers", "window_size", "window_stride", "learn_depth"):
        positive_integer(config, name)
    for name in ("sources", "cmake_arguments", "training_arguments"):
        string_list(config, name)
    if not isinstance(config.get("strip_comments"), bool):
        raise ValueError("Config field 'strip_comments' must be a boolean.")
    if config.get("build_type") not in ("debug", "release"):
        raise ValueError("Config field 'build_type' must be 'debug' or 'release'.")
    if not isinstance(config.get("model_directory"), str) or not config["model_directory"]:
        raise ValueError("Config field 'model_directory' must be a non-empty string.")
    if "incremental_window" in config and not isinstance(config["incremental_window"], bool):
        raise ValueError("Config field 'incremental_window' must be a boolean.")
    return config


def split_source(source: str) -> tuple[str, str | None]:
    path, separator, weight = source.rpartition(":")
    if separator and path and weight:
        return path, weight
    return source, None


def previous_model_directory(num_layers: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for directory in ROOT.glob("models-[0-9]*"):
        suffix = directory.name.removeprefix("models-")
        if not directory.is_dir() or not suffix.isdigit():
            continue
        depth = int(suffix)
        if depth >= num_layers:
            continue
        if ((directory / "checkpoint-best-window.st").is_file()
                or (directory / "after_training.st").is_file()):
            candidates.append((depth, directory))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def output(command: list[str]) -> str:
    return run(command, capture=True).stdout.strip()


def compatible_model(candidate: Path, runtime_header: Path, *, required: bool = False) -> bool:
    runtime_vocab = output([sys.executable, str(ROOT / "runtime_vocab_size.py"), str(runtime_header)])
    model_vocab = output([sys.executable, str(ROOT / "model_vocab_size.py"), str(candidate)])
    if model_vocab.startswith("ERROR:"):
        print(f"Cannot inspect resume model '{candidate}': {model_vocab.removeprefix('ERROR:')}", file=sys.stderr)
        if required:
            raise RuntimeError("explicit resume model could not be inspected")
        return False
    if model_vocab and model_vocab != runtime_vocab:
        print(
            f"Skipping incompatible resume model '{candidate}' "
            f"(model vocab={model_vocab}, runtime vocab={runtime_vocab}).",
            file=sys.stderr,
        )
        if required:
            raise RuntimeError("explicit resume model has an incompatible vocabulary")
        return False
    return True


def clean_intermediate_checkpoints(model_dir: Path) -> Path:
    numbered = list(model_dir.glob("checkpoint-[0-9]*.st"))
    destination = model_dir / "checkpoint-latest.st"
    destination_sidecar = Path(f"{destination}.training.json")
    candidates = numbered + ([destination] if destination.is_file() else [])
    if not candidates:
        raise FileNotFoundError(f"No numbered checkpoint found in '{model_dir}'.")

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    latest_sidecar = Path(f"{latest}.training.json")
    if latest == destination:
        print(f"Preserving existing latest checkpoint {destination}")
    else:
        print(f"Preserving latest checkpoint {latest} as {destination}")
        latest.replace(destination)
        if latest_sidecar.is_file():
            latest_sidecar.replace(destination_sidecar)
        elif destination_sidecar.is_file():
            destination_sidecar.unlink()

    for checkpoint in numbered:
        if checkpoint == latest:
            continue
        if checkpoint.is_file():
            print(f"Deleting intermediate checkpoint {checkpoint}")
            checkpoint.unlink()
    for sidecar in model_dir.glob("checkpoint-[0-9]*.st.training.json"):
        print(f"Deleting intermediate checkpoint sidecar {sidecar}")
        sidecar.unlink()
    return destination


def lowest_loss_checkpoint(model_dir: Path) -> tuple[Path, float]:
    progress_file = model_dir / "train.json"
    try:
        with progress_file.open(encoding="utf-8") as file:
            progress = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise FileNotFoundError(
            f"Cannot read validation history '{progress_file}': {error}"
        ) from error
    if not isinstance(progress, list):
        raise FileNotFoundError(f"Validation history '{progress_file}' is not a JSON array.")

    timestamped_checkpoints: list[tuple[int, Path]] = []
    for checkpoint in model_dir.glob("checkpoint-[0-9]*.st"):
        timestamp_text = checkpoint.stem.removeprefix("checkpoint-")
        if timestamp_text.isdigit():
            timestamped_checkpoints.append((int(timestamp_text), checkpoint))
    timestamped_checkpoints.sort()

    # Timed validation checkpoints are created immediately after evaluation;
    # allow a small filesystem/logging delay but do not associate an unrelated
    # checkpoint from the next checkpoint interval.
    max_pairing_delay_ms = 5_000
    candidates: list[tuple[float, int, Path]] = []
    for entry in progress:
        if not isinstance(entry, dict) or entry.get("item_type") != "validation":
            continue
        loss = entry.get("validation_loss")
        timestamp = entry.get("timestamp_ms")
        if (not isinstance(loss, (int, float)) or isinstance(loss, bool) or
                not math.isfinite(loss) or not isinstance(timestamp, int)):
            continue
        checkpoint_match = next(
            ((checkpoint_timestamp, checkpoint)
             for checkpoint_timestamp, checkpoint in timestamped_checkpoints
             if 0 <= checkpoint_timestamp - timestamp <= max_pairing_delay_ms),
            None,
        )
        if checkpoint_match is not None:
            candidates.append((float(loss), checkpoint_match[0], checkpoint_match[1]))

    if not candidates:
        raise FileNotFoundError(
            f"No numbered checkpoint in '{model_dir}' is paired with a validation result."
        )
    loss, _, checkpoint = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return checkpoint, loss


def select_resume_model(
    model_dir: Path,
    previous_dir: Path | None,
    runtime_header: Path,
    *,
    fresh_start: bool,
    latest: bool,
    resume_model: str | None,
    lowest_loss: bool = False,
) -> Path | None:
    print("Locating checkpoint to resume from...")
    if fresh_start:
        print("Fresh start configured: ignoring existing checkpoints and starting from random weights.")
        return None

    if resume_model is not None:
        candidate = Path(resume_model)
        if not candidate.is_file():
            raise FileNotFoundError(f"Explicit resume model '{candidate}' does not exist.")
        compatible_model(candidate, runtime_header, required=True)
        print(f"Resuming from {candidate}")
        return candidate

    if lowest_loss:
        candidate, loss = lowest_loss_checkpoint(model_dir)
        compatible_model(candidate, runtime_header, required=True)
        print(f"Resuming from lowest-loss checkpoint {candidate} (validation loss {loss:.6f})")
        return candidate

    if latest:
        checkpoints = list(model_dir.glob("checkpoint-[0-9]*.st"))
        candidate = max(checkpoints, key=lambda path: path.stat().st_mtime, default=None)
        if candidate is None:
            cleaned_latest = model_dir / "checkpoint-latest.st"
            candidate = cleaned_latest if cleaned_latest.is_file() else None
        if candidate is not None and compatible_model(candidate, runtime_header, required=True):
            print(f"Resuming from latest checkpoint {candidate}")
            return candidate
        raise FileNotFoundError(f"No latest checkpoint found in '{model_dir}'.")

    best = model_dir / "checkpoint-best-window.st"
    if best.is_file() and compatible_model(best, runtime_header):
        print(f"Resuming from {best}")
        return best

    final = model_dir / "after_training.st"
    if final.is_file() and compatible_model(final, runtime_header):
        print(f"Resuming from {final}")
        return final

    checkpoints = list(model_dir.glob("checkpoint-*.st"))
    latest = max(checkpoints, key=lambda path: path.stat().st_mtime, default=None)
    if latest is not None and compatible_model(latest, runtime_header):
        print(f"Resuming from {latest}")
        return latest

    if previous_dir is not None:
        for name in ("checkpoint-best-window.st", "after_training.st"):
            candidate = previous_dir / name
            if candidate.is_file() and compatible_model(candidate, runtime_header):
                print(f"Upgrading from {candidate}")
                return candidate

    print("No compatible checkpoint found, starting from random weights.")
    return None


def prepare_sources(sources: tuple[str, ...], strip_comments: bool, temporary_root: Path | None) -> list[str]:
    arguments: list[str] = []
    for index, source in enumerate(sources):
        path_text, weight = split_source(source)
        path = Path(path_text)
        if not path.is_dir():
            raise FileNotFoundError(f"Training directory '{path}' does not exist.")
        if strip_comments:
            assert temporary_root is not None
            stripped = temporary_root / f"source-{index}"
            print(f"Stripping comments from {path}...")
            run([
                sys.executable, str(ROOT / "training_postprocessor.py"),
                "--dir", str(path), "--output-dir", str(stripped), "--strip-comments",
            ])
            value = str(stripped) + (f":{weight}" if weight is not None else "")
        else:
            value = source
        arguments.extend(("--train-dir", value))
    return arguments


def configured_epochs(training_arguments: tuple[str, ...]) -> int:
    epochs: int | None = None
    for index, argument in enumerate(training_arguments[:-1]):
        if argument == "--epochs":
            try:
                epochs = int(training_arguments[index + 1])
            except ValueError as error:
                raise ValueError("The value following '--epochs' must be an integer.") from error
    if epochs is None or epochs < 1:
        raise ValueError("Incremental-window mode requires a positive '--epochs' training argument.")
    return epochs


def incremental_window_stages(
    target_layers: int,
    target_window: int,
    target_stride: int,
    total_epochs: int,
    stage_epochs: int = 4,
) -> list[tuple[int, int, int, int]]:
    if target_layers < 3:
        raise ValueError("Incremental-window mode requires a target of at least 3 layers.")
    if stage_epochs < 1:
        raise ValueError("incremental_stage_epochs must be positive.")
    stage_count = target_layers - 2
    growth_epoch_count = stage_epochs * (stage_count - 1)
    if total_epochs <= growth_epoch_count:
        raise ValueError(
            f"Incremental-window mode needs more than {growth_epoch_count} epochs "
            f"for {stage_count} stages of {stage_epochs} epochs."
        )

    # Shallow stages focus on short-range syntax before deeper stages introduce
    # progressively broader context. The final stage always honors the target
    # window from the configuration.
    curriculum_windows = {
        3: 12,
        4: 16,
        5: 24,
        6: 48,
        7: 72,
    }
    stages: list[tuple[int, int, int, int]] = []
    for layers in range(3, target_layers + 1):
        window = target_window if layers == target_layers else min(
            target_window,
            curriculum_windows.get(layers, target_window),
        )
        stride = max(1, round(target_stride * window / target_window))
        epochs = total_epochs - growth_epoch_count if layers == target_layers else stage_epochs
        stages.append((layers, window, stride, epochs))
    return stages


def incremental_training_steps(
    stages: list[tuple[int, int, int, int]],
    new_block_epochs: int = 2,
) -> list[tuple[int, int, int, int, str]]:
    if new_block_epochs < 1:
        raise ValueError("incremental_new_block_epochs must be positive.")
    first_layers, first_window, first_stride, first_epochs = stages[0]
    steps = [(first_layers, first_window, first_stride, first_epochs, "bootstrap")]
    for layers, window, stride, epochs in stages[1:]:
        if epochs <= new_block_epochs:
            raise ValueError(
                f"Layer {layers} needs more than {new_block_epochs} epochs so both "
                "new-block and all-blocks steps can run.")
        steps.append((layers, window, stride, new_block_epochs, "growth"))
        steps.append((layers, window, stride, epochs - new_block_epochs, "all-blocks"))
    return steps


def incremental_resume_step(
    checkpoint: Path,
    steps: list[tuple[int, int, int, int, str]],
) -> int:
    sidecar = Path(f"{checkpoint}.training.json")
    try:
        with sidecar.open(encoding="utf-8") as file:
            parameters = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot determine the incremental stage for '{checkpoint}' from "
            f"'{sidecar}': {error}"
        ) from error
    if not isinstance(parameters, dict):
        raise ValueError(f"Checkpoint sidecar '{sidecar}' must contain a JSON object.")

    shape = (
        parameters.get("layers"),
        parameters.get("window_size"),
        parameters.get("window_stride"),
    )
    phase = parameters.get("incremental_step")
    if phase not in ("bootstrap", "growth", "all-blocks"):
        if parameters.get("all_blocks_read_write") is True:
            phase = "all-blocks"
        elif parameters.get("upgrade_mode") is True:
            phase = "growth"
        else:
            phase = "bootstrap"
    for index, (layers, window, stride, _epochs, step_phase) in enumerate(steps):
        if shape == (layers, window, stride) and phase == step_phase:
            return index
    raise ValueError(
        f"Latest checkpoint shape {shape[0]} layers, window {shape[1]}, stride {shape[2]} "
        f"at step '{phase}' does not match the configured incremental plan."
    )


def has_incremental_stage_marker(progress_file: Path, stage: int) -> bool:
    if not progress_file.is_file():
        return False
    with progress_file.open(encoding="utf-8") as file:
        entries = json.load(file)
    if not isinstance(entries, list):
        raise ValueError(f"{progress_file} must contain a JSON array.")
    markers = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get("item_type") == "incremental_stage"
    ]
    return bool(markers) and markers[-1].get("stage") == stage


def append_incremental_stage_marker(
    progress_file: Path,
    stage: int,
    stage_count: int,
    epoch_offset: int,
    layers: int,
    window_size: int,
    window_stride: int,
    epochs: int,
    phase: str = "growth",
) -> None:
    if progress_file.is_file():
        with progress_file.open(encoding="utf-8") as file:
            entries = json.load(file)
        if not isinstance(entries, list):
            raise ValueError(f"{progress_file} must contain a JSON array.")
    else:
        entries = []
    entries.append({
        "item_type": "incremental_stage",
        "timestamp_ms": int(time.time() * 1000),
        "stage": stage,
        "stage_count": stage_count,
        "epoch_offset": epoch_offset,
        "layers": layers,
        "window_size": window_size,
        "window_stride": window_stride,
        "epochs": epochs,
        "phase": phase,
    })
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2)
        file.write("\n")


def write_incremental_controller_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def load_incremental_controller_state(
    path: Path,
    steps: list[tuple[int, int, int, int, str]],
    config: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read incremental controller state '{path}': {error}") from error
    if not isinstance(state, dict) or state.get("version") != 1:
        raise ValueError(f"Incremental controller state '{path}' has an unsupported format.")
    expected_plan = [list(step) for step in steps]
    if state.get("plan") != expected_plan:
        raise ValueError(
            f"Incremental controller state '{path}' does not match the current configuration. "
            "Use --latest to explicitly recover by checkpoint shape.")
    if config is not None and state.get("config") != config:
        raise ValueError(
            f"Incremental controller state '{path}' was created with different training "
            "parameters. Use --latest to explicitly adopt the current configuration.")
    active_step = state.get("active_step")
    if not isinstance(active_step, int) or isinstance(active_step, bool) or not 0 <= active_step <= len(steps):
        raise ValueError(f"Incremental controller state '{path}' has an invalid active_step.")
    return state


def newest_timed_checkpoint(model_dir: Path) -> Path | None:
    candidates = list(model_dir.glob("checkpoint-[0-9]*.st"))
    cleaned = model_dir / "checkpoint-latest.st"
    if cleaned.is_file():
        candidates.append(cleaned)
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def log_incremental_controller_step(
    model_dir: Path,
    step: int,
    step_count: int,
    layers: int,
    window: int,
    stride: int,
    epochs: int,
    phase: str,
    command: list[str],
) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (model_dir / "train.log").open("a", encoding="utf-8") as file:
        file.write(
            f"[{timestamp}] Incremental controller step active: {phase} "
            f"({step}/{step_count}, layers {layers}, window {window}, stride {stride}, "
            f"epochs {epochs})\n"
        )
        file.write(f"[{timestamp}] Incremental controller command: {json.dumps(command)}\n")


def parse_launcher_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=Path)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--fresh-start",
        action="store_true",
        help="ignore existing checkpoints and start with random weights",
    )
    resume.add_argument(
        "--latest",
        action="store_true",
        help="resume from the newest numbered or cleaned latest checkpoint instead of the best checkpoint",
    )
    resume.add_argument(
        "--lowest-loss",
        action="store_true",
        help="resume from the numbered checkpoint paired with the lowest recorded head-0 validation loss",
    )
    resume.add_argument(
        "--clean",
        action="store_true",
        help="preserve the newest numbered checkpoint as checkpoint-latest and remove intermediates, then exit",
    )
    resume.add_argument(
        "--resume-model",
        metavar="PATH",
        help="resume from this model instead of selecting a checkpoint automatically",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="pass --skip-warmup to rllm so the lowering schedule starts at the full base rate",
    )
    parser.add_argument(
        "--restart-learning-rate-schedule",
        action="store_true",
        help="pass --restart-learning-rate-schedule to rllm when resuming training",
    )
    parser.add_argument(
        "--incremental-window",
        action="store_true",
        help="grow training from 3 layers and a small window to the configured target in staged runs",
    )
    options = parser.parse_args(arguments)
    if options.incremental_window and any((
        options.fresh_start, options.lowest_loss,
        options.clean, options.resume_model is not None,
    )):
        parser.error(
            "--incremental-window cannot be combined with this checkpoint selection or cleanup option")
    return options


def main(arguments: list[str] | None = None) -> int:
    os.chdir(ROOT)
    options = parse_launcher_arguments(
        arguments if arguments is not None else sys.argv[1:]
    )
    try:
        config = load_config(options.config_file)
        num_layers = positive_integer(config, "layers")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    incremental_mode = options.incremental_window or bool(config.get("incremental_window", False))
    if incremental_mode and any((
        options.fresh_start, options.lowest_loss,
        options.clean, options.resume_model is not None,
    )):
        print(
            "Incremental-window mode cannot be combined with checkpoint selection or cleanup options.",
            file=sys.stderr,
        )
        return 2

    window_size = positive_integer(config, "window_size")
    window_stride = positive_integer(config, "window_stride")
    learn_depth = positive_integer(config, "learn_depth")
    model_dir = Path(str(config["model_directory"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    if options.clean or options.latest:
        try:
            clean_intermediate_checkpoints(model_dir)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 1
    if options.clean:
        return 0
    environment = os.environ.copy()
    environment["RLLM_MODEL_DIR"] = str(model_dir)

    print(
        f"Training shape: {num_layers} layers, learn depth {learn_depth}, "
        f"window size {window_size}, stride {window_stride}"
    )
    print(f"Model artifacts: {model_dir}/")

    build_type = str(config["build_type"])
    build_dir = ROOT / f"build_{build_type}"
    print(f"Configuring and building {build_type} before training...")
    run([str(ROOT / f"build_{build_type}.sh"), *string_list(config, "cmake_arguments")])

    for corpus_root in ("training_data0", "curriculum", "training_data2"):
        print(f"Normalizing {corpus_root} with training_postprocessor.py...")
        run([sys.executable, str(ROOT / "training_postprocessor.py"), "--dir", corpus_root])

    sources = string_list(config, "sources")
    training_arguments = string_list(config, "training_arguments")
    strip_comments = bool(config["strip_comments"])
    previous_dir = previous_model_directory(num_layers)
    runtime_header = build_dir / "generated/tokenizer_map.hpp"

    with tempfile.TemporaryDirectory(prefix="rllm-training-no-comments-") if strip_comments else _NullTemporaryDirectory() as temporary:
        temporary_root = Path(temporary) if temporary is not None else None
        if temporary_root is not None:
            print(f"Comment-free corpus copies: {temporary_root}")
        train_dirs = prepare_sources(sources, strip_comments, temporary_root)
        if incremental_mode:
            try:
                total_epochs = configured_epochs(training_arguments)
                stage_epochs = int(config.get("incremental_stage_epochs", 4))
                stages = incremental_window_stages(
                    num_layers, window_size, window_stride, total_epochs, stage_epochs)
                new_block_epochs = int(config.get("incremental_new_block_epochs", 2))
                steps = incremental_training_steps(stages, new_block_epochs)
                upgrade_output_scale = float(
                    config.get("incremental_upgrade_output_scale", 0.1))
                if not math.isfinite(upgrade_output_scale) or not 0.0 < upgrade_output_scale <= 1.0:
                    raise ValueError("incremental_upgrade_output_scale must be in (0, 1].")
            except (TypeError, ValueError) as error:
                print(error, file=sys.stderr)
                return 2

            previous_output: Path | None = None
            first_step_index = 0
            resume_active_step = False
            controller_path = model_dir / "incremental-controller.json"
            if not options.latest:
                try:
                    controller = load_incremental_controller_state(controller_path, steps, config)
                except ValueError as error:
                    print(error, file=sys.stderr)
                    return 1
                if controller is not None:
                    first_step_index = int(controller["active_step"])
                    if first_step_index == len(steps):
                        print("Incremental controller reports that all steps are complete.")
                        return 0
                    if controller.get("status") == "running":
                        started_ms = controller.get("started_ms", 0)
                        timed = newest_timed_checkpoint(model_dir)
                        if (timed is not None and isinstance(started_ms, int)
                                and timed.stat().st_mtime_ns // 1_000_000 >= started_ms):
                            previous_output = timed
                            resume_active_step = True
                        else:
                            input_name = controller.get("input_checkpoint")
                            previous_output = Path(input_name) if isinstance(input_name, str) else None
                        print(
                            f"Recovering interrupted incremental step {first_step_index + 1}/{len(steps)} "
                            f"from {previous_output or 'its initial weights'}"
                        )
                    else:
                        completed_name = controller.get("completed_output")
                        previous_output = Path(completed_name) if isinstance(completed_name, str) else None
            if options.latest:
                try:
                    previous_output = select_resume_model(
                        model_dir, None, runtime_header,
                        fresh_start=False, latest=True, resume_model=None)
                    assert previous_output is not None
                    first_step_index = incremental_resume_step(previous_output, steps)
                    resume_active_step = True
                except (FileNotFoundError, RuntimeError, ValueError) as error:
                    print(error, file=sys.stderr)
                    return 1
                print(
                    f"Continuing incremental step {first_step_index + 1}/{len(steps)} "
                    f"from {previous_output}"
                )

            total_stage_count = len(steps)
            epoch_offset = sum(step[3] for step in steps[:first_step_index])
            for zero_based_step in range(first_step_index, len(steps)):
                stage_index = zero_based_step + 1
                stage_layers, stage_window, stage_stride, epochs, phase = steps[zero_based_step]
                if phase == "growth":
                    output_model = model_dir / f"incremental-{stage_layers}l-{stage_window}w-new-block.st"
                elif stage_layers == num_layers and phase == "all-blocks":
                    output_model = model_dir / "after_training.st"
                else:
                    output_model = model_dir / f"incremental-{stage_layers}l-{stage_window}w.st"
                resume_arguments: list[str] = []
                if previous_output is not None:
                    resume_arguments = ["-i", str(previous_output)]
                    resuming_current_step = resume_active_step and zero_based_step == first_step_index
                    if phase == "growth" and not resuming_current_step:
                        resume_arguments.extend((
                            "--upgrade-mode", "--freeze-old-blocks",
                            "--upgrade-residual-output-scale", str(upgrade_output_scale),
                        ))
                    elif phase == "all-blocks":
                        resume_arguments.append("--all-blocks-read-write")
                        if not resuming_current_step:
                            resume_arguments.append("--reset-training-cursor")
                schedule_arguments: list[str] = []
                if options.restart_learning_rate_schedule:
                    schedule_arguments.append("--restart-learning-rate-schedule")
                if options.skip_warmup:
                    schedule_arguments.append("--skip-warmup")
                command = [
                    str(build_dir / "rllm"), "--train", *resume_arguments,
                    "-o", str(output_model), *train_dirs,
                    *training_arguments,
                    "--method", f"window:{stage_window}",
                    "--window-stride", str(stage_stride),
                    "--layers", str(stage_layers), "--learn-depth", str(learn_depth),
                    "--epochs", str(epochs), *schedule_arguments,
                ]
                try:
                    progress_file = model_dir / "train.json"
                    if not has_incremental_stage_marker(progress_file, stage_index):
                        append_incremental_stage_marker(
                            progress_file, stage_index, total_stage_count,
                            epoch_offset, stage_layers, stage_window, stage_stride, epochs,
                            phase,
                        )
                except (OSError, json.JSONDecodeError, ValueError) as error:
                    print(f"Cannot publish incremental stage to visualizer: {error}", file=sys.stderr)
                    return 1
                print(
                    f"--- Incremental stage {stage_index}/{total_stage_count}: "
                    f"{stage_layers} layers, window {stage_window}, stride {stage_stride}, "
                    f"step {phase}, {epochs} epochs ---"
                )
                log_incremental_controller_step(
                    model_dir, stage_index, total_stage_count,
                    stage_layers, stage_window, stage_stride, epochs, phase, command)
                write_incremental_controller_state(controller_path, {
                    "version": 1,
                    "status": "running",
                    "active_step": zero_based_step,
                    "plan": [list(step) for step in steps],
                    "config": config,
                    "input_checkpoint": str(previous_output) if previous_output is not None else None,
                    "output_checkpoint": str(output_model),
                    "layers": stage_layers,
                    "window_size": stage_window,
                    "window_stride": stage_stride,
                    "epochs": epochs,
                    "phase": phase,
                    "command": command,
                    "started_ms": int(time.time() * 1000),
                })
                try:
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
                except subprocess.CalledProcessError as error:
                    print(
                        f"Incremental step failed with exit code {error.returncode}; controller state "
                        f"was preserved in {controller_path}. Rerun the same command to recover.",
                        file=sys.stderr,
                    )
                    return error.returncode or 1
                previous_output = output_model
                epoch_offset += epochs
                resume_active_step = False
                write_incremental_controller_state(controller_path, {
                    "version": 1,
                    "status": "ready",
                    "active_step": zero_based_step + 1,
                    "plan": [list(step) for step in steps],
                    "config": config,
                    "completed_output": str(output_model),
                    "completed_ms": int(time.time() * 1000),
                })
            return 0

        try:
            resume = select_resume_model(
                model_dir, previous_dir, runtime_header,
                fresh_start=options.fresh_start, latest=options.latest,
                resume_model=options.resume_model, lowest_loss=options.lowest_loss,
            )
        except (FileNotFoundError, RuntimeError) as error:
            print(error, file=sys.stderr)
            return 1

        resume_arguments: list[str] = []
        if resume is not None:
            resume_arguments = ["-i", str(resume), "--upgrade-mode", "--freeze-old-blocks"]
            print("Layer upgrade defaults: increase checkpoint depth with the original blocks read-only.")
        schedule_arguments: list[str] = []
        if options.restart_learning_rate_schedule:
            schedule_arguments.append("--restart-learning-rate-schedule")
        if options.skip_warmup:
            schedule_arguments.append("--skip-warmup")

        command = [
            str(build_dir / "rllm"), "--train", *resume_arguments,
            "-o", str(model_dir / "after_training.st"), *train_dirs,
            "--method", f"window:{window_size}", "--window-stride", str(window_stride),
            "--layers", str(num_layers), "--learn-depth", str(learn_depth),
            *schedule_arguments,
            *training_arguments,
        ]
        print(f"--- Starting {build_type} training ---")
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


class _NullTemporaryDirectory:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *unused: object) -> None:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
