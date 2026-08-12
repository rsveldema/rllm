#!/usr/bin/env python3
"""Build and launch an rLLM training run.

This replaces the debug/release shell wrappers and their shared sourced script.
The selected JSON file is the sole source of build and training options.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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


def delete_superseded_checkpoints(model_dir: Path) -> None:
    for checkpoint in model_dir.glob("checkpoint-[0-9]*.st"):
        print(f"Deleting superseded checkpoint {checkpoint}")
        checkpoint.unlink()


def select_resume_model(
    model_dir: Path,
    previous_dir: Path | None,
    runtime_header: Path,
    *,
    fresh_start: bool,
    resume_model: str | None,
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

    best = model_dir / "checkpoint-best-window.st"
    if best.is_file() and compatible_model(best, runtime_header):
        print(f"Resuming from {best}")
        delete_superseded_checkpoints(model_dir)
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
        "--resume-model",
        metavar="PATH",
        help="resume from this model instead of selecting a checkpoint automatically",
    )
    return parser.parse_args(arguments)


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

    window_size = positive_integer(config, "window_size")
    window_stride = positive_integer(config, "window_stride")
    learn_depth = positive_integer(config, "learn_depth")
    model_dir = Path(str(config["model_directory"]))
    model_dir.mkdir(parents=True, exist_ok=True)
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
    strip_comments = bool(config["strip_comments"])
    previous_dir = previous_model_directory(num_layers)
    runtime_header = build_dir / "generated/tokenizer_map.hpp"

    with tempfile.TemporaryDirectory(prefix="rllm-training-no-comments-") if strip_comments else _NullTemporaryDirectory() as temporary:
        temporary_root = Path(temporary) if temporary is not None else None
        if temporary_root is not None:
            print(f"Comment-free corpus copies: {temporary_root}")
        train_dirs = prepare_sources(sources, strip_comments, temporary_root)
        try:
            resume = select_resume_model(
                model_dir,
                previous_dir,
                runtime_header,
                fresh_start=options.fresh_start,
                resume_model=options.resume_model,
            )
        except (FileNotFoundError, RuntimeError) as error:
            print(error, file=sys.stderr)
            return 1

        resume_arguments: list[str] = []
        if resume is not None:
            resume_arguments = ["-i", str(resume), "--upgrade-mode", "--freeze-old-blocks"]
            print("Layer upgrade defaults: increase checkpoint depth with the original blocks read-only.")

        command = [
            str(build_dir / "rllm"), "--train", *resume_arguments,
            "-o", str(model_dir / "after_training.st"), *train_dirs,
            "--method", f"window:{window_size}", "--window-stride", str(window_stride),
            "--layers", str(num_layers), "--learn-depth", str(learn_depth),
            *string_list(config, "training_arguments"),
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
