#!/usr/bin/env python3
"""Visualize training and validation metrics stored in train.json."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button


def discover_default_inputs() -> list[Path]:
    return sorted(Path.cwd().glob("models-*/train.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="training metrics JSON files (default: all models-*/train.json files)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("training_metrics.png"))
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="open an interactive plot window (the default)",
    )
    display.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="save the plot without opening an interactive window",
    )
    parser.set_defaults(show=True)
    args = parser.parse_args()
    args.discover_inputs = not args.inputs
    if args.discover_inputs:
        args.inputs = discover_default_inputs()
    return args


def number(entry: dict[str, Any], key: str) -> float | None:
    value = entry.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def position(entry: dict[str, Any], validation: bool) -> float:
    epoch = float(entry.get("epoch", 0)) + float(entry.get("_epoch_offset", 0))
    if validation:
        progress = entry.get("epoch_progress", 1.0)
        return epoch + float(progress if isinstance(progress, (int, float)) else 1.0)
    total, end = entry.get("total_items"), entry.get("range_end")
    if isinstance(total, (int, float)) and total > 0 and isinstance(end, (int, float)):
        return epoch + float(end) / float(total)
    return epoch


def series(entries: list[dict[str, Any]], key: str, validation: bool) -> tuple[list[float], list[float]]:
    points = [(position(entry, validation), number(entry, key)) for entry in entries]
    return (
        [x for x, y in points if y is not None],
        [y for _, y in points if y is not None],
    )


def optimizer_gradient_series(
    entries: list[dict[str, Any]],
) -> dict[str, tuple[list[float], list[float]]]:
    grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for entry in entries:
        diagnostics = entry.get("optimizer_diagnostics")
        if not isinstance(diagnostics, list):
            continue
        x = position(entry, False)
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            group = diagnostic.get("parameter_group")
            value = number(diagnostic, "normalized_global_gradient_norm")
            if isinstance(group, str) and value is not None:
                grouped[group][0].append(x)
                grouped[group][1].append(value)
    return dict(grouped)


def load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array")
    entries: list[dict[str, Any]] = []
    stage_offset = 0.0
    resume_offset = 0.0
    latest_position = 0.0
    for raw_entry in raw:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        if entry.get("item_type") == "incremental_stage":
            marker_offset = number(entry, "epoch_offset")
            if marker_offset is not None:
                stage_offset = marker_offset
                resume_offset = 0.0
                latest_position = max(latest_position, stage_offset)

        # Older checkpoints logged every restart baseline at the beginning of
        # its local epoch. Rebase a backwards baseline, and the records that
        # follow it, onto the latest position already shown for this stage.
        if entry.get("item_type") == "validation" and entry.get("phase") == "baseline":
            epoch = number(entry, "epoch") or 0.0
            progress = number(entry, "epoch_progress") or 0.0
            raw_position = stage_offset + epoch + progress
            if raw_position + resume_offset < latest_position:
                resume_offset = latest_position - raw_position

        entry["_epoch_offset"] = stage_offset + resume_offset
        entries.append(entry)
        if entry.get("item_type") in ("window", "validation"):
            latest_position = max(
                latest_position,
                position(entry, entry.get("item_type") == "validation"),
            )
    return entries


def epoch_layer_label(epoch: float, stages: list[tuple[float, int]]) -> str:
    epoch_text = f"{epoch:g}"
    active_layer = next(
        (layers for offset, layers in reversed(stages) if epoch >= offset),
        None,
    )
    return epoch_text if active_layer is None else f"{epoch_text} ({active_layer}L)"


def render_metrics(
    fig: Any,
    axes: dict[str, Any],
    datasets: list[tuple[Path, list[dict[str, Any]]]],
) -> None:
    for axis in axes.values():
        axis.set_visible(True)
        axis.clear()

    has_loss = False
    has_gradient = False
    has_metric = {"perplexity": False, "mtp": False, "probability": False}
    specifications = (
        ("perplexity", "perplexity", "Validation perplexity"),
        ("mtp", "all_mtp_loss", "Validation all-MTP loss"),
        ("probability", "correct_token_probability_percent", "Correct-token probability (%)"),
    )
    incremental_stages: list[tuple[float, int]] = []

    for input_path, entries in datasets:
        label = input_path.parent.name
        training = [entry for entry in entries if entry.get("item_type") != "validation"]
        validation = [entry for entry in entries if entry.get("item_type") == "validation"]
        stages = [entry for entry in entries if entry.get("item_type") == "incremental_stage"]
        if stages and not incremental_stages:
            incremental_stages = sorted(
                (offset, int(layers))
                for stage in stages
                if (offset := number(stage, "epoch_offset")) is not None
                and (layers := number(stage, "layers")) is not None
            )

        for stage in stages:
            stage_x = number(stage, "epoch_offset")
            if stage_x is None:
                continue
            for axis in axes.values():
                axis.axvline(stage_x, color="0.55", linestyle="--", linewidth=0.8, alpha=0.45)
            layers = stage.get("layers", "?")
            window = stage.get("window_size", "?")
            axes["loss"].annotate(
                f"{layers}L/{window}W", (stage_x, 1),
                xycoords=("data", "axes fraction"), xytext=(3, -3),
                textcoords="offset points", rotation=90,
                va="top", ha="left", fontsize=7, color="0.35",
            )

        train_x, train_loss = series(training, "training_loss", False)
        val_x, val_loss = series(validation, "validation_loss", True)
        if train_loss:
            axes["loss"].plot(
                train_x, train_loss, label=f"{label} training", alpha=0.45, linewidth=1)
            has_loss = True
        if val_loss:
            axes["loss"].plot(
                val_x, val_loss, "o-", label=f"{label} validation", linewidth=2)
            has_loss = True

        for axis_name, key, _title in specifications:
            xs, ys = series(validation, key, True)
            if ys:
                axes[axis_name].plot(xs, ys, marker="o", linewidth=2, label=label)
                has_metric[axis_name] = True

        for group, (xs, ys) in optimizer_gradient_series(training).items():
            axes["gradient"].plot(
                xs, ys, "o-", label=f"{label}: {group}", linewidth=1.5, markersize=4)
            has_gradient = True

    if not has_loss:
        axes["loss"].text(
            0.5, 0.5, "No loss records in these files",
            ha="center", va="center", transform=axes["loss"].transAxes)
    axes["loss"].set_title("Loss")
    if has_loss:
        axes["loss"].legend()

    for axis_name, _key, title in specifications:
        axis = axes[axis_name]
        if has_metric[axis_name]:
            axis.legend()
        else:
            axis.text(
                0.5, 0.5, "No validation records in these files",
                ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
    axes["perplexity"].set_yscale("log")
    axes["perplexity"].set_ylabel("perplexity (log scale)")

    gradient_axis = axes["gradient"]
    if has_gradient:
        gradient_axis.legend(fontsize="small", ncols=min(4, len(datasets)))
        gradient_axis.set_title("Normalized global gradient norm by parameter group")
        gradient_axis.set_ylabel("L2 norm after global clipping")
    else:
        gradient_axis.set_visible(False)

    for axis in axes.values():
        if not axis.get_visible():
            continue
        axis.set_xlabel("epoch")
        if incremental_stages:
            axis.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _position: epoch_layer_label(value, incremental_stages)
            ))
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"Training metrics — {len(datasets)} run(s)", fontsize=14)


def main() -> None:
    args = parse_args()

    def current_input_paths() -> list[Path]:
        return discover_default_inputs() if args.discover_inputs else args.inputs

    def load_datasets() -> list[tuple[Path, list[dict[str, Any]]]]:
        paths = current_input_paths()
        if not paths:
            raise FileNotFoundError("no models-*/train.json files found")
        return [(path, load_entries(path)) for path in paths]

    try:
        datasets = load_datasets()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"Cannot load training metrics: {error}") from error

    fig, axes = plt.subplot_mosaic(
        [
            ["loss", "perplexity"],
            ["mtp", "probability"],
            ["gradient", "gradient"],
        ],
        figsize=(14, 12),
        constrained_layout=True,
    )
    render_metrics(fig, axes, datasets)

    status = fig.text(0.875, 0.945, "", ha="right", va="center", fontsize=9)
    button_axis = fig.add_axes((0.88, 0.955, 0.1, 0.03))
    reload_button = Button(button_axis, "Reload")

    def reload_metrics(_event: Any) -> None:
        try:
            latest_datasets = load_datasets()
            render_metrics(fig, axes, latest_datasets)
            fig.savefig(args.output, dpi=160)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            status.set_text(f"Reload failed: {error}")
            status.set_color("tab:red")
        else:
            record_count = sum(len(entries) for _path, entries in latest_datasets)
            status.set_text(f"Reloaded {record_count} records from {len(latest_datasets)} files")
            status.set_color("tab:green")
        fig.canvas.draw_idle()

    reload_button.on_clicked(reload_metrics)
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
