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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("train.json"))
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
    return parser.parse_args()


def number(entry: dict[str, Any], key: str) -> float | None:
    value = entry.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def position(entry: dict[str, Any], validation: bool) -> float:
    epoch = float(entry.get("epoch", 0))
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


def plot_or_note(axis: Any, xs: list[float], ys: list[float], label: str, **kwargs: Any) -> None:
    if ys:
        axis.plot(xs, ys, label=label, **kwargs)
    else:
        axis.text(
            0.5, 0.5, "No validation records in this file\n(new runs record this metric)",
            ha="center", va="center", transform=axis.transAxes,
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


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list):
        raise SystemExit(f"{args.input} must contain a JSON array")
    entries = [entry for entry in raw if isinstance(entry, dict)]
    training = [entry for entry in entries if entry.get("item_type") != "validation"]
    validation = [entry for entry in entries if entry.get("item_type") == "validation"]

    fig, axes = plt.subplot_mosaic(
        [
            ["loss", "perplexity"],
            ["mtp", "probability"],
            ["gradient", "gradient"],
        ],
        figsize=(14, 12),
        constrained_layout=True,
    )
    train_x, train_loss = series(training, "training_loss", False)
    val_x, val_loss = series(validation, "validation_loss", True)
    plot_or_note(axes["loss"], train_x, train_loss, "training loss", alpha=0.45, linewidth=1)
    if val_loss:
        axes["loss"].plot(val_x, val_loss, "o-", label="validation loss", linewidth=2)
    axes["loss"].set_title("Loss")
    axes["loss"].legend()

    specifications = (
        (axes["perplexity"], "perplexity", "Validation perplexity"),
        (axes["mtp"], "all_mtp_loss", "Validation all-MTP loss"),
        (axes["probability"], "correct_token_probability_percent", "Correct-token probability (%)"),
    )
    for axis, key, title in specifications:
        xs, ys = series(validation, key, True)
        plot_or_note(axis, xs, ys, title, marker="o", linewidth=2)
        axis.set_title(title)

    gradient_axis = axes["gradient"]
    gradient_series = optimizer_gradient_series(training)
    if gradient_series:
        for group, (xs, ys) in gradient_series.items():
            gradient_axis.plot(xs, ys, "o-", label=group, linewidth=1.5, markersize=4)
        gradient_axis.legend(fontsize="small", ncols=min(4, len(gradient_series)))
    else:
        gradient_axis.text(
            0.5, 0.5,
            "No optimizer diagnostics in this file\n(new diagnostic-enabled runs record this metric)",
            ha="center", va="center", transform=gradient_axis.transAxes,
        )
    gradient_axis.set_title("Normalized global gradient norm by parameter group")
    gradient_axis.set_ylabel("L2 norm after global clipping")

    for axis in axes.values():
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"Training metrics — {args.input}", fontsize=14)
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
