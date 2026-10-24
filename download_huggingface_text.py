#!/usr/bin/env python3
"""Download a Hugging Face text dataset into resumable per-example files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata


def load_dataset_function():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face text download requires 'datasets'; install the "
            "repository requirements.txt"
        ) from error
    return load_dataset


def is_english_text(text: str) -> bool:
    """Reject prose containing letters from non-Latin writing systems."""
    for character in text:
        if character.isalpha() and "LATIN" not in unicodedata.name(character, ""):
            return False
    return True


def download(
    repository: str,
    output_dir: Path,
    max_files: int,
    configuration: str | None = None,
    split: str = "train",
    text_field: str = "text",
    english_only: bool = False,
) -> int:
    load_dataset = load_dataset_function()
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(files_dir.glob("*.txt"))
    if english_only:
        for path in existing_files:
            if not is_english_text(path.read_text(encoding="utf-8")):
                path.unlink()
        existing_files = list(files_dir.glob("*.txt"))
    existing = len(existing_files)
    if max_files and existing >= max_files:
        print(f"Hugging Face text cache already contains {existing} files")
        return 0

    rows = load_dataset(repository, configuration, split=split, streaming=True)
    target = max_files or sys.maxsize
    manifest_path = output_dir / "manifest.jsonl"
    completed = existing
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, row in enumerate(rows):
            if completed >= target:
                break
            destination = files_dir / f"{index:09d}.txt"
            if destination.is_file():
                continue
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            if english_only and not is_english_text(text):
                continue
            temporary = destination.with_suffix(".txt.tmp")
            temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
            temporary.replace(destination)
            manifest.write(json.dumps(
                {"index": index, "language": "en" if english_only else None},
                sort_keys=True,
            ) + "\n")
            completed += 1
            if completed % 1000 == 0:
                print(f"Downloaded {completed}/{max_files or 'all'} text examples")
    print(f"Hugging Face text cache contains {completed} files in {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--configuration")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--english-only", action="store_true",
                        help="Keep only text using Latin-script letters")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Maximum cached examples; 0 downloads the full split")
    args = parser.parse_args()
    if args.max_files < 0:
        parser.error("--max-files must be non-negative")
    try:
        return download(
            args.repository, args.output_dir, args.max_files,
            args.configuration, args.split, args.text_field, args.english_only,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
