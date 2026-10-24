#!/usr/bin/env python3
"""Download a resumable sample of SmolLM Corpus python-edu source files."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
import io
import json
from pathlib import Path
import sys
import tokenize
from typing import Any
import unicodedata
import warnings


REPOSITORY = "HuggingFaceTB/smollm-corpus"
CONFIGURATION = "python-edu"
BUCKET = "softwareheritage"


def python_comments(content: str) -> list[str]:
    """Return `#` comments and Python docstrings from source text."""
    comments: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        comments.extend(token.string.lstrip("#").strip() for token in tokens
                        if token.type == tokenize.COMMENT)
    except (IndentationError, tokenize.TokenError):
        pass
    try:
        # Corpus files may contain invalid escape sequences that are accepted by
        # Python but emit SyntaxWarning on newer interpreters. They do not
        # affect extracting comments or docstrings, so keep those warnings from
        # leaking into the training launcher output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return comments
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring:
                comments.append(docstring)
    return comments


def has_english_comments(content: str) -> bool:
    """Accept files with Latin-script comment or docstring text only."""
    comments = python_comments(content)
    letters = [character for comment in comments for character in comment
               if character.isalpha()]
    return bool(letters) and all(
        "LATIN" in unicodedata.name(character, "") for character in letters
    )


def dependencies() -> tuple[Any, Any, Any]:
    try:
        import boto3
        import botocore
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "python-edu download requires 'datasets', 'boto3', and 'botocore'; "
            "install them from the repository requirements.txt"
        ) from error
    return boto3, botocore, load_dataset


def download_blob(s3: Any, output_dir: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    blob_id = str(row["blob_id"])
    destination = output_dir / "files" / f"{blob_id}.py"
    if destination.is_file():
        return None
    try:
        response = s3.get_object(Bucket=BUCKET, Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=response["Body"]) as compressed:
            content = compressed.read().decode("utf-8", errors="ignore")
    except Exception as error:
        return {"blob_id": blob_id, "error": str(error)}
    if not has_english_comments(content):
        return {"blob_id": blob_id, "rejected": "comments_not_english"}
    temporary = destination.with_suffix(".py.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)
    return {
        "blob_id": blob_id,
        "repo_name": row.get("repo_name"),
        "path": row.get("path"),
        "score": row.get("score"),
    }


def download(output_dir: Path, max_files: int, workers: int) -> int:
    boto3, botocore, load_dataset = dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "files").mkdir(exist_ok=True)
    rejected: set[str] = set()
    manifest = output_dir / "manifest.jsonl"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("rejected") == "comments_not_english":
                rejected.add(str(record.get("blob_id", "")))

    newly_rejected: list[str] = []
    for path in (output_dir / "files").glob("*.py"):
        if not has_english_comments(path.read_text(encoding="utf-8")):
            rejected.add(path.stem)
            newly_rejected.append(path.stem)
            path.unlink()
    existing = sum(1 for _ in (output_dir / "files").glob("*.py"))
    if max_files and existing >= max_files:
        print(f"python-edu cache already contains {existing} files")
        return 0

    s3 = boto3.client(
        "s3", region_name="us-west-2",
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    rows = load_dataset(REPOSITORY, CONFIGURATION, split="train", streaming=True)
    target = max_files or sys.maxsize
    completed = existing
    with manifest.open("a", encoding="utf-8") as manifest_file, ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        for blob_id in newly_rejected:
            manifest_file.write(json.dumps({
                "blob_id": blob_id,
                "rejected": "comments_not_english",
            }, sort_keys=True) + "\n")
        manifest_file.flush()
        batch: list[dict[str, Any]] = []
        for row in rows:
            if str(row["blob_id"]) in rejected:
                continue
            destination = output_dir / "files" / f"{row['blob_id']}.py"
            if destination.is_file():
                continue
            batch.append(row)
            batch_limit = min(workers, target - completed)
            if len(batch) < batch_limit:
                continue
            for result in executor.map(lambda item: download_blob(s3, output_dir, item), batch):
                if result is None:
                    continue
                manifest_file.write(json.dumps(result, sort_keys=True) + "\n")
                manifest_file.flush()
                if "error" not in result and "rejected" not in result:
                    completed += 1
                    if completed % 100 == 0:
                        print(f"Downloaded {completed}/{max_files or 'all'} python-edu files")
            batch.clear()
            if completed >= target:
                break
        if batch and completed < target:
            for result in executor.map(lambda item: download_blob(s3, output_dir, item), batch):
                if result is not None:
                    manifest_file.write(json.dumps(result, sort_keys=True) + "\n")
                    if "error" not in result and "rejected" not in result:
                        completed += 1
    print(f"python-edu cache contains {completed} files in {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Maximum cached files; 0 downloads the full dataset")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.max_files < 0 or args.workers < 1:
        parser.error("--max-files must be non-negative and --workers must be positive")
    try:
        return download(args.output_dir, args.max_files, args.workers)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
