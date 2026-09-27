#!/usr/bin/env python3
"""Export rLLM identifier engrams for TensorFlow Embedding Projector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Iterator


TENSOR_NAME = "input_layer.identifier_embeddings"
DTYPE_FORMATS = {
    "F16": ("e", 2),
    "F32": ("f", 4),
    "F64": ("d", 8),
}
IDENTIFIER_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789_"


def engram_hash(engram: str, bucket_count: int) -> int:
    """Apply the same 32-bit FNV-1a hash used by InputLayer.cc."""
    value = 2166136261
    for byte in engram.encode("ascii"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value % bucket_count


def representative_engrams(bucket_count: int, limit: int = 6) -> list[list[str]]:
    """Find readable identifier trigrams that map to every embedding bucket."""
    examples: list[list[str]] = [[] for _ in range(bucket_count)]

    # These are precisely the trigram forms produced by ^ + lowercase(name) + $.
    candidates = (
        (f"^{first}$" for first in IDENTIFIER_CHARACTERS),
        (f"^{first}{second}" for first in IDENTIFIER_CHARACTERS
            for second in IDENTIFIER_CHARACTERS),
        (f"{first}{second}$" for first in IDENTIFIER_CHARACTERS
            for second in IDENTIFIER_CHARACTERS),
        (f"{first}{second}{third}" for first in IDENTIFIER_CHARACTERS
            for second in IDENTIFIER_CHARACTERS for third in IDENTIFIER_CHARACTERS),
    )
    remaining = bucket_count
    for candidate_group in candidates:
        for candidate in candidate_group:
            bucket = engram_hash(candidate, bucket_count)
            if len(examples[bucket]) >= limit:
                continue
            if not examples[bucket]:
                remaining -= 1
            examples[bucket].append(candidate)
        if remaining == 0 and all(len(bucket) >= limit for bucket in examples):
            break
    return examples


def tensor_descriptor(
    checkpoint: Path,
    tensor_name: str = TENSOR_NAME,
) -> tuple[tuple[int, int], int, str, int]:
    """Return matrix shape, absolute data offset, scalar format, and scalar size."""
    with checkpoint.open("rb") as file:
        encoded_header_size = file.read(8)
        if len(encoded_header_size) != 8:
            raise ValueError(f"{checkpoint} is too short to be a safetensors checkpoint")
        header_size = struct.unpack("<Q", encoded_header_size)[0]
        try:
            header = json.loads(file.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{checkpoint} has an invalid safetensors header") from exc

    if tensor_name not in header:
        available = sorted(name for name in header if name != "__metadata__")
        sample = ", ".join(available[:5]) or "none"
        raise KeyError(f"tensor {tensor_name!r} is missing; first available tensors: {sample}")

    descriptor = header[tensor_name]
    dtype_name = descriptor.get("dtype")
    if dtype_name not in DTYPE_FORMATS:
        raise ValueError(f"unsupported dtype {dtype_name!r} for {tensor_name}")
    shape = tuple(int(value) for value in descriptor.get("shape", []))
    offsets = descriptor.get("data_offsets", [])
    if len(shape) != 2 or len(offsets) != 2:
        raise ValueError(f"{tensor_name} must be a rank-2 tensor")

    begin, end = (int(value) for value in offsets)
    scalar_format, scalar_size = DTYPE_FORMATS[dtype_name]
    expected_bytes = shape[0] * shape[1] * scalar_size
    if begin < 0 or end - begin != expected_bytes:
        raise ValueError(f"invalid byte range for {tensor_name}")
    data_offset = 8 + header_size + begin
    if data_offset + expected_bytes > checkpoint.stat().st_size:
        raise ValueError(f"{tensor_name} extends past the end of {checkpoint}")
    return (shape[0], shape[1]), data_offset, scalar_format, scalar_size


def tensor_rows(
    checkpoint: Path,
    tensor_name: str = TENSOR_NAME,
) -> Iterator[tuple[float, ...]]:
    """Yield rows from a rank-2 numeric safetensors tensor."""
    (row_count, column_count), offset, scalar_format, scalar_size = tensor_descriptor(
        checkpoint, tensor_name
    )
    row_size = column_count * scalar_size
    row_struct = struct.Struct(f"<{column_count}{scalar_format}")
    with checkpoint.open("rb") as file:
        file.seek(offset)
        for _ in range(row_count):
            encoded = file.read(row_size)
            if len(encoded) != row_size:
                raise ValueError(f"unexpected end of {checkpoint} while reading {tensor_name}")
            yield row_struct.unpack(encoded)


def export_projector_files(
    checkpoint: Path,
    vectors_path: Path,
    metadata_path: Path,
    tensor_name: str = TENSOR_NAME,
) -> tuple[int, int]:
    """Write vectors and bucket metadata accepted by projector.tensorflow.org."""
    shape, _, _, _ = tensor_descriptor(checkpoint, tensor_name)
    examples = representative_engrams(shape[0])
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with vectors_path.open("w", encoding="utf-8", newline="\n") as vectors_file, \
            metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
        metadata_file.write("engram_examples\tbucket\tl2_norm\n")
        for bucket, row in enumerate(tensor_rows(checkpoint, tensor_name)):
            vectors_file.write("\t".join(format(value, ".9g") for value in row))
            vectors_file.write("\n")
            norm = math.sqrt(math.fsum(value * value for value in row))
            label = ", ".join(examples[bucket])
            metadata_file.write(f"{label}\t{bucket}\t{norm:.9g}\n")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="rLLM .st or .safetensors checkpoint")
    parser.add_argument("-o", "--output", type=Path, help="output vectors TSV path")
    parser.add_argument("--metadata", type=Path, help="output metadata TSV path")
    parser.add_argument(
        "--tensor",
        default=TENSOR_NAME,
        help=f"tensor to export (default: {TENSOR_NAME})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vectors_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}-identifier-engrams.tsv"
    )
    metadata_path = args.metadata or vectors_path.with_name(
        f"{vectors_path.stem}-metadata.tsv"
    )
    try:
        rows, columns = export_projector_files(
            args.checkpoint, vectors_path, metadata_path, args.tensor
        )
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Exported {rows} embeddings with {columns} dimensions")
    print(f"Vectors:  {vectors_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
