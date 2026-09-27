#!/usr/bin/env python3
"""Report NaN values in an rLLM Safetensors checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

try:
    import numpy as np
except ImportError:  # The dependency-free path remains useful for small files.
    np = None


HEADER_LENGTH_BYTES = 8
MAX_HEADER_BYTES = 100 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class FloatFormat:
    byte_width: int
    exponent_mask: int
    mantissa_mask: int


FLOAT_FORMATS = {
    "F16": FloatFormat(2, 0x7C00, 0x03FF),
    "BF16": FloatFormat(2, 0x7F80, 0x007F),
    "F32": FloatFormat(4, 0x7F800000, 0x007FFFFF),
    "F64": FloatFormat(8, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
}


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class NaNResult:
    tensor: Tensor
    count: int
    first_index: int | None


def _parse_tensor(name: str, descriptor: object, data_size: int) -> Tensor:
    if not isinstance(descriptor, dict):
        raise ValueError(f"tensor {name!r} has a non-object descriptor")
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if not isinstance(dtype, str):
        raise ValueError(f"tensor {name!r} has an invalid dtype")
    if (not isinstance(shape, list) or
            any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
                for dimension in shape)):
        raise ValueError(f"tensor {name!r} has an invalid shape")
    if (not isinstance(offsets, list) or len(offsets) != 2 or
            any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)):
        raise ValueError(f"tensor {name!r} has invalid data offsets")
    start, end = offsets
    if start < 0 or end < start or end > data_size:
        raise ValueError(
            f"tensor {name!r} has out-of-range data offsets [{start}, {end})")

    tensor = Tensor(name, dtype, tuple(shape), start, end)
    float_format = FLOAT_FORMATS.get(dtype)
    if float_format is not None:
        expected_size = math.prod(tensor.shape) * float_format.byte_width
        if end - start != expected_size:
            raise ValueError(
                f"tensor {name!r} has {end - start} data bytes; "
                f"shape {shape} with dtype {dtype} requires {expected_size}")
    return tensor


def read_header(file: BinaryIO, file_size: int) -> tuple[int, list[Tensor]]:
    encoded_length = file.read(HEADER_LENGTH_BYTES)
    if len(encoded_length) != HEADER_LENGTH_BYTES:
        raise ValueError("file is too short to contain a Safetensors header")
    header_length = struct.unpack("<Q", encoded_length)[0]
    if header_length > MAX_HEADER_BYTES:
        raise ValueError(f"Safetensors header is implausibly large ({header_length} bytes)")
    data_start = HEADER_LENGTH_BYTES + header_length
    if data_start > file_size:
        raise ValueError("Safetensors header extends past the end of the file")
    try:
        header = json.loads(file.read(header_length))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Safetensors JSON header: {error}") from error
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")

    data_size = file_size - data_start
    tensors = [
        _parse_tensor(name, descriptor, data_size)
        for name, descriptor in header.items()
        if name != "__metadata__"
    ]
    return data_start, tensors


def _nan_indices(data: bytes, float_format: FloatFormat, base_index: int) -> Iterator[int]:
    for offset in range(0, len(data), float_format.byte_width):
        bits = int.from_bytes(data[offset:offset + float_format.byte_width], "little")
        if ((bits & float_format.exponent_mask) == float_format.exponent_mask and
                bits & float_format.mantissa_mask):
            yield base_index + offset // float_format.byte_width


def _nan_summary(data: bytes, float_format: FloatFormat, base_index: int) -> tuple[int, int | None]:
    if np is None:
        indices = _nan_indices(data, float_format, base_index)
        first = next(indices, None)
        return (0, None) if first is None else (1 + sum(1 for _ in indices), first)

    integer_dtype = np.dtype(f"<u{float_format.byte_width}")
    bits = np.frombuffer(data, dtype=integer_dtype)
    matches = ((bits & float_format.exponent_mask) == float_format.exponent_mask) & (
        (bits & float_format.mantissa_mask) != 0)
    count = int(np.count_nonzero(matches))
    if not count:
        return 0, None
    return count, base_index + int(np.flatnonzero(matches)[0])


def check_checkpoint(path: Path) -> tuple[list[NaNResult], int, int]:
    """Return per-tensor results, number of checked tensors, and checked values."""
    file_size = path.stat().st_size
    with path.open("rb") as file:
        data_start, tensors = read_header(file, file_size)
        results: list[NaNResult] = []
        checked_tensors = 0
        checked_values = 0
        for tensor in tensors:
            float_format = FLOAT_FORMATS.get(tensor.dtype)
            if float_format is None:
                continue
            checked_tensors += 1
            value_count = math.prod(tensor.shape)
            checked_values += value_count
            count = 0
            first_index = None
            file.seek(data_start + tensor.start)
            values_read = 0
            remaining = tensor.end - tensor.start
            chunk_size = max(
                float_format.byte_width,
                CHUNK_BYTES - CHUNK_BYTES % float_format.byte_width)
            while remaining:
                data = file.read(min(remaining, chunk_size))
                if not data:
                    raise ValueError(f"unexpected end of file while reading tensor {tensor.name!r}")
                if len(data) % float_format.byte_width:
                    raise ValueError(f"unaligned data in tensor {tensor.name!r}")
                chunk_count, chunk_first = _nan_summary(data, float_format, values_read)
                if first_index is None and chunk_first is not None:
                    first_index = chunk_first
                count += chunk_count
                values_read += len(data) // float_format.byte_width
                remaining -= len(data)
            if count:
                results.append(NaNResult(tensor, count, first_index))
    return results, checked_tensors, checked_values


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check every floating-point tensor in an rLLM Safetensors checkpoint for NaNs.")
    parser.add_argument("checkpoint", type=Path, help="checkpoint file (.st or .safetensors)")
    args = parser.parse_args(arguments)

    try:
        results, checked_tensors, checked_values = check_checkpoint(args.checkpoint)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if results:
        total = sum(result.count for result in results)
        print(f"FOUND {total} NaN value(s) in {len(results)} tensor(s):")
        for result in results:
            print(
                f"  {result.tensor.name}: {result.count} NaN(s), "
                f"first flat index {result.first_index}, shape {list(result.tensor.shape)}")
        return 1

    print(
        f"OK: no NaNs in {checked_values} floating-point values "
        f"across {checked_tensors} tensor(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
