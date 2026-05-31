#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from offloadize_common import transform_tree, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Vulkanized source copies for OFFLOAD_* loop macros.")
    parser.add_argument("--src-dir", required=True, help="Input source directory (e.g. src)")
    parser.add_argument("--out-dir", required=True, help="Output directory in build tree")
    parser.add_argument("--manifest", required=False, help="Optional output .cmake list of generated files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    generated_files = transform_tree(
        src_dir,
        out_dir,
        backend_namespace="vulkan",
        include_line="#include <vulkan_kernel_calls.hpp>",
    )

    if args.manifest:
        manifest = Path(args.manifest).resolve()
        write_manifest(manifest, "RLLM_VULKANIZED_SOURCES", generated_files)

    print(f"vulkanize.py: generated {len(generated_files)} source files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())