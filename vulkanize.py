#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from offloadize_common import LoopContext, resolve_symbol_values, transform_tree, write_manifest


_SIMD_REDUCTION_PLUS_RE = re.compile(r"^\s*RLLM_OMP_SIMD_REDUCTION_PLUS\s*\([^)]*\)\s*;?\s*$")
_STATIC_CAST_SIZE_T_RE = re.compile(r"static_cast\s*<\s*size_t\s*>\s*\(([^()]*)\)")
_SIZE_T_WORD_RE = re.compile(r"\bsize_t\b")


def _sanitize_kernel_line_for_glsl(line: str) -> str:
    # GLSL has no size_t; map common generated C++ forms to int.
    line = _STATIC_CAST_SIZE_T_RE.sub(r"int(\1)", line)
    line = _SIZE_T_WORD_RE.sub("int", line)
    return line


@dataclass
class VulkanKernelSpec:
    rel_path: str
    lineno: int
    is_2d: bool
    vars: list[str]
    range_expr: str
    body_lines: list[str]


def _sanitize_name_component(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _kernel_file_path(kernel_root: Path, spec: VulkanKernelSpec) -> Path:
    rel = Path(spec.rel_path)
    stem = _sanitize_name_component(rel.stem)
    return kernel_root / rel.parent / f"{stem}.L{spec.lineno}.comp"


def _spirv_file_path(kernel_root: Path, spec: VulkanKernelSpec) -> Path:
    rel = Path(spec.rel_path)
    stem = _sanitize_name_component(rel.stem)
    return kernel_root / rel.parent / f"{stem}.L{spec.lineno}.spv"


def _render_kernel_stub(spec: VulkanKernelSpec) -> str:
    filtered_body_lines = [line for line in spec.body_lines if not _SIMD_REDUCTION_PLUS_RE.match(line)]
    filtered_body_lines = [_sanitize_kernel_line_for_glsl(line) for line in filtered_body_lines]
    body_text = "\n".join(
        f"    {line.lstrip()}" if line.strip() else ""
        for line in filtered_body_lines
    )
    return (
        "#version 450\n"
        "\n"
        "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
        "\n"
        "void main()\n"
        "{\n"
        f"{body_text}\n"
        "}\n"
    )


def _write_kernel_stubs(kernel_specs: list[VulkanKernelSpec], kernel_root: Path) -> list[Path]:
    generated_kernel_files: list[Path] = []
    for spec in kernel_specs:
        kernel_path = _kernel_file_path(kernel_root, spec)
        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text(_render_kernel_stub(spec), encoding="utf-8")
        generated_kernel_files.append(kernel_path)
    return generated_kernel_files


def _compile_to_spirv(compiler: str, input_path: Path, output_path: Path) -> None:
    if Path(compiler).name == "glslangValidator":
        cmd = [compiler, "-V", input_path.as_posix(), "-o", output_path.as_posix()]
    else:
        cmd = [compiler, input_path.as_posix(), "-o", output_path.as_posix()]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Shader compiler failed for {input_path}:\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _compile_kernel_stubs(
    compiler: str,
    kernel_specs: list[VulkanKernelSpec],
    kernel_root: Path,
) -> list[Path]:
    generated_spirv_files: list[Path] = []
    for spec in kernel_specs:
        src = _kernel_file_path(kernel_root, spec)
        spv = _spirv_file_path(kernel_root, spec)
        spv.parent.mkdir(parents=True, exist_ok=True)
        _compile_to_spirv(compiler, src, spv)
        generated_spirv_files.append(spv)
    return generated_spirv_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Vulkanized source copies for OFFLOAD_* loop macros.")
    parser.add_argument("--src-dir", required=True, help="Input source directory (e.g. src)")
    parser.add_argument("--out-dir", required=True, help="Output directory in build tree")
    parser.add_argument("--manifest", required=False, help="Optional output .cmake list of generated files")
    parser.add_argument(
        "--enum-value-tool",
        required=False,
        help="Optional executable used to print JSON search/replace pairs",
    )
    parser.add_argument(
        "--shader-compiler",
        required=False,
        help="Path to Vulkan GLSL compiler (glslc or glslangValidator)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    kernel_root = out_dir.parent / "kernels"
    kernel_specs: list[VulkanKernelSpec] = []
    symbol_values = resolve_symbol_values(args.enum_value_tool)

    def collect_kernel(ctx: LoopContext) -> None:
        kernel_specs.append(
            VulkanKernelSpec(
                rel_path=ctx.rel_path,
                lineno=ctx.lineno,
                is_2d=ctx.is_2d,
                vars=list(ctx.vars),
                range_expr=ctx.range_expr,
                body_lines=list(ctx.body_lines),
            )
        )

    generated_files = transform_tree(
        src_dir,
        out_dir,
        backend_namespace="vulkan",
        include_line="#include <vulkan_kernel_calls.hpp>",
        emit_named_kernels=True,
        on_emit_loop=collect_kernel,
        symbol_values=symbol_values,
    )
    generated_kernels = _write_kernel_stubs(kernel_specs, kernel_root)
    generated_spirv: list[Path] = []

    if args.shader_compiler:
        generated_spirv = _compile_kernel_stubs(args.shader_compiler, kernel_specs, kernel_root)

    if args.manifest:
        manifest = Path(args.manifest).resolve()
        write_manifest(manifest, "RLLM_VULKANIZED_SOURCES", generated_files)
        kernel_manifest = manifest.with_name("vulkan_kernel_sources.cmake")
        write_manifest(kernel_manifest, "RLLM_VULKAN_KERNEL_SOURCES", generated_kernels)
        if generated_spirv:
            spirv_manifest = manifest.with_name("vulkan_kernel_spirv.cmake")
            write_manifest(spirv_manifest, "RLLM_VULKAN_KERNEL_SPIRV", generated_spirv)

    print(
        "vulkanize.py: generated "
        f"{len(generated_files)} source files in {out_dir} and "
        f"{len(generated_kernels)} kernel files in {kernel_root}"
        + (
            f" and {len(generated_spirv)} SPIR-V files"
            if generated_spirv
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())