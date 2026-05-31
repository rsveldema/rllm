#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OFFLOAD_1D_MACROS = {"OFFLOADABLE_PARFOR", "OFFLOAD_PARFOR"}
OFFLOAD_2D_MACROS = {"OFFLOADABLE_PARFOR_2D", "OFFLOAD_PARFOR_2D"}
OFFLOAD_ALL_MACROS = OFFLOAD_1D_MACROS | OFFLOAD_2D_MACROS


@dataclass
class LoopContext:
    indent: str


def split_top_level_args(raw: str) -> list[str]:
    args: list[str] = []
    depth_round = 0
    depth_square = 0
    depth_curly = 0
    depth_angle = 0
    start = 0

    for i, ch in enumerate(raw):
        if ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round = max(depth_round - 1, 0)
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square = max(depth_square - 1, 0)
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly = max(depth_curly - 1, 0)
        elif ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(depth_angle - 1, 0)
        elif (
            ch == ","
            and depth_round == 0
            and depth_square == 0
            and depth_curly == 0
            and depth_angle == 0
        ):
            args.append(raw[start:i].strip())
            start = i + 1

    args.append(raw[start:].strip())
    return args


def parse_macro_invocation(line: str) -> tuple[str, list[str], str] | None:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    for macro in OFFLOAD_ALL_MACROS:
        prefix = f"{macro}("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            continue
        inside = stripped[len(prefix) : -1]
        args = split_top_level_args(inside)
        return macro, args, indent

    return None


def inject_kernel_include(lines: list[str], include_line: str) -> list[str]:
    if any(line.strip() == include_line for line in lines):
        return lines

    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#include "):
            insert_at = i + 1

    return lines[:insert_at] + [include_line] + lines[insert_at:]


def transform_source(src_text: str, rel_path: str, backend_namespace: str, include_line: str) -> tuple[str, bool]:
    in_lines = src_text.splitlines()
    out_lines: list[str] = []
    loop_stack: list[LoopContext] = []
    changed = False

    for lineno, line in enumerate(in_lines, start=1):
        parsed = parse_macro_invocation(line)
        if parsed is None:
            stripped = line.strip()
            if stripped == "ENDFOR" and loop_stack:
                ctx = loop_stack.pop()
                out_lines.append(f"{ctx.indent}    }}")
                out_lines.append(f"{ctx.indent});")
                changed = True
            else:
                out_lines.append(line)
            continue

        macro, args, indent = parsed
        if macro in OFFLOAD_1D_MACROS and len(args) == 2:
            var = args[0]
            range_expr = args[1]
            out_lines.append(f"{indent}rllm::{backend_namespace}::launch_kernel_1d(")
            out_lines.append(f"{indent}    \"{rel_path}:{lineno}\",")
            out_lines.append(f"{indent}    {range_expr},")
            out_lines.append(f"{indent}    [&](auto {var}) {{")
            loop_stack.append(LoopContext(indent=indent))
            changed = True
            continue

        if macro in OFFLOAD_2D_MACROS and len(args) >= 3:
            var1 = args[0]
            var2 = args[1]
            range_expr = ", ".join(args[2:])
            out_lines.append(f"{indent}rllm::{backend_namespace}::launch_kernel_2d(")
            out_lines.append(f"{indent}    \"{rel_path}:{lineno}\",")
            out_lines.append(f"{indent}    {range_expr},")
            out_lines.append(f"{indent}    [&](auto {var1}, auto {var2}) {{")
            loop_stack.append(LoopContext(indent=indent))
            changed = True
            continue

        out_lines.append(line)

    if changed:
        out_lines = inject_kernel_include(out_lines, include_line)

    return "\n".join(out_lines) + "\n", changed


def transform_tree(src_dir: Path, out_dir: Path, backend_namespace: str, include_line: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for src_path in sorted(src_dir.rglob("*")):
        if not src_path.is_file() or src_path.suffix not in {".cc", ".cpp"}:
            continue

        rel_path = src_path.relative_to(src_dir)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        transformed, _ = transform_source(
            src_path.read_text(encoding="utf-8"),
            rel_path.as_posix(),
            backend_namespace,
            include_line,
        )
        out_path.write_text(transformed, encoding="utf-8")
        generated_files.append(out_path)

    return generated_files


def write_manifest(manifest: Path, variable_name: str, generated_files: list[Path]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"set({variable_name}"]
    lines.extend(f'    "{path.as_posix()}"' for path in generated_files)
    lines.append(")")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
