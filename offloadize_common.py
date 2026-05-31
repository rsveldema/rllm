#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Callable


OFFLOAD_1D_MACROS = {"OFFLOADABLE_PARFOR", "OFFLOAD_PARFOR"}
OFFLOAD_2D_MACROS = {"OFFLOADABLE_PARFOR_2D", "OFFLOAD_PARFOR_2D"}
OFFLOAD_ALL_MACROS = OFFLOAD_1D_MACROS | OFFLOAD_2D_MACROS


@dataclass
class LoopContext:
    indent: str
    backend_namespace: str
    rel_path: str
    lineno: int
    is_2d: bool
    vars: list[str]
    range_expr: str
    body_lines: list[str]
    emit_named_kernel: bool


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


def _sanitize_name_component(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _vulkan_kernel_symbol_and_rel_spv(rel_path: str, lineno: int) -> tuple[str, str]:
    rel = Path(rel_path)
    stem = _sanitize_name_component(rel.stem)
    parent = rel.parent.as_posix()
    rel_spv = f"{stem}.L{lineno}.spv"
    if parent and parent != ".":
        rel_spv = f"{parent}/{rel_spv}"
    symbol = f"__rllm_vulkan_kernel_{stem}_L{lineno}"
    return symbol, rel_spv


def apply_symbol_values(text: str, symbol_values: dict[str, str] | None) -> str:
    if not symbol_values:
        return text

    out = text
    for symbol, value in symbol_values.items():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
        out = re.sub(pattern, value, out)
    return out


def resolve_symbol_values(enum_value_tool: str | None) -> dict[str, str]:
    if not enum_value_tool:
        return {}

    cmd = [enum_value_tool, "--print-json"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Enum value tool failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Enum value tool returned invalid JSON.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
            f"error: {exc}"
        )

    if not isinstance(payload, list):
        raise RuntimeError(
            "Enum value tool JSON must be an array of search/replace objects.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    replacements: dict[str, str] = {}
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(
                "Enum value tool JSON entries must be objects with string fields 'search' and 'replace'.\n"
                f"entry index: {idx}\n"
                f"entry value: {item!r}"
            )

        search = item.get("search")
        replace = item.get("replace")
        if not isinstance(search, str) or not isinstance(replace, str):
            raise RuntimeError(
                "Enum value tool JSON entries must have string fields 'search' and 'replace'.\n"
                f"entry index: {idx}\n"
                f"entry value: {item!r}"
            )

        replacements[search] = replace

    return replacements


def _emit_loop_invocation(ctx: LoopContext) -> list[str]:
    lines: list[str] = []

    if ctx.backend_namespace == "vulkan":
        kernel_symbol, rel_spv = _vulkan_kernel_symbol_and_rel_spv(ctx.rel_path, ctx.lineno)
        launch_name = "launch_2d" if ctx.is_2d else "launch_1d"
        lines.append(f"{ctx.indent}static rllm::vulkan::ComputeKernel {kernel_symbol}(")
        lines.append(f"{ctx.indent}    \"{ctx.rel_path}:{ctx.lineno}\",")
        lines.append(f"{ctx.indent}    \"{rel_spv}\"")
        lines.append(f"{ctx.indent});")
        lines.append(f"{ctx.indent}{kernel_symbol}.{launch_name}(")
        lines.append(f"{ctx.indent}    {ctx.range_expr}")
        lines.append(f"{ctx.indent});")
        return lines

    if ctx.emit_named_kernel:
        kernel_name = f"__rllm_{ctx.backend_namespace}_kernel_{ctx.lineno}"
        if ctx.is_2d:
            lines.append(f"{ctx.indent}auto {kernel_name} = [&](auto {ctx.vars[0]}, auto {ctx.vars[1]}) {{")
        else:
            lines.append(f"{ctx.indent}auto {kernel_name} = [&](auto {ctx.vars[0]}) {{")
        lines.extend(ctx.body_lines)
        lines.append(f"{ctx.indent}}};")
        launch_arg = kernel_name
    else:
        if ctx.is_2d:
            lines.append(
                f"{ctx.indent}rllm::{ctx.backend_namespace}::launch_kernel_2d("
            )
        else:
            lines.append(
                f"{ctx.indent}rllm::{ctx.backend_namespace}::launch_kernel_1d("
            )
        lines.append(f"{ctx.indent}    \"{ctx.rel_path}:{ctx.lineno}\",")
        lines.append(f"{ctx.indent}    {ctx.range_expr},")
        if ctx.is_2d:
            lines.append(f"{ctx.indent}    [&](auto {ctx.vars[0]}, auto {ctx.vars[1]}) {{")
        else:
            lines.append(f"{ctx.indent}    [&](auto {ctx.vars[0]}) {{")
        lines.extend(ctx.body_lines)
        lines.append(f"{ctx.indent}    }}")
        lines.append(f"{ctx.indent});")
        return lines

    if ctx.is_2d:
        lines.append(
            f"{ctx.indent}rllm::{ctx.backend_namespace}::launch_kernel_2d("
        )
    else:
        lines.append(
            f"{ctx.indent}rllm::{ctx.backend_namespace}::launch_kernel_1d("
        )
    lines.append(f"{ctx.indent}    \"{ctx.rel_path}:{ctx.lineno}\",")
    lines.append(f"{ctx.indent}    {ctx.range_expr},")
    lines.append(f"{ctx.indent}    {launch_arg}")
    lines.append(f"{ctx.indent});")
    return lines


_ENUM_ITERATOR_FOR_RE = re.compile(
    r"^(?P<indent>\s*)for\s*\(\s*const\s+auto\s+(?P<var>[A-Za-z_]\w*)\s*:\s*"
    r"enum_iterator\s*<\s*(?P<type>[^>]+?)\s*>\s*\(\s*(?P<arg>[^)]*)\s*\)\s*\)\s*(?P<brace>\{)?\s*$"
)


def rewrite_enum_iterator_loops(
    lines: list[str],
    symbol_values: dict[str, str] | None = None,
) -> list[str]:
    rewritten: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        match = _ENUM_ITERATOR_FOR_RE.match(line)
        if match is None:
            rewritten.append(line)
            i += 1
            continue

        indent = match.group("indent")
        var = match.group("var")
        enum_type = match.group("type").strip()
        range_arg = apply_symbol_values(match.group("arg").strip(), symbol_values)
        has_brace = match.group("brace") is not None

        idx_name = f"__rllm_idx_{var}"
        if range_arg:
            limit_expr = f"static_cast<size_t>({range_arg})"
        else:
            max_symbol = f"{enum_type}::MAX"
            limit_expr = f"static_cast<size_t>({apply_symbol_values(max_symbol, symbol_values)})"

        for_header = (
            f"{indent}for (size_t {idx_name} = 0; {idx_name} < {limit_expr}; ++{idx_name})"
        )

        if has_brace:
            rewritten.append(f"{for_header} {{")
            rewritten.append(
                f"{indent}    const auto {var} = static_cast<{enum_type}>({idx_name});"
            )
            i += 1
            continue

        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1

        rewritten.append(f"{for_header} {{")
        rewritten.append(
            f"{indent}    const auto {var} = static_cast<{enum_type}>({idx_name});"
        )

        if i >= len(lines):
            rewritten.append(f"{indent}}}")
            continue

        next_line = lines[i]
        if next_line.strip() == "{":
            # Brace on the next line: preserve braced body while injecting the alias.
            i += 1
            continue

        rewritten.append(f"{indent}    {next_line.lstrip()}")
        rewritten.append(f"{indent}}}")
        i += 1

    return rewritten


def transform_source(
    src_text: str,
    rel_path: str,
    backend_namespace: str,
    include_line: str,
    emit_named_kernels: bool = False,
    on_emit_loop: Callable[[LoopContext], None] | None = None,
    symbol_values: dict[str, str] | None = None,
) -> tuple[str, bool]:
    in_lines = src_text.splitlines()
    out_lines: list[str] = []
    loop_stack: list[LoopContext] = []
    changed = False

    def append_to_current(line: str) -> None:
        if loop_stack:
            loop_stack[-1].body_lines.append(line)
        else:
            out_lines.append(line)

    def append_many_to_current(lines: list[str]) -> None:
        if loop_stack:
            loop_stack[-1].body_lines.extend(lines)
        else:
            out_lines.extend(lines)

    for lineno, line in enumerate(in_lines, start=1):
        parsed = parse_macro_invocation(line)
        stripped = line.strip()
        if stripped == "ENDFOR" and loop_stack:
            ctx = loop_stack.pop()
            ctx.range_expr = apply_symbol_values(ctx.range_expr, symbol_values)
            ctx.body_lines = [apply_symbol_values(body_line, symbol_values) for body_line in ctx.body_lines]
            ctx.body_lines = rewrite_enum_iterator_loops(ctx.body_lines, symbol_values)
            if on_emit_loop is not None:
                on_emit_loop(ctx)
            append_many_to_current(_emit_loop_invocation(ctx))
            changed = True
            continue

        if parsed is None:
            append_to_current(line)
            continue

        macro, args, indent = parsed
        if macro in OFFLOAD_1D_MACROS and len(args) == 2:
            loop_stack.append(
                LoopContext(
                    indent=indent,
                    backend_namespace=backend_namespace,
                    rel_path=rel_path,
                    lineno=lineno,
                    is_2d=False,
                    vars=[args[0]],
                    range_expr=apply_symbol_values(args[1], symbol_values),
                    body_lines=[],
                    emit_named_kernel=emit_named_kernels,
                )
            )
            changed = True
            continue

        if macro in OFFLOAD_2D_MACROS and len(args) >= 3:
            loop_stack.append(
                LoopContext(
                    indent=indent,
                    backend_namespace=backend_namespace,
                    rel_path=rel_path,
                    lineno=lineno,
                    is_2d=True,
                    vars=[args[0], args[1]],
                    range_expr=apply_symbol_values(", ".join(args[2:]), symbol_values),
                    body_lines=[],
                    emit_named_kernel=emit_named_kernels,
                )
            )
            changed = True
            continue

        append_to_current(line)

    if changed:
        out_lines = inject_kernel_include(out_lines, include_line)

    return "\n".join(out_lines) + "\n", changed


def transform_tree(
    src_dir: Path,
    out_dir: Path,
    backend_namespace: str,
    include_line: str,
    emit_named_kernels: bool = False,
    on_emit_loop: Callable[[LoopContext], None] | None = None,
    symbol_values: dict[str, str] | None = None,
) -> list[Path]:
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
            emit_named_kernels=emit_named_kernels,
            on_emit_loop=on_emit_loop,
            symbol_values=symbol_values,
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
