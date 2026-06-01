#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Callable


OFFLOAD_1D_PARAM_MACROS = {"OFFLOAD_PARFOR_1D_PARAM"}
OFFLOAD_2D_MACROS = {"OFFLOAD_PARFOR_2D"}
OFFLOAD_2D_PARAM_MACROS = {"OFFLOAD_PARFOR_2D_PARAM"}
OFFLOAD_2D_TRIANGULAR_PARAM_MACROS = {"OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM"}
OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS = {"OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM"}
OFFLOAD_ALL_MACROS = (
    OFFLOAD_1D_PARAM_MACROS
    | OFFLOAD_2D_MACROS
    | OFFLOAD_2D_PARAM_MACROS
    | OFFLOAD_2D_TRIANGULAR_PARAM_MACROS
    | OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS
)


@dataclass
class LoopContext:
    indent: str
    backend_namespace: str
    rel_path: str
    lineno: int
    is_2d: bool
    vars: list[str]
    range_expr: str
    kernel_guard_expr: str | None
    extra_params: str | None
    extra_param_types: dict[str, str] | None
    offload_param_lines: list[str] | None
    body_lines: list[str]
    emit_named_kernel: bool


_OFFLOAD_PARAMETERS_START_RE = re.compile(
    r"^\s*(?://\s*)?OFFLOAD_PARAMETERS\s*\((?P<names>[^)]*)\)\s*$"
)
_OFFLOAD_PARAMETERS_END_RE = re.compile(r"^\s*(?://\s*)?END_OFFLOAD_PARAMETERS\s*$")


def parse_identifier_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def parse_extra_param_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    if len(text) >= 2 and text[0] in "[{(" and text[-1] in "]})":
        text = text[1:-1].strip()
    return parse_identifier_list(text)


def parse_bound_param_name(raw: str) -> str | None:
    text = raw.strip()
    match = re.search(r"([A-Za-z_]\w*)\s*\)?\s*$", text)
    if match is None:
        return None
    return match.group(1)


def parse_offload_param_types_from_declaration_line(line: str, names: list[str]) -> dict[str, str]:
    # Matches both function-parameter style lines and local declarations inside
    # OFFLOAD_PARAMETERS ... END_OFFLOAD_PARAMETERS.
    # Examples:
    #   const flexible_rows_matrix<...>& x,
    #   std::vector<std::atomic<int>> visits(N);
    found: dict[str, str] = {}
    for name in names:
        stripped = line.strip()
        param_pattern = re.compile(rf"^\s*(?P<type>.+?)\s+{re.escape(name)}\s*(?:,|$)")
        local_pattern = re.compile(
            rf"^\s*(?P<type>.+?)\s+{re.escape(name)}\s*(?:\([^;]*\)|\{{[^;]*\}}|=[^;]*)?\s*;\s*$"
        )

        match = param_pattern.match(stripped)
        if match is None:
            match = local_pattern.match(stripped)
        if match is not None:
            found[name] = match.group("type").strip()
    return found


def default_symbol_values() -> dict[str, str]:
    # Fallback substitutions used when the enum-value helper executable is unavailable
    # at configure time (common for fresh build trees).
    values: dict[str, str] = {
        "EmbeddingDimension::START": "0",
        "EmbeddingDimension::MAX": "512",
        "PositionIndex::START": "0",
        "PositionIndex::MAX": "128",
        "PositionIndex::UNKNOWN_POSITION_INDEX": str((1 << 64) - 1),
        "HeadsIndex::START": "0",
        "HeadsIndex::MAX": "8",
        "MultiTokenPredictionIndex::START": "0",
        "MultiTokenPredictionIndex::ONE": "1",
        "MultiTokenPredictionIndex::TWO": "2",
        "MultiTokenPredictionIndex::THREE": "3",
        "MultiTokenPredictionIndex::FOUR": "4",
        "MultiTokenPredictionIndex::MAX": "4",
        "HeadDimension::START": "0",
        "HeadDimension::MAX": "64",
        "FFDimension::START": "0",
        "FFDimension::MAX": "2048",
        "NeuronConnectionIndex::START": "0",
        "NeuronConnectionIndex::MAX": "128",
        "TransformerBlock::MOMENTUM_BETA": "0.9",
        "TransformerBlock::GRAD_CLIP": "1.0",
        "TransformerBlock::VEL_CLIP": "0.1",
        "TransformerBlock::WEIGHT_CLAMP": "2.0",
        "rllm::TransformerBlock::MOMENTUM_BETA": "0.9",
        "rllm::TransformerBlock::GRAD_CLIP": "1.0",
        "rllm::TransformerBlock::VEL_CLIP": "0.1",
        "rllm::TransformerBlock::WEIGHT_CLAMP": "2.0",
    }

    return values


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


def _defined_macros_for_backend(backend_namespace: str) -> set[str]:
    defined: set[str] = set()
    if backend_namespace == "vulkan":
        defined.add("USE_VULKAN_OFFLOAD")
    elif backend_namespace == "hip":
        defined.add("USE_HIP_OFFLOAD")
    return defined


def _eval_preprocessor_expr(expr: str, defined_macros: set[str]) -> bool:
    text = expr.strip()
    if not text:
        return False

    text = re.sub(
        r"defined\s*\(\s*([A-Za-z_]\w*)\s*\)",
        lambda m: "True" if m.group(1) in defined_macros else "False",
        text,
    )
    text = re.sub(
        r"defined\s+([A-Za-z_]\w*)",
        lambda m: "True" if m.group(1) in defined_macros else "False",
        text,
    )

    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)

    def _replace_ident(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in {"True", "False", "and", "or", "not"}:
            return token
        return "True" if token in defined_macros else "False"

    text = re.sub(r"\b[A-Za-z_]\w*\b", _replace_ident, text)

    if re.search(r"[^()\sA-Za-z]", text):
        return False

    try:
        return bool(eval(text, {"__builtins__": {}}, {}))
    except Exception:
        return False


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
        return default_symbol_values()

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

        # Keep C++ enum casts intact in transformed sources; Vulkan GLSL
        # conversion handles casts separately in vulkanize.py sanitization.
        if search.startswith("static_cast<"):
            continue

        replacements[search] = replace

    return replacements


def _emit_loop_invocation(ctx: LoopContext) -> list[str]:
    lines: list[str] = []

    if ctx.backend_namespace == "vulkan":
        kernel_symbol, rel_spv = _vulkan_kernel_symbol_and_rel_spv(ctx.rel_path, ctx.lineno)
        launch_name = "launch_2d" if ctx.is_2d else "launch_1d"
        extra_param_names = parse_extra_param_names(ctx.extra_params)
        if extra_param_names:
            kernel_template_args = ", ".join(f"decltype(({name}))" for name in extra_param_names)
        else:
            kernel_template_args = ""
        lines.append(
            f"{ctx.indent}static rllm::vulkan::ComputeKernel<{kernel_template_args}> {kernel_symbol}("
        )
        lines.append(f"{ctx.indent}    \"{ctx.rel_path}:{ctx.lineno}\",")
        lines.append(f"{ctx.indent}    \"{rel_spv}\"")
        lines.append(f"{ctx.indent});")
        lines.append(f"{ctx.indent}{kernel_symbol}.{launch_name}(")
        lines.append(f"{ctx.indent}    {ctx.range_expr}{',' if extra_param_names else ''}")
        for idx, name in enumerate(extra_param_names):
            suffix = "," if idx + 1 < len(extra_param_names) else ""
            lines.append(f"{ctx.indent}    {name}{suffix}")
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
    r"enum_iterator\s*<\s*(?P<type>[^>]+?)\s*>\s*\(\s*(?P<arg>.*)\s*\)\s*\)\s*(?P<brace>\{)?\s*$"
)
_INC_CALL_RE = re.compile(r"^inc\s*\(\s*(?P<expr>.+)\s*\)$")


def _normalize_enum_iterator_bound(expr: str, symbol_values: dict[str, str] | None = None) -> str:
    e = apply_symbol_values(expr.strip(), symbol_values)
    m = _INC_CALL_RE.match(e)
    if m is not None:
        inner = m.group("expr").strip()
        return f"(int({inner}) + 1)"
    return f"int({e})"


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
        range_args = [arg for arg in split_top_level_args(match.group("arg").strip()) if arg.strip()]
        has_brace = match.group("brace") is not None

        # Avoid reserved double-underscore identifiers in generated C++/GLSL.
        idx_name = f"rllm_idx_{var}"
        if len(range_args) == 0:
            max_symbol = f"{enum_type}::MAX"
            start_expr = "0"
            limit_expr = f"int({apply_symbol_values(max_symbol, symbol_values)})"
        elif len(range_args) == 1:
            start_expr = "0"
            limit_expr = _normalize_enum_iterator_bound(range_args[0], symbol_values)
        else:
            start_expr = _normalize_enum_iterator_bound(range_args[0], symbol_values)
            limit_expr = _normalize_enum_iterator_bound(range_args[1], symbol_values)

        for_header = f"{indent}for (int {idx_name} = {start_expr}; {idx_name} < {limit_expr}; ++{idx_name})"

        if has_brace:
            rewritten.append(f"{for_header} {{")
            alias_line = f"{indent}    const int {var} = int({idx_name});"
            rewritten.append(apply_symbol_values(alias_line, symbol_values))
            i += 1
            continue

        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1

        rewritten.append(f"{for_header} {{")
        alias_line = f"{indent}    const int {var} = int({idx_name});"
        rewritten.append(apply_symbol_values(alias_line, symbol_values))

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
    defined_macros = _defined_macros_for_backend(backend_namespace)
    conditional_stack: list[tuple[bool, bool]] = []
    active_code = True
    active_offload_param_types: dict[str, str] = {}
    active_offload_param_lines: list[str] = []
    collecting_offload_params = False
    collecting_param_names: list[str] = []
    collecting_param_types: dict[str, str] = {}
    collecting_param_lines: list[str] = []

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

    def append_directive(line: str) -> None:
        # Keep preprocessor control lines out of captured kernel bodies.
        if loop_stack:
            out_lines.append(line)
        else:
            out_lines.append(line)

    for lineno, line in enumerate(in_lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if_match = re.match(r"^#\s*if\s+(.*)$", stripped)
            ifdef_match = re.match(r"^#\s*ifdef\s+([A-Za-z_]\w*)\s*$", stripped)
            ifndef_match = re.match(r"^#\s*ifndef\s+([A-Za-z_]\w*)\s*$", stripped)
            elif_match = re.match(r"^#\s*elif\s+(.*)$", stripped)
            else_match = re.match(r"^#\s*else\s*$", stripped)
            endif_match = re.match(r"^#\s*endif\s*$", stripped)

            if if_match:
                parent_active = active_code
                branch_active = parent_active and _eval_preprocessor_expr(if_match.group(1), defined_macros)
                conditional_stack.append((parent_active, branch_active))
                active_code = branch_active
                append_directive(line)
                continue

            if ifdef_match:
                parent_active = active_code
                branch_active = parent_active and (ifdef_match.group(1) in defined_macros)
                conditional_stack.append((parent_active, branch_active))
                active_code = branch_active
                append_directive(line)
                continue

            if ifndef_match:
                parent_active = active_code
                branch_active = parent_active and (ifndef_match.group(1) not in defined_macros)
                conditional_stack.append((parent_active, branch_active))
                active_code = branch_active
                append_directive(line)
                continue

            if elif_match and conditional_stack:
                parent_active, any_taken = conditional_stack[-1]
                branch_active = parent_active and (not any_taken) and _eval_preprocessor_expr(elif_match.group(1), defined_macros)
                conditional_stack[-1] = (parent_active, any_taken or branch_active)
                active_code = branch_active
                append_directive(line)
                continue

            if else_match and conditional_stack:
                parent_active, any_taken = conditional_stack[-1]
                branch_active = parent_active and (not any_taken)
                conditional_stack[-1] = (parent_active, True)
                active_code = branch_active
                append_directive(line)
                continue

            if endif_match and conditional_stack:
                parent_active, _ = conditional_stack.pop()
                active_code = parent_active
                append_directive(line)
                continue

        if not active_code:
            if loop_stack:
                out_lines.append(line)
            else:
                out_lines.append(line)
            continue

        if collecting_offload_params:
            if _OFFLOAD_PARAMETERS_END_RE.match(line.strip()):
                collecting_offload_params = False
                active_offload_param_types = dict(collecting_param_types)
                active_offload_param_lines = list(collecting_param_lines)
                collecting_param_names = []
                collecting_param_types = {}
                collecting_param_lines = []
                append_to_current(line)
                continue

            collecting_param_types.update(
                parse_offload_param_types_from_declaration_line(line, collecting_param_names)
            )
            collecting_param_lines.append(line)
            append_to_current(line)
            continue

        offload_params_start = _OFFLOAD_PARAMETERS_START_RE.match(line.strip())
        if offload_params_start:
            collecting_offload_params = True
            collecting_param_names = parse_identifier_list(offload_params_start.group("names"))
            collecting_param_types = {}
            append_to_current(line)
            continue

        parsed = parse_macro_invocation(line)
        if stripped == "ENDFOR" and loop_stack:
            ctx = loop_stack.pop()
            ctx.range_expr = apply_symbol_values(ctx.range_expr, symbol_values)
            if ctx.extra_params is not None:
                ctx.extra_params = apply_symbol_values(ctx.extra_params, symbol_values)
            if ctx.extra_param_types:
                ctx.extra_param_types = {
                    name: apply_symbol_values(type_name, symbol_values)
                    for name, type_name in ctx.extra_param_types.items()
                }
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
        if macro in OFFLOAD_1D_PARAM_MACROS and len(args) >= 3:
            extra_param_names = parse_extra_param_names(", ".join(args[2:]))
            extra_param_types = {
                name: active_offload_param_types[name]
                for name in extra_param_names
                if name in active_offload_param_types
            }
            loop_stack.append(
                LoopContext(
                    indent=indent,
                    backend_namespace=backend_namespace,
                    rel_path=rel_path,
                    lineno=lineno,
                    is_2d=False,
                    vars=[args[0]],
                    range_expr=apply_symbol_values(args[1], symbol_values),
                    kernel_guard_expr=None,
                    extra_params=", ".join(args[2:]),
                    extra_param_types=extra_param_types,
                    offload_param_lines=list(active_offload_param_lines),
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
                    kernel_guard_expr=None,
                    extra_params=None,
                    extra_param_types=None,
                    offload_param_lines=list(active_offload_param_lines),
                    body_lines=[],
                    emit_named_kernel=emit_named_kernels,
                )
            )
            changed = True
            continue

        if macro in OFFLOAD_2D_PARAM_MACROS and len(args) >= 4:
            extra_param_names = parse_extra_param_names(", ".join(args[3:]))
            extra_param_types = {
                name: active_offload_param_types[name]
                for name in extra_param_names
                if name in active_offload_param_types
            }
            loop_stack.append(
                LoopContext(
                    indent=indent,
                    backend_namespace=backend_namespace,
                    rel_path=rel_path,
                    lineno=lineno,
                    is_2d=True,
                    vars=[args[0], args[1]],
                    range_expr=apply_symbol_values(args[2], symbol_values),
                    kernel_guard_expr=None,
                    extra_params=", ".join(args[3:]),
                    extra_param_types=extra_param_types,
                    offload_param_lines=list(active_offload_param_lines),
                    body_lines=[],
                    emit_named_kernel=emit_named_kernels,
                )
            )
            changed = True
            continue

        if macro in OFFLOAD_2D_TRIANGULAR_PARAM_MACROS and len(args) >= 4:
            extra_param_names = parse_extra_param_names(", ".join(args[3:]))
            bound_param_name = parse_bound_param_name(args[2])
            if bound_param_name and bound_param_name not in extra_param_names:
                extra_param_names.append(bound_param_name)
            extra_param_types = {
                name: active_offload_param_types[name]
                for name in extra_param_names
                if name in active_offload_param_types
            }
            bound_expr = apply_symbol_values(args[2], symbol_values)
            guard_expr = f"{args[1]} > {args[0]}"
            if bound_param_name:
                guard_expr = (
                    f"{args[0]} >= int({bound_param_name}) || "
                    f"{args[1]} >= int({bound_param_name}) || "
                    f"{guard_expr}"
                )
            loop_stack.append(
                LoopContext(
                    indent=indent,
                    backend_namespace=backend_namespace,
                    rel_path=rel_path,
                    lineno=lineno,
                    is_2d=True,
                    vars=[args[0], args[1]],
                    range_expr=f"rllm::enum_iterator2D<decltype({bound_expr}), decltype({bound_expr})>({bound_expr})",
                    kernel_guard_expr=guard_expr,
                    extra_params=", ".join(extra_param_names),
                    extra_param_types=extra_param_types,
                    offload_param_lines=list(active_offload_param_lines),
                    body_lines=[],
                    emit_named_kernel=emit_named_kernels,
                )
            )
            changed = True
            continue

        if macro in OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS and len(args) >= 4:
            extra_param_names = parse_extra_param_names(", ".join(args[3:]))
            bound_param_name = parse_bound_param_name(args[2])
            if bound_param_name and bound_param_name not in extra_param_names:
                extra_param_names.append(bound_param_name)
            extra_param_types = {
                name: active_offload_param_types[name]
                for name in extra_param_names
                if name in active_offload_param_types
            }
            bound_expr = apply_symbol_values(args[2], symbol_values)
            guard_expr = f"{args[0]} > {args[1]}"
            if bound_param_name:
                guard_expr = (
                    f"{args[0]} >= int({bound_param_name}) || "
                    f"{args[1]} >= int({bound_param_name}) || "
                    f"{guard_expr}"
                )
            loop_stack.append(
                LoopContext(
                    indent=indent,
                    backend_namespace=backend_namespace,
                    rel_path=rel_path,
                    lineno=lineno,
                    is_2d=True,
                    vars=[args[0], args[1]],
                    range_expr=f"rllm::enum_iterator2D<decltype({bound_expr}), decltype({bound_expr})>({bound_expr})",
                    kernel_guard_expr=guard_expr,
                    extra_params=", ".join(extra_param_names),
                    extra_param_types=extra_param_types,
                    offload_param_lines=list(active_offload_param_lines),
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
