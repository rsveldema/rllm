#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
from typing import Callable


OFFLOAD_1D_PARAM_MACROS = {"OFFLOAD_PARFOR_1D_PARAM"}
OFFLOAD_2D_MACROS = {"OFFLOAD_PARFOR_2D"}
OFFLOAD_2D_PARAM_MACROS = {"OFFLOAD_PARFOR_2D_PARAM"}
OFFLOAD_3D_PARAM_MACROS = {"OFFLOAD_PARFOR_3D_PARAM"}
OFFLOAD_3D_TRIANGULAR_PARAM_MACROS = {"OFFLOAD_PARFOR_3D_TRIANGULAR_PARAM"}
OFFLOAD_2D_TRIANGULAR_PARAM_MACROS = {"OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM"}
OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS = {"OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM"}
OFFLOAD_ALL_MACROS = (
    OFFLOAD_1D_PARAM_MACROS
    | OFFLOAD_2D_MACROS
    | OFFLOAD_2D_PARAM_MACROS
    | OFFLOAD_3D_PARAM_MACROS
    | OFFLOAD_3D_TRIANGULAR_PARAM_MACROS
    | OFFLOAD_2D_TRIANGULAR_PARAM_MACROS
    | OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS
)



def load_json(path: Path) -> dict | list | str | int | float | bool | None:
    """Load and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict | list | str | int | float | bool | None) -> None:
    """Write data as JSON to a file."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

@dataclass
class LoopContext:
    indent: str
    backend_namespace: str
    rel_path: str
    lineno: int
    is_2d: bool
    is_3d: bool
    vars: list[str]
    range_expr: str
    kernel_guard_expr: str | None
    extra_params: str | None
    extra_param_types: dict[str, str] | None
    offload_param_lines: list[str] | None
    body_lines: list[str]
    emit_named_kernel: bool
    parfor_invocation: str | None = None  # original PARFOR invocation line text
    raw_body_lines: list[str] | None = None  # unmodified source lines (before symbol substitution)
    raw_offload_param_lines: list[str] | None = None  # unmodified OFFLOAD_PARAMETERS content
    shared_vars: dict[str, str] | None = field(default=None)  # var_name -> type (extracted from PARFOR_SHARED_VARIABLES blocks)
    constexpr_defines: list[tuple[str, str]] | None = field(default=None)  # [(name, expr), ...] extracted from body
    triangular_kind: str | None = None  # "upper" or "lower" for triangular parfors
    queue_expr: str | None = None


_OFFLOAD_PARAMETERS_START_RE = re.compile(
    r"^\s*(?://\s*)?OFFLOAD_PARAMETERS\s*\((?P<names>[^)]*)\)\s*$"
)
_OFFLOAD_PARAMETERS_END_RE = re.compile(r"^\s*(?://\s*)?END_OFFLOAD_PARAMETERS\s*$")
_SHARED_VARIABLES_START_RE = re.compile(
    r"^\s*//\s*PARFOR_SHARED_VARIABLES\s*\((?P<names>[^)]*)\)\s*$"
)
_SHARED_VARIABLES_END_RE = re.compile(r"^\s*//\s*ENDPARFOR_SHARED_VARIABLES\s*$")


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


def extract_shared_variables_from_body_lines(body_lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """
    Extract PARFOR_SHARED_VARIABLES blocks from body_lines.
    Returns (shared_vars_dict, filtered_body_lines) where:
      shared_vars_dict maps variable names to their types
      filtered_body_lines has the PARFOR_SHARED_VARIABLES blocks removed
    """
    shared_vars: dict[str, str] = {}
    filtered: list[str] = []
    i = 0

    # Regex to match: // PARFOR_SHARED_VARIABLES(var_name1, var_name2, ...)
    # Regex to extract type and variable name from declaration: float max_val = -std::numeric_limits<float>::infinity();
    var_decl_re = re.compile(
        r"^\s*(?P<type>(?:const\s+)?(?:[a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)*\s*)+)\s+(?P<name>[a-zA-Z_]\w*)\s*(?:=|;|;)"
    )

    while i < len(body_lines):
        line = body_lines[i]
        match = _SHARED_VARIABLES_START_RE.match(line)

        if match:
            # Found start of shared variables block
            var_names_str = match.group(1)
            var_names = [name.strip() for name in var_names_str.split(",")]

            # Skip the PARFOR_SHARED_VARIABLES line
            i += 1

            # Collect variable declarations until ENDPARFOR_SHARED_VARIABLES
            while i < len(body_lines):
                decl_line = body_lines[i]
                end_match = _SHARED_VARIABLES_END_RE.match(decl_line)

                if end_match:
                    # Skip the ENDPARFOR_SHARED_VARIABLES line
                    i += 1
                    break

                # Try to extract type from the declaration
                decl_match = var_decl_re.match(decl_line)
                if decl_match:
                    var_name = decl_match.group("name")
                    var_type = decl_match.group("type").strip()
                    if var_name in var_names:
                        shared_vars[var_name] = var_type

                i += 1
        else:
            filtered.append(line)
            i += 1

    return shared_vars, filtered


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
    return defined


def _eval_preprocessor_expr(expr: str, defined_macros: set[str]) -> bool:
    text = expr.split("//", 1)[0].strip()
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
    for _ in range(4):
        prev = out
        for symbol, value in symbol_values.items():
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
            out = re.sub(pattern, value, out)
        if out == prev:
            break
    return out

def hard_apply_symbol_values(text: str, symbol_values: dict[str, str] | None) -> str:
    if not symbol_values:
        return text

    out = text

    out = out.replace("enum_iterator1D<TokenID>(",
                      "limit<TokenID::MAX>(")
    out = out.replace("enum_iterator1D<TempStorage>(",
                      "limit<TempStorage::MAX>(")
    out = out.replace("enum_iterator1D<PositionIndex>(",
                      "limit<PositionIndex::MAX>(")
    out = out.replace("enum_iterator1D<EmbeddingDimension>(",
                      "limit<EmbeddingDimension::MAX>(")
    out = out.replace("enum_iterator1D<HeadsIndex>(",
                      "limit<HeadsIndex::MAX>(")
    out = out.replace("enum_iterator1D<HeadDimension>(",
                      "limit<HeadDimension::MAX>(")
    out = out.replace("GpuInputLine",
                      "fixed_size_vector<TokenID, PositionIndex>")
    out = out.replace("CpuInputLine",
                      "cpu_fixed_vector<TokenID, PositionIndex>")
    out = out.replace("InputLine",
                      "fixed_size_vector<TokenID, PositionIndex>")
    out = out.replace("PositionIndex::START", "0")
    out = out.replace("EmbeddingDimension::START", "0")
    out = out.replace("HeadsIndex::START", "0")
    out = out.replace("TokenID::START", "0")
    out = out.replace("TempStorage::START", "0")
    out = out.replace("fixed_size_vector<TokenID,", "fixed_size_vector<int,")
    out = out.replace(", TokenID,", ", TokenID::MAX,")
    out = out.replace(", TokenID>", ", TokenID::MAX>")
    out = out.replace(", TempStorage", ", TempStorage::MAX")
    out = out.replace(", HeadsIndex", ", HeadsIndex::MAX")
    out = out.replace(", PositionIndex", ", PositionIndex::MAX")
    out = out.replace(", EmbeddingDimension", ", EmbeddingDimension::MAX")
    out = out.replace(", FFDimension", ", FFDimension::MAX")
    out = out.replace(", HeadDimension", ", HeadDimension::MAX")
    out = out.replace(", RmsNormPartialSumIndex", ", RmsNormPartialSumIndex::MAX")
    out = out.replace(", MultiTokenPredictionIndex", ", MultiTokenPredictionIndex::MAX")
    out = out.replace(", NeuronConnectionIndex", ", NeuronConnectionIndex::MAX")
    out = out.replace(", ConflictIndex", ", ConflictIndex::MAX")




    # MAX suffix fallback: use \b word boundaries to avoid matching inside <...> templates
    if out.find("::MAX") < 0 and (out.find("_matrix") >= 0 or out.find("_vector") >= 0):
        out = re.sub(r"\bTempStorage\b", "TempStorage::MAX", out)
        out = re.sub(r"\bHeadsIndex\b", "HeadsIndex::MAX", out)
        out = re.sub(r"\bPositionIndex\b", "PositionIndex::MAX", out)
        out = re.sub(r"\bEmbeddingDimension\b", "EmbeddingDimension::MAX", out)

    # enum as a loose parameter type:
    out = out.replace(" PositionIndex ", " int ")
    out = out.replace("EmbeddingDimension ", " int ")

    # Replace bare obsolete enum-type names used as loose parameter types with int.
    # Use negative lookbehind (?<!<) to exclude matches inside <...> template args,
    # and negative lookahead (?!::) to avoid corrupting already-suffixed types like PositionIndex::MAX.
    for _obsolete_type in (
        "TokenID", "PositionIndex", "EmbeddingDimension", "HeadsIndex",
        "TempStorage", "FFDimension", "HeadDimension",
        "RmsNormPartialSumIndex", "MultiTokenPredictionIndex",
        "NeuronConnectionIndex", "ConflictIndex"
    ):
        out = re.sub(rf"(?<!<)\b{_obsolete_type}\b(?!::)(?!\s*>)", "int", out)
    # Kernel dumps are parsed by kernel_compiler, not a C++ compiler. Strip
    # C++ casts and normalize C++-only scalar spellings before parsing.
    out = _strip_static_casts(out)

    out = re.sub(r"\bsize_t\b", "int", out)
    out = out.replace(" auto ", " int ")
    out = out.replace("std::pow", " pow ")
    out = out.replace("std::exp", " exp ")
    out = out.replace("std::sin", " sin ")
    out = out.replace("std::cos", " cos ")
    out = out.replace("std::tanh", " tanh ")
    out = out.replace("std::tan", " tan ")
    out = out.replace("math::clamp", " clamp ")
    out = out.replace("math::max", " max ")
    out = out.replace("std::sqrt", " sqrt ")
    out = re.sub(r"\b([A-Za-z_]\w*)\s*&\s*~1\b", r"(\1 - (\1 % 2))", out)
    out = re.sub(r"\b([A-Za-z_]\w*)\s*%\s*2\s*==\s*0\b", r"((\1 % 2) == 0)", out)
    out = re.sub(r"limit<([^>]+)>\(\s*inc\(([^)]+)\)\s*\)", r"limit<\1>(0, inc(\2))", out)

    for symbol, value in symbol_values.items():
        out = out.replace(symbol, value)

    out = out.replace("[[maybe_unused]]", "")
    return out


def _strip_static_casts(text: str) -> str:
    needle = "static_cast"
    out = text
    while True:
        start = out.find(needle)
        if start < 0:
            return out

        lt = out.find("<", start + len(needle))
        if lt < 0 or out[start + len(needle):lt].strip():
            return out
        gt = out.find(">", lt + 1)
        if gt < 0:
            return out
        lparen = out.find("(", gt + 1)
        if lparen < 0 or out[gt + 1:lparen].strip():
            return out

        depth = 0
        end = None
        for idx in range(lparen, len(out)):
            ch = out[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end is None:
            return out

        inner = out[lparen + 1:end]
        out = out[:start] + inner + out[end + 1:]


def resolve_symbol_values(enum_value_tool: str | None) -> dict[str, str]:
    assert enum_value_tool

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
        extra_param_names = parse_extra_param_names(ctx.extra_params)
        _ = rel_spv
        queue_expr = ctx.queue_expr or "rllm::vulkan_runtime::get_queue(0)"
        lines.append(f"{ctx.indent}rllm::vulkan::generated::{kernel_symbol}(")
        lines.append(f"{ctx.indent}    {ctx.range_expr},")
        lines.append(f"{ctx.indent}    {queue_expr}{',' if extra_param_names else ''}")
        for idx, name in enumerate(extra_param_names):
            suffix = "," if idx + 1 < len(extra_param_names) else ""
            lines.append(f"{ctx.indent}    {name}{suffix}")
        lines.append(f"{ctx.indent});")
        return lines

    if ctx.emit_named_kernel:
        kernel_name = f"__rllm_{ctx.backend_namespace}_kernel_{ctx.lineno}"
        if ctx.is_3d:
            lines.append(
                f"{ctx.indent}auto {kernel_name} = [&](auto {ctx.vars[0]}, auto {ctx.vars[1]}, auto {ctx.vars[2]}) {{"
            )
        elif ctx.is_2d:
            lines.append(f"{ctx.indent}auto {kernel_name} = [&](auto {ctx.vars[0]}, auto {ctx.vars[1]}) {{")
        else:
            lines.append(f"{ctx.indent}auto {kernel_name} = [&](auto {ctx.vars[0]}) {{")
        lines.extend(ctx.body_lines)
        lines.append(f"{ctx.indent}}};")
        launch_arg = kernel_name
    else:
        if ctx.is_3d:
            lines.append(
                f"{ctx.indent}rllm::{ctx.backend_namespace}::launch_kernel_3d("
            )
        elif ctx.is_2d:
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
    parfor_dump_dir: Path | None = None,
    symbol_values: dict[str, str] | None = None,
) -> tuple[str, bool]:
    from transform_source import TransformSourceRewriter

    return TransformSourceRewriter(
        src_text,
        rel_path,
        backend_namespace,
        include_line,
        emit_named_kernels=emit_named_kernels,
        on_emit_loop=on_emit_loop,
        parfor_dump_dir=parfor_dump_dir,
        symbol_values=symbol_values,
    ).transform()

def transform_tree(
    src_dir: Path,
    out_dir: Path,
    backend_namespace: str,
    include_line: str,
    emit_named_kernels: bool = False,
    on_emit_loop: Callable[[LoopContext], None] | None = None,
    parfor_dump_dir: Path | None = None,
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

        parfor_dump_dir.mkdir(parents=True, exist_ok=True)

        transformed, _ = transform_source(
            src_path.read_text(encoding="utf-8"),
            rel_path.as_posix(),
            backend_namespace,
            include_line,
            emit_named_kernels=emit_named_kernels,
            on_emit_loop=on_emit_loop,
            symbol_values=symbol_values,
            parfor_dump_dir=parfor_dump_dir
        )
        out_path.write_text(transformed, encoding="utf-8")
        generated_files.append(out_path)

    return generated_files



def _write_parfor_dump(ctx: "LoopContext", dump_dir: Path,
        symbol_values: dict[str, str] | None = None) -> None:
    """Write a PARFOR block and its OFFLOAD_PARAMETERS to a separate file."""
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: offload_parfor_<basename>_<lineno>.cc
    src_basename = ctx.rel_path.rsplit("/", 1)[-1] if "/" in ctx.rel_path else ctx.rel_path
    name_no_ext = src_basename.rsplit(".", 1)[0] if "." in src_basename else src_basename
    filename = f"offload_parfor_{name_no_ext}_{ctx.lineno}.kernel"
    dump_path = dump_dir / filename

    lines: list[str] = []
    lines.append(f"PROGRAM(\"{ctx.rel_path}:{ctx.lineno}\")")
    lines.append("")

    # Write the original PARFOR invocation line
    if ctx.parfor_invocation:
        #print(f"CHECK: {ctx.parfor_invocation}")
        lines.append(hard_apply_symbol_values(ctx.parfor_invocation, symbol_values))
        lines.append("")

    # Write the OFFLOAD_PARAMETERS block content
    lines.append("PARAMETERS")
    if ctx.offload_param_lines:
        for line in ctx.offload_param_lines:
            #print(f"CHECK: {line}")
            line = hard_apply_symbol_values(line, symbol_values)
            lines.append(line)
        if len(lines) > 0:
            if lines[-1].endswith(","):
                lines[-1] = lines[-1][:-2]
        lines.append("")

    # Write the body lines
    lines.append("BEGIN")
    # Initialize PARFOR loop variables from gl_GlobalInvocationID
    _init_lines_added: list[str] = []
    _DIM_NAMES = ("x", "y", "z")
    if ctx.vars:
        _idx = 0
        for _var_name in ctx.vars:
            if 'queue' in _var_name:
                continue
            _dim = _DIM_NAMES[_idx]
            lines.append(f'const int {_var_name} = int(gl_GlobalInvocationID.{_dim});')
            lines.append(f"if ({_var_name} >= rllm_bound_{_dim}) return;")
            _init_lines_added.extend([
                f'const int {_var_name} = int(gl_GlobalInvocationID.{_dim});',
                f"if ({_var_name} >= rllm_bound_{_dim}) return;",
            ])
            _idx += 1

    # Remove leading init-line duplicates from raw_body_lines to avoid redefinition.
    # Some .kernel files already contain these lines in the body (e.g. from the original
    # C++ source); the dump generator now always emits them, so deduplicate.
    body_idx = 0
    if ctx.raw_body_lines and _init_lines_added:
        for body_line in ctx.raw_body_lines:
            if body_idx < len(_init_lines_added) and body_line.strip() == _init_lines_added[body_idx]:
                body_idx += 1
            else:
                break
        remaining_body = ctx.raw_body_lines[body_idx:]
    else:
        remaining_body = ctx.raw_body_lines or []

    # Triangular guard (if applicable)
    if ctx.triangular_kind:
        parts = []
        row_var = ctx.vars[-2]
        col_var = ctx.vars[-1]
        if ctx.triangular_kind == "upper":
            parts.append(f"{row_var} > {col_var}")
        else:
            parts.append(f"{col_var} > {row_var}")
        guard_expr = " || ".join(parts)
        lines.append(f"if ({guard_expr}) return;")

    if remaining_body:
        for l in remaining_body:
            lines.append(hard_apply_symbol_values(l, symbol_values))
        lines.append("")

    # Extract constexpr declarations for Vulkan compatibility.
    # These are kept in the kernel file (above) and also emitted as #define during GLSL generation.
    if not getattr(ctx, '_constexpr_defines_set', False):
        import re as _re2
        _EXPR_RE2 = _re2.compile(
            r"^\s*constexpr\s+(?P<type>[A-Za-z_][A-Za-z0-9_]*(?:\s*::[A-Za-z_][A-Za-z0-9_]*)?)\s+"
            r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<expr>.+?)\s*;\s*$"
        )
        defines = []
        for _l in ctx.raw_body_lines:
            _m = _EXPR_RE2.match(_l)
            if _m:
                defines.append((_m.group("name"), _m.group("expr")))
        if defines:
            ctx.constexpr_defines = defines
    setattr(ctx, '_constexpr_defines_set', True)

    lines.append("END_PROGRAM")

    dump_path.write_text("\n".join(lines), encoding="utf-8")


PARFOR_LINE_RE = re.compile(r'PROGRAM\("([^:]+):(\d+)"\)')
PARAMETERS_RE = re.compile(r"^PARAMETERS\s*$")
BEGIN_RE = re.compile(r"^BEGIN$")
END_PROGRAM_RE = re.compile(r"^END_PROGRAM$")


def _parse_kernel_file(kernel_path: Path) -> LoopContext | None:
    """Parse a .kernel file and return a LoopContext suitable for _write_parfor_dump."""
    text = kernel_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Parse PROGRAM line for rel_path and lineno
    prog_match = None
    rel_path = "unknown"
    lineno = 0
    for line in lines:
        m = PARFOR_LINE_RE.match(line.strip())
        if m:
            prog_match = m
            rel_path = m.group(1)
            lineno = int(m.group(2))
            break
    if not prog_match:
        return None

    # Parse PARFOR invocation line to extract vars and macro type
    parfor_invocation = None
    vars_list: list[str] = []
    macro_name = None
    triangular_kind = None
    for line in lines:
        parsed = parse_macro_invocation(line)
        if parsed:
            macro_name, args, _ = parsed
            parfor_invocation = line.strip()
            if macro_name in OFFLOAD_1D_PARAM_MACROS and len(args) >= 1:
                vars_list = [args[0]]
            elif macro_name in OFFLOAD_2D_MACROS and len(args) >= 2:
                vars_list = [args[0], args[1]]
            elif macro_name in OFFLOAD_2D_PARAM_MACROS and len(args) >= 2:
                vars_list = [args[0], args[1]]
            elif macro_name in OFFLOAD_3D_PARAM_MACROS and len(args) >= 3:
                vars_list = [args[0], args[1], args[2]]
            elif macro_name in OFFLOAD_3D_TRIANGULAR_PARAM_MACROS and len(args) >= 3:
                vars_list = [args[0], args[1], args[2]]
                triangular_kind = "lower"
            elif macro_name in OFFLOAD_2D_TRIANGULAR_PARAM_MACROS and len(args) >= 2:
                vars_list = [args[0], args[1]]
                triangular_kind = "lower"
            elif macro_name in OFFLOAD_2D_UPPER_TRIANGULAR_PARAM_MACROS and len(args) >= 2:
                vars_list = [args[0], args[1]]
                triangular_kind = "upper"
            break

    if not vars_list:
        return None

    is_2d = len(vars_list) == 2
    is_3d = len(vars_list) == 3

    # Parse PARAMETERS block and BEGIN/END body
    offload_param_lines: list[str] = []
    raw_body_lines: list[str] = []
    in_parameters = False
    in_begin = False

    for line in lines:
        stripped = line.strip()
        if PARAMETERS_RE.match(stripped):
            in_parameters = True
            in_begin = False
            continue
        if BEGIN_RE.match(stripped):
            in_parameters = False
            in_begin = True
            continue
        if END_PROGRAM_RE.match(stripped):
            in_parameters = False
            in_begin = False
            continue
        if in_parameters:
            offload_param_lines.append(line)
        if in_begin:
            raw_body_lines.append(line)

    # Build LoopContext
    ctx = LoopContext(
        indent="",
        backend_namespace="vulkan",
        rel_path=rel_path,
        lineno=lineno,
        is_2d=is_2d,
        is_3d=is_3d,
        vars=vars_list,
        range_expr="grid",
        kernel_guard_expr=None,
        extra_params=None,
        extra_param_types=None,
        offload_param_lines=offload_param_lines,
        emit_named_kernel=False,
        body_lines=[],
        raw_body_lines=raw_body_lines,
        parfor_invocation=parfor_invocation,
        triangular_kind=triangular_kind,
        shared_vars=None,
    )
    return ctx


def write_parfor_dump_from_kernel(kernel_path: Path, dump_dir: Path,
        symbol_values: dict[str, str] | None = None) -> Path:
    """Parse a .kernel file and write a dump file with PARFOR loop variable init lines."""
    ctx = _parse_kernel_file(kernel_path)
    if ctx is None:
        raise ValueError(f"Could not parse PARFOR info from {kernel_path}")
    _write_parfor_dump(ctx, dump_dir, symbol_values)
    return dump_dir / f"offload_parfor_{ctx.rel_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_{ctx.lineno}.kernel"


def write_manifest(manifest: Path, variable_name: str, generated_files: list[Path]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"set({variable_name}"]
    lines.extend(f'    "{path.as_posix()}"' for path in generated_files)
    lines.append(")")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
