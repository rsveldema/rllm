#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys

from offloadize_common import (
    LoopContext,
    apply_symbol_values,
    resolve_symbol_values,
    split_top_level_args,
    transform_tree,
    write_manifest,
)


_SIMD_REDUCTION_PLUS_RE = re.compile(r"^\s*RLLM_OMP_SIMD_REDUCTION_PLUS\s*\([^)]*\)\s*;?\s*$")
_STATIC_CAST_SIZE_T_RE = re.compile(r"static_cast\s*<\s*size_t\s*>\s*\(([^()]*)\)")
_STATIC_CAST_GENERIC_RE = re.compile(r"static_cast\s*<\s*([^>]+?)\s*>\s*\(")
_SIZE_T_WORD_RE = re.compile(r"\bsize_t\b")
_RLMM_FLOAT_WORD_RE = re.compile(r"\brlmm_float(?:_small)?\b")
_STD_SQRT_RE = re.compile(r"\bstd\s*::\s*sqrt\s*\(")
_STD_EXP_RE = re.compile(r"\bstd\s*::\s*exp\s*\(")
_STD_SIN_RE = re.compile(r"\bstd\s*::\s*sin\s*\(")
_STD_COS_RE = re.compile(r"\bstd\s*::\s*cos\s*\(")
_STD_POW_RE = re.compile(r"\bstd\s*::\s*pow\s*\(")
_MATH_MAX_RE = re.compile(r"\bmath\s*::\s*max\s*\(")
_MATH_CLAMP_RE = re.compile(r"\bmath\s*::\s*clamp\s*\(")
_CONSTEXPR_WORD_RE = re.compile(r"\bconstexpr\b")
_CSTYLE_FLOAT_CAST_RE = re.compile(r"\(\s*float\s*\)\s*\(")
_CSTYLE_INT_CAST_RE = re.compile(r"\(\s*int\s*\)\s*\(")
_QUALIFIED_GLSL_SCALAR_RE = re.compile(r"\b(?:[A-Za-z_]\w*::)+(float|int|uint)\b")
_ATOMIC_INC_CALL_RE = re.compile(r"\bATOMIC_INC\s*\((.+)\)\s*;\s*$")
_OVERFLOW_CHECK_ADD_RE = re.compile(r"^\s*OVERFLOW_CHECK_ADD\s*\([^;]*\)\s*;?\s*$")
_COMMA_INDEX3_RE = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\]")
_COMMA_INDEX_RE = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\]")
# Scalar kernel args are passed via push constants, so we only match by-value
# float aliases here and leave reference/pointer forms on the buffer path.
_FLOAT_SCALAR_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*(?:float|rlmm_float(?:_small)?)\s*(?P<ref>[&*])?\s*$"
)
_OFFLOAD_SHARED_MEMORY_RE = re.compile(r"^\s*OFFLOAD_SHARED_MEMORY\s*\((?P<args>.*)\)\s*;?\s*$")


def _sanitize_kernel_line_for_glsl(line: str) -> str:
    if _OVERFLOW_CHECK_ADD_RE.match(line):
        return ""
    # GLSL has no size_t; map common generated C++ forms to uint.
    line = _STATIC_CAST_SIZE_T_RE.sub(r"uint(\1)", line)
    # Replace remaining C++ static_cast<T>(...) with C-style casts before further cleanup.
    line = _STATIC_CAST_GENERIC_RE.sub(r"(\1)(", line)
    line = _SIZE_T_WORD_RE.sub("uint", line)
    # Map project scalar aliases to GLSL scalar type.
    line = _RLMM_FLOAT_WORD_RE.sub("float", line)
    # GLSL primitive types cannot be namespace-qualified (e.g. rllm::float).
    line = _QUALIFIED_GLSL_SCALAR_RE.sub(r"\1", line)
    # GLSL builtins are unqualified (no std:: namespace).
    line = _STD_SQRT_RE.sub("sqrt(", line)
    line = _STD_EXP_RE.sub("exp(", line)
    line = _STD_SIN_RE.sub("sin(", line)
    line = _STD_COS_RE.sub("cos(", line)
    line = _STD_POW_RE.sub("pow(", line)
    line = _MATH_MAX_RE.sub("max(", line)
    line = _MATH_CLAMP_RE.sub("clamp(", line)
    # GLSL does not support constexpr declarations.
    line = _CONSTEXPR_WORD_RE.sub("const", line)
    # Prefer GLSL constructor-style casts over C-style casts.
    line = _CSTYLE_FLOAT_CAST_RE.sub("float(", line)
    line = _CSTYLE_INT_CAST_RE.sub("int(", line)
    # Kernel extraction happens before C preprocessor expansion, so map
    # test-side ATOMIC_INC(expr) directly to GLSL increment form.
    atomic_inc = _ATOMIC_INC_CALL_RE.search(line)
    if atomic_inc:
        target = atomic_inc.group(1).strip()
        line = _ATOMIC_INC_CALL_RE.sub(f"{target}++;", line)
    # GLSL array indexing is nested: a[i][j], not a[i, j].
    while _COMMA_INDEX3_RE.search(line):
        line = _COMMA_INDEX3_RE.sub(r"[\1][\2][\3]", line)
    while _COMMA_INDEX_RE.search(line):
        line = _COMMA_INDEX_RE.sub(r"[\1][\2]", line)
    return line


def _sanitize_offload_param_line_for_glsl(line: str) -> str | None:
    match = _GLSL_CONST_DECL_RE.match(line.strip())
    if match is None:
        return None

    name = match.group("name")
    expr = _sanitize_kernel_line_for_glsl(match.group("expr"))
    return f"const uint {name} = {expr};"


@dataclass
class VulkanKernelSpec:
    rel_path: str
    lineno: int
    is_2d: bool
    is_3d: bool
    vars: list[str]
    kernel_guard_expr: str | None
    extra_params: str | None
    extra_param_types: dict[str, str] | None
    offload_param_lines: list[str] | None
    range_expr: str
    body_lines: list[str]
    shared_vars: dict[str, str] | None = None  # var_name -> type for shared variables


@dataclass
class MatrixViewSpec:
    name: str
    glsl_scalar: str
    rows: int
    cols: int
    is_const: bool
    flat_len: int
    levels: int | None = None
    dynamic_rows_name: str | None = None
    dynamic_cols_name: str | None = None
    separate_level_bindings: bool = False


@dataclass
class BufferViewSpec:
    name: str
    glsl_scalar: str
    is_const: bool


@dataclass
class ScalarParamSpec:
    name: str
    glsl_type: str


@dataclass(frozen=True)
class VulkanConfig:
    workgroup_size_x: int = 1
    workgroup_size_y: int = 1
    workgroup_size_z: int = 1


_MATRIX_TYPE_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*(?:fixed_size_matrix|flexible_size_matrix|flexible_rows_matrix|flexible_cols_matrix|flexible_rows_cols_matrix)\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<rows>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*(?P<ref>[&*])?\s*$"
)
_MATRIX_3D_TYPE_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*(?:flexible_rows_cols_levels_matrix|fixed_size_levels_rows_cols_matrix)\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<levels>[^,]+)\s*,\s*(?P<rows>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*(?P<ref>[&*])?\s*$"
)
_OBJ_VECTOR_MATRIX_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*fixed_size_obj_vector\s*<\s*(?:[A-Za-z_]\w*::)*(?:fixed_size_matrix|flexible_size_matrix)\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<rows>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*,\s*(?P<levels>[^>]+)\s*>\s*(?P<ref>[&*])?\s*$"
)
_STD_VECTOR_ATOMIC_INT_RE = re.compile(
    r"^(?P<const>const\s+)?std::vector\s*<\s*std::atomic\s*<\s*int\s*>\s*>\s*(?P<ref>[&*])?\s*$"
)
_STD_VECTOR_INT_RE = re.compile(
    r"^(?P<const>const\s+)?std::vector\s*<\s*int\s*>\s*(?P<ref>[&*])?\s*$"
)
_INPUT_LINE_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*InputLine\s*(?P<ref>[&*])?\s*$"
)
_FIXED_SIZE_VECTOR_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?P<scalar>[^,]+)\s*,\s*[^>]+>\s*(?P<ref>[&*])?\s*$"
)
_NESTED_FIXED_SIZE_VECTOR_MATRIX_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*,\s*(?P<rows>[^>]+)\s*>\s*(?P<ref>[&*])?\s*$"
)
_DEVICE_POINTER_INT_RE = re.compile(
    r"^(?P<const>const\s+)?DevicePointer\s*<\s*int\s*>\s*(?P<ref>[&*])?\s*$"
)
_GLSL_CONST_DECL_RE = re.compile(
    r"^\s*(?:constexpr|const)\s+(?:size_t|int|unsigned|auto)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<expr>.+?)\s*;\s*$"
)


def _normalize_enum_type_name(raw: str) -> str:
    token = raw.strip()
    if "::" in token:
        token = token.split("::")[-1]
    return token


def _map_cpp_buffer_scalar_to_glsl(raw_scalar: str) -> str | None:
    token = raw_scalar.strip()
    if "::" in token:
        token = token.split("::")[-1]

    if token in {"int", "int32_t"}:
        return "int"
    if token in {"float", "rlmm_float", "rlmm_float_small"}:
        return "float"
    return None


def _resolve_generated_token_id_max() -> int | None:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "build_release/generated/tokenizer_map.hpp",
        repo_root / "build/generated/tokenizer_map.hpp",
    ]
    pattern = re.compile(r"\bMAX\s*=\s*(\d+)\s*,")
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1))
    return None


def _resolve_enum_max(raw_enum_type: str, symbol_values: dict[str, str]) -> int | None:
    enum_type = _normalize_enum_type_name(raw_enum_type)
    key = f"{enum_type}::MAX"
    raw = symbol_values.get(key)
    if raw is None:
        if enum_type == "TokenID":
            return _resolve_generated_token_id_max()
        return None
    try:
        return int(raw)
    except ValueError:
        if enum_type == "TokenID":
            return _resolve_generated_token_id_max()
        return None


def _map_cpp_extra_param_to_vulkan(
    name: str,
    cpp_type: str,
    symbol_values: dict[str, str],
) -> tuple[str, str, MatrixViewSpec | None, BufferViewSpec | None, ScalarParamSpec | None]:
    # Returns: (kernel_param_declaration, main_local_setup_declaration, optional_matrix_view_spec)
    t = cpp_type.strip()
    core_type = re.sub(r"^const\s+", "", t)
    core_type = re.sub(r"\s*[&*]\s*$", "", core_type).strip()
    matrix_3d_match = _MATRIX_3D_TYPE_RE.match(t)
    if matrix_3d_match:
        glsl_scalar = _map_cpp_buffer_scalar_to_glsl(matrix_3d_match.group("scalar"))
        levels_max = _resolve_enum_max(matrix_3d_match.group("levels"), symbol_values)
        rows_max = _resolve_enum_max(matrix_3d_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(matrix_3d_match.group("cols"), symbol_values)
        if glsl_scalar is not None and levels_max is not None and rows_max is not None and cols_max is not None:
            is_const = matrix_3d_match.group("const") is not None
            flat_len = levels_max * rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
                glsl_scalar=glsl_scalar,
                rows=rows_max,
                cols=cols_max,
                is_const=is_const,
                flat_len=flat_len,
                levels=levels_max,
            ), None, None

    obj_vector_matrix_match = _OBJ_VECTOR_MATRIX_RE.match(t)
    if obj_vector_matrix_match:
        glsl_scalar = _map_cpp_buffer_scalar_to_glsl(obj_vector_matrix_match.group("scalar"))
        levels_max = _resolve_enum_max(obj_vector_matrix_match.group("levels"), symbol_values)
        rows_max = _resolve_enum_max(obj_vector_matrix_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(obj_vector_matrix_match.group("cols"), symbol_values)
        if glsl_scalar is not None and levels_max is not None and rows_max is not None and cols_max is not None:
            is_const = obj_vector_matrix_match.group("const") is not None
            return "", "", MatrixViewSpec(
                name=name,
                glsl_scalar=glsl_scalar,
                rows=rows_max,
                cols=cols_max,
                is_const=is_const,
                flat_len=rows_max * cols_max,
                levels=levels_max,
                separate_level_bindings=True,
            ), None, None

    matrix_match = _MATRIX_TYPE_RE.match(t)
    if matrix_match:
        glsl_scalar = _map_cpp_buffer_scalar_to_glsl(matrix_match.group("scalar"))
        rows_max = _resolve_enum_max(matrix_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(matrix_match.group("cols"), symbol_values)
        if glsl_scalar is not None and rows_max is not None and cols_max is not None:
            is_const = matrix_match.group("const") is not None
            # Matrix-like tensors are exposed as SSBO bindings, not copied into function-local arrays.
            flat_len = rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
                glsl_scalar=glsl_scalar,
                rows=rows_max,
                cols=cols_max,
                is_const=is_const,
                flat_len=flat_len,
            ), None, None

    vector_atomic_match = _STD_VECTOR_ATOMIC_INT_RE.match(t)
    if vector_atomic_match:
        is_const = vector_atomic_match.group("const") is not None
        return "", "", None, BufferViewSpec(name=name, glsl_scalar="int", is_const=is_const), None

    vector_int_match = _STD_VECTOR_INT_RE.match(t)
    if vector_int_match:
        is_const = vector_int_match.group("const") is not None
        return "", "", None, BufferViewSpec(name=name, glsl_scalar="int", is_const=is_const), None

    input_line_match = _INPUT_LINE_RE.match(t)
    if input_line_match or core_type.endswith("InputLine"):
        is_const = input_line_match.group("const") is not None if input_line_match else t.startswith("const ")
        return "", "", None, BufferViewSpec(name=name, glsl_scalar="int", is_const=is_const), None

    nested_fixed_size_vector_matrix_match = _NESTED_FIXED_SIZE_VECTOR_MATRIX_RE.match(t)
    if nested_fixed_size_vector_matrix_match:
        glsl_scalar = _map_cpp_buffer_scalar_to_glsl(nested_fixed_size_vector_matrix_match.group("scalar"))
        rows_max = _resolve_enum_max(nested_fixed_size_vector_matrix_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(nested_fixed_size_vector_matrix_match.group("cols"), symbol_values)
        if glsl_scalar is not None and rows_max is not None and cols_max is not None:
            is_const = nested_fixed_size_vector_matrix_match.group("const") is not None
            flat_len = rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
                glsl_scalar=glsl_scalar,
                rows=rows_max,
                cols=cols_max,
                is_const=is_const,
                flat_len=flat_len,
            ), None, None

    fixed_size_vector_match = _FIXED_SIZE_VECTOR_RE.match(t)
    if fixed_size_vector_match:
        glsl_scalar = _map_cpp_buffer_scalar_to_glsl(fixed_size_vector_match.group("scalar"))
        if glsl_scalar is not None:
            is_const = fixed_size_vector_match.group("const") is not None
            return "", "", None, BufferViewSpec(name=name, glsl_scalar=glsl_scalar, is_const=is_const), None

    nested_core_match = re.match(
        r"^(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*,\s*(?P<rows>[^>]+)\s*>$",
        core_type,
    )
    if nested_core_match:
        glsl_scalar = _map_cpp_buffer_scalar_to_glsl(nested_core_match.group("scalar"))
        rows_max = _resolve_enum_max(nested_core_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(nested_core_match.group("cols"), symbol_values)
        if glsl_scalar is not None and rows_max is not None and cols_max is not None:
            flat_len = rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
                glsl_scalar=glsl_scalar,
                rows=rows_max,
                cols=cols_max,
                is_const=t.startswith("const "),
                flat_len=flat_len,
            ), None, None

    device_pointer_int_match = _DEVICE_POINTER_INT_RE.match(t)
    if device_pointer_int_match:
        is_const = device_pointer_int_match.group("const") is not None
        return "", "", None, BufferViewSpec(name=name, glsl_scalar="int", is_const=is_const), None

    float_scalar_match = _FLOAT_SCALAR_RE.match(t)
    if float_scalar_match and float_scalar_match.group("ref") is None:
        return f"float {name}", f"float {name} = rllm_push.{name};", None, None, ScalarParamSpec(name=name, glsl_type="float")

    # Keep unknown scalar-like params off the SSBO path. The rewriter already
    # normalizes enum-ish and size-like expressions to integer GLSL usage.
    return f"int {name}", f"int {name} = rllm_push.{name};", None, None, ScalarParamSpec(name=name, glsl_type="int")


def _rewrite_matrix_view_indexing(line: str, specs: dict[str, MatrixViewSpec]) -> str:
    out = line

    def rewrite_reads(text: str) -> str:
        rewritten = text
        for read_name, read_spec in specs.items():
            if read_spec.levels is not None and read_spec.separate_level_bindings:
                read_pattern = re.compile(
                    rf"\b{re.escape(read_name)}\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]"
                )
                rewritten = read_pattern.sub(
                    lambda match: (
                        f"rllm_load_{read_name}("
                        f"{match.group(1).strip()}, {match.group(2).strip()}, {match.group(3).strip()})"
                    ),
                    rewritten,
                )
            elif read_spec.levels is not None:
                read_pattern = re.compile(
                    rf"\b{re.escape(read_name)}\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]"
                )

                def repl3(match: re.Match[str]) -> str:
                    level = match.group(1).strip()
                    row = match.group(2).strip()
                    col = match.group(3).strip()
                    row_stride = read_spec.dynamic_rows_name or str(read_spec.rows)
                    col_stride = read_spec.dynamic_cols_name or str(read_spec.cols)
                    return f"{read_name}[((({level}) * {row_stride}) + ({row})) * {col_stride} + ({col})]"

                rewritten = read_pattern.sub(repl3, rewritten)
            else:
                read_pattern = re.compile(
                    rf"\b{re.escape(read_name)}\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]"
                )
                rewritten = read_pattern.sub(
                    lambda match: (
                        f"{read_name}[(({match.group(1).strip()}) * {read_spec.cols}) + ({match.group(2).strip()})]"
                    ),
                    rewritten,
                )
        return rewritten

    for name, spec in specs.items():
        if spec.levels is not None and spec.separate_level_bindings:
            indexed = (
                rf"\b{re.escape(name)}\s*\[\s*(?P<level>[^\[\]]+?)\s*\]\s*"
                rf"\[\s*(?P<row>[^\[\]]+?)\s*\]\s*\[\s*(?P<col>[^\[\]]+?)\s*\]"
            )
            assignment = re.compile(
                rf"^(?P<prefix>.*?)({indexed})\s*(?P<op>[+\-*/]?=)\s*(?P<rhs>.+);\s*$"
            )
            match = assignment.match(line)
            if match is not None:
                level = match.group("level").strip()
                row = match.group("row").strip()
                col = match.group("col").strip()
                rhs = rewrite_reads(match.group("rhs").strip())
                op = match.group("op")
                if op != "=":
                    rhs = f"rllm_load_{name}({level}, {row}, {col}) {op[0]} ({rhs})"
                return f"{match.group('prefix')}rllm_store_{name}({level}, {row}, {col}, {rhs});"

    for name, spec in specs.items():
        if spec.levels is not None and spec.separate_level_bindings:
            out = rewrite_reads(out)
            continue

        if spec.levels is not None:
            pattern = re.compile(
                rf"\b{re.escape(name)}\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]"
            )

            def repl3(match: re.Match[str]) -> str:
                level = match.group(1).strip()
                row = match.group(2).strip()
                col = match.group(3).strip()
                row_stride = spec.dynamic_rows_name or str(spec.rows)
                col_stride = spec.dynamic_cols_name or str(spec.cols)
                return f"{name}[((({level}) * {row_stride}) + ({row})) * {col_stride} + ({col})]"

            out = pattern.sub(repl3, out)
            continue

        pattern = re.compile(
            rf"\b{re.escape(name)}\s*\[\s*([^\[\]]+?)\s*\]\s*\[\s*([^\[\]]+?)\s*\]"
        )

        def repl(match: re.Match[str]) -> str:
            row = match.group(1).strip()
            col = match.group(2).strip()
            return f"{name}[(({row}) * {spec.cols}) + ({col})]"

        out = pattern.sub(repl, out)
    return out


def _parse_extra_param_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    if len(text) >= 2 and text[0] in "[{(" and text[-1] in "]})":
        text = text[1:-1].strip()
    if not text:
        return []

    names: list[str] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        m = re.search(r"([A-Za-z_]\w*)\s*$", token)
        if m:
            names.append(m.group(1))
    return names


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


def _kernel_glsl_file_path(kernel_root: Path, spec: VulkanKernelSpec) -> Path:
    rel = Path(spec.rel_path)
    stem = _sanitize_name_component(rel.stem)
    return kernel_root / rel.parent / f"{stem}.L{spec.lineno}.glsl"


def _kernel_header_file_path(kernel_root: Path, spec: VulkanKernelSpec) -> Path:
    rel = Path(spec.rel_path)
    stem = _sanitize_name_component(rel.stem)
    return kernel_root / rel.parent / f"{stem}.L{spec.lineno}.hpp"


def _spirv_file_path(kernel_root: Path, spec: VulkanKernelSpec) -> Path:
    rel = Path(spec.rel_path)
    stem = _sanitize_name_component(rel.stem)
    return kernel_root / rel.parent / f"{stem}.L{spec.lineno}.spv"


def _load_vulkan_config(config_path: Path) -> VulkanConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    workgroup = raw.get("workgroup_size", {})

    x = int(workgroup.get("x", 1))
    y = int(workgroup.get("y", 1))
    z = int(workgroup.get("z", 1))
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError(
            f"Invalid Vulkan workgroup size in {config_path}: expected positive integers, got x={x}, y={y}, z={z}"
        )

    return VulkanConfig(workgroup_size_x=x, workgroup_size_y=y, workgroup_size_z=z)


def _sanitize_cpp_type_to_glsl(cpp_type: str) -> str:
    """Convert a C++ type to GLSL type for shared variables."""
    glsl_type = _sanitize_kernel_line_for_glsl(cpp_type)
    # Convert C++ floating point types to float
    glsl_type = re.sub(r"\brlmm_float(?:_small)?\b", "float", glsl_type)
    glsl_type = re.sub(r"\bsize_t\b", "uint", glsl_type)
    # Remove qualifiers that don't apply in shared context
    glsl_type = re.sub(r"\bconst\b", "", glsl_type).strip()
    return glsl_type


def _requires_atomic_float_extensions(lines: list[str]) -> bool:
    joined = "\n".join(lines)
    return "atomicAdd(" in joined or "atomicMax(" in joined or "atomicMin(" in joined


def _extract_offload_shared_memory_decls(
    body_lines: list[str],
    default_count: int,
    symbol_values: dict[str, str],
) -> tuple[list[str], list[str]]:
    shared_decls: list[str] = []
    filtered_body_lines: list[str] = []

    for line in body_lines:
        match = _OFFLOAD_SHARED_MEMORY_RE.match(line)
        if match is None:
            filtered_body_lines.append(line)
            continue

        args = split_top_level_args(match.group("args"))
        if len(args) not in {2, 3}:
            raise ValueError(
                "OFFLOAD_SHARED_MEMORY expects (type, name) or (type, name, count); "
                f"got {len(args)} argument(s): {line.strip()}"
            )

        count = str(default_count)
        if len(args) == 2:
            type_arg = args[0].strip()
            name = args[1].strip()
        elif re.match(r"^[A-Za-z_]\w*$", args[2].strip()):
            count = _sanitize_kernel_line_for_glsl(apply_symbol_values(args[0].strip(), symbol_values))
            type_arg = args[1].strip()
            name = args[2].strip()
        else:
            type_arg = args[0].strip()
            name = args[1].strip()
            count = _sanitize_kernel_line_for_glsl(apply_symbol_values(args[2].strip(), symbol_values))
        if "gl_" in count and "gl_WorkGroupSize" not in count:
            count = str(default_count)

        glsl_type = _sanitize_cpp_type_to_glsl(type_arg)

        if not re.match(r"^[A-Za-z_]\w*$", name):
            raise ValueError(f"OFFLOAD_SHARED_MEMORY has invalid shared variable name '{name}' in: {line.strip()}")

        shared_decls.append(f"shared {glsl_type} {name}[{count}];")

    return shared_decls, filtered_body_lines


def _render_kernel_stub(spec: VulkanKernelSpec, symbol_values: dict[str, str], config: VulkanConfig) -> str:
    local_size_x = config.workgroup_size_x
    local_size_y = config.workgroup_size_y if (spec.is_2d or spec.is_3d) else 1
    local_size_z = config.workgroup_size_z if spec.is_3d else (config.workgroup_size_z if spec.is_2d else 1)

    const_preamble: list[str] = []
    for line in spec.offload_param_lines or []:
        rendered = _sanitize_offload_param_line_for_glsl(line)
        if rendered is not None:
            const_preamble.append(rendered)
    const_block = ("\n".join(const_preamble) + "\n\n") if const_preamble else ""

    filtered_body_lines = [line for line in spec.body_lines if not _SIMD_REDUCTION_PLUS_RE.match(line)]
    if spec.is_2d and any("subgroup" in line or "gl_Subgroup" in line for line in filtered_body_lines):
        local_size_y = 1
        local_size_z = 1
    default_shared_count = local_size_x * local_size_y * local_size_z
    offload_shared_decls, filtered_body_lines = _extract_offload_shared_memory_decls(
        filtered_body_lines,
        default_shared_count,
        symbol_values,
    )
    filtered_body_lines = [_sanitize_kernel_line_for_glsl(line) for line in filtered_body_lines]
    matrix_view_specs: dict[str, MatrixViewSpec] = {}
    buffer_view_specs: dict[str, BufferViewSpec] = {}
    scalar_param_specs: list[ScalarParamSpec] = []
    body_text = "\n".join(
        f"    {_rewrite_matrix_view_indexing(line.lstrip(), matrix_view_specs)}" if line.strip() else ""
        for line in filtered_body_lines
    )
    extra_param_names = _parse_extra_param_names(spec.extra_params)
    param_names = list(spec.vars) + extra_param_names

    typed_params: list[str] = []
    for var in spec.vars:
        typed_params.append(f"int {var}")
    extra_setup_decls: list[str] = []
    ssbo_decls: list[str] = []
    matrix_helper_decls: list[str] = []
    ssbo_binding = 0
    for name in extra_param_names:
        if spec.extra_param_types and name in spec.extra_param_types:
            mapped_param, mapped_setup, matrix_view_spec, buffer_view_spec, scalar_param_spec = _map_cpp_extra_param_to_vulkan(
                name,
                spec.extra_param_types[name],
                symbol_values,
            )
            if matrix_view_spec is not None:
                matrix_view_specs[name] = matrix_view_spec
                if matrix_view_spec.separate_level_bindings:
                    assert matrix_view_spec.levels is not None
                    readonly = "readonly " if matrix_view_spec.is_const else ""
                    array_extent = "" if matrix_view_spec.flat_len >= (1 << 31) else str(matrix_view_spec.flat_len)
                    for level in range(matrix_view_spec.levels):
                        level_name = f"{name}_{level}"
                        block_name = f"RllmBuffer_{level_name}"
                        ssbo_decls.append(
                            f"layout(std430, set = 0, binding = {ssbo_binding}) {readonly}buffer {block_name} {{ {matrix_view_spec.glsl_scalar} {level_name}[{array_extent}]; }};"
                        )
                        ssbo_binding += 1

                    dynamic_cols = (
                        f"rllm_push.{name}_cols"
                        if f"{name}_cols" in extra_param_names
                        else "rllm_push.active_seq_len"
                        if "active_seq_len" in extra_param_names
                        else "rllm_push.seq_len"
                        if "seq_len" in extra_param_names
                        else "rllm_push.rllm_bound_y"
                    )
                    index_expr = f"(row * {dynamic_cols}) + col"
                    load_cases = "\n".join(
                        f"        case {level}: return {name}_{level}[idx];" for level in range(matrix_view_spec.levels)
                    )
                    matrix_helper_decls.append(
                        f"{matrix_view_spec.glsl_scalar} rllm_load_{name}(int level, int row, int col)\n"
                        "{\n"
                        f"    int idx = {index_expr};\n"
                        "    switch (level)\n"
                        "    {\n"
                        f"{load_cases}\n"
                        "        default: return 0;\n"
                        "    }\n"
                        "}"
                    )
                    if not matrix_view_spec.is_const:
                        store_cases = "\n".join(
                            f"        case {level}: {name}_{level}[idx] = value; return;"
                            for level in range(matrix_view_spec.levels)
                        )
                        matrix_helper_decls.append(
                            f"void rllm_store_{name}(int level, int row, int col, {matrix_view_spec.glsl_scalar} value)\n"
                            "{\n"
                            f"    int idx = {index_expr};\n"
                            "    switch (level)\n"
                            "    {\n"
                            f"{store_cases}\n"
                            "        default: return;\n"
                            "    }\n"
                            "}"
                        )
                else:
                    readonly = "readonly " if matrix_view_spec.is_const else ""
                    block_name = f"RllmBuffer_{name}"
                    array_extent = "" if matrix_view_spec.flat_len >= (1 << 31) else str(matrix_view_spec.flat_len)
                    ssbo_decls.append(
                        f"layout(std430, set = 0, binding = {ssbo_binding}) {readonly}buffer {block_name} {{ {matrix_view_spec.glsl_scalar} {name}[{array_extent}]; }};"
                    )
                    ssbo_binding += 1
            elif buffer_view_spec is not None:
                buffer_view_specs[name] = buffer_view_spec
                readonly = "readonly " if buffer_view_spec.is_const else ""
                block_name = f"RllmBuffer_{name}"
                ssbo_decls.append(
                    f"layout(std430, set = 0, binding = {ssbo_binding}) {readonly}buffer {block_name} {{ {buffer_view_spec.glsl_scalar} {name}[]; }};"
                )
                ssbo_binding += 1
            elif scalar_param_spec is not None:
                # Scalars are emitted as function parameters for readability but
                # sourced from a single push-constant block in main().
                scalar_param_specs.append(scalar_param_spec)
                typed_params.append(mapped_param)
                extra_setup_decls.append(mapped_setup)
            else:
                typed_params.append(mapped_param)
                extra_setup_decls.append(mapped_setup)
        else:
            typed_params.append(f"int {name}")
            extra_setup_decls.append(f"int {name} = 0;")
    param_list = ", ".join(typed_params)

    for matrix_name, matrix_view_spec in matrix_view_specs.items():
        rows_name = f"{matrix_name}_rows"
        cols_name = f"{matrix_name}_cols"
        if rows_name in extra_param_names:
            matrix_view_spec.dynamic_rows_name = rows_name
        if cols_name in extra_param_names:
            matrix_view_spec.dynamic_cols_name = cols_name

    # Re-render body once matrix view specs are known.
    body_text = "\n".join(
        f"    {_rewrite_matrix_view_indexing(line.lstrip(), matrix_view_specs)}" if line.strip() else ""
        for line in filtered_body_lines
    )
    if spec.is_3d and len(spec.vars) >= 3:
        arg_setup = (
            f"    int {spec.vars[0]} = int(gl_GlobalInvocationID.z);\n"
            f"    int {spec.vars[1]} = int(gl_GlobalInvocationID.x);\n"
            f"    int {spec.vars[2]} = int(gl_GlobalInvocationID.y);"
        )
        call_arg_names = [spec.vars[0], spec.vars[1], spec.vars[2]]
    elif spec.is_2d and len(spec.vars) >= 2:
        if spec.kernel_guard_expr is not None:
            arg_setup = (
                f"    int {spec.vars[0]} = int(gl_GlobalInvocationID.y);\n"
                f"    int {spec.vars[1]} = int(gl_GlobalInvocationID.x);"
            )
        else:
            arg_setup = (
                f"    int {spec.vars[0]} = int(gl_GlobalInvocationID.x);\n"
                f"    int {spec.vars[1]} = int(gl_GlobalInvocationID.y);"
            )
        call_arg_names = [spec.vars[0], spec.vars[1]]
    elif spec.vars:
        arg_setup = f"    int {spec.vars[0]} = int(gl_GlobalInvocationID.x);"
        call_arg_names = [spec.vars[0]]
    else:
        arg_setup = ""
        call_arg_names = []

    extra_arg_setup_lines = [f"    {decl}" for decl in extra_setup_decls]
    if extra_arg_setup_lines:
        if arg_setup:
            arg_setup = arg_setup + "\n" + "\n".join(extra_arg_setup_lines)
        else:
            arg_setup = "\n".join(extra_arg_setup_lines)
    guard_terms: list[str] = []
    if not spec.shared_vars:
        if spec.is_3d and len(spec.vars) >= 3:
            guard_terms.append(
                f"{spec.vars[1]} >= rllm_push.rllm_bound_x || "
                f"{spec.vars[2]} >= rllm_push.rllm_bound_y || "
                f"{spec.vars[0]} >= rllm_push.rllm_bound_z"
            )
        elif spec.is_2d and len(spec.vars) >= 2:
            guard_terms.append(f"{spec.vars[0]} >= rllm_push.rllm_bound_x || {spec.vars[1]} >= rllm_push.rllm_bound_y")
        elif spec.vars:
            guard_terms.append(f"{spec.vars[0]} >= rllm_push.rllm_bound_x")
    if spec.kernel_guard_expr:
        guard_terms.append(_sanitize_kernel_line_for_glsl(spec.kernel_guard_expr))
    guard_block = ""
    if guard_terms:
        guard_expr = " || ".join(f"({term})" for term in guard_terms)
        guard_block = f"    if ({guard_expr}) {{\n        return;\n    }}\n"
    call_arg_names.extend(
        name for name in extra_param_names if name not in matrix_view_specs and name not in buffer_view_specs
    )
    call_args = ", ".join(call_arg_names)
    ssbo_block = "\n".join(ssbo_decls)
    if ssbo_block:
        ssbo_block = ssbo_block + "\n\n"
    matrix_helper_block = "\n\n".join(matrix_helper_decls)
    if matrix_helper_block:
        matrix_helper_block = matrix_helper_block + "\n\n"
    
    # Generate shared variable declarations. GLSL requires workgroup storage to
    # live at global scope, not inside function scope.
    shared_decls: list[str] = list(offload_shared_decls)
    if spec.shared_vars:
        for var_name, var_type in spec.shared_vars.items():
            glsl_type = _sanitize_cpp_type_to_glsl(var_type)
            shared_decls.append(f"shared {glsl_type} {var_name}[{default_shared_count}];")
    shared_block = "\n".join(shared_decls)
    if shared_block:
        shared_block = shared_block + "\n\n"

    extension_block = ""
    if _requires_atomic_float_extensions(filtered_body_lines):
        extension_block = (
            "#extension GL_EXT_shader_atomic_float : require\n"
            "#extension GL_EXT_shader_atomic_float2 : require\n\n"
        )
    
    push_constant_fields = [
        "    int rllm_bound_x;",
        "    int rllm_bound_y;",
    ]
    if spec.is_3d:
        push_constant_fields.append("    int rllm_bound_z;")
    if scalar_param_specs:
        # Vulkan gives us a compact scalar-argument path without inventing a
        # one-off SSBO for values like learning rates.
        push_constant_fields.extend(f"    {scalar.glsl_type} {scalar.name};" for scalar in scalar_param_specs)
    fields = "\n".join(push_constant_fields)
    push_constant_block = f"layout(push_constant) uniform RllmPushConstants {{\n{fields}\n}} rllm_push;\n\n"
    return (
        "#version 460\n"
        "#define USE_VULKAN_OFFLOAD 1\n"
        "#extension GL_KHR_shader_subgroup_arithmetic : require\n"
        "#extension GL_KHR_shader_subgroup_clustered : require\n"
        "\n"
        f"{extension_block}"
        f"layout(local_size_x = {local_size_x}, local_size_y = {local_size_y}, local_size_z = {local_size_z}) in;\n"
        "\n"
        f"{const_block}"
        f"{push_constant_block}"
        f"{ssbo_block}"
        f"{matrix_helper_block}"
        f"{shared_block}"
        f"void rllm_kernel_body({param_list})\n"
        "{\n"
        f"{body_text}\n"
        "}\n"
        "\n"
        "void main()\n"
        "{\n"
        f"{arg_setup}\n"
        f"{guard_block}"
        f"    rllm_kernel_body({call_args});\n"
        "}\n"
    )


def _write_kernel_stubs(
    kernel_specs: list[VulkanKernelSpec],
    kernel_root: Path,
    symbol_values: dict[str, str],
    config: VulkanConfig,
) -> list[Path]:
    generated_kernel_files: list[Path] = []
    for spec in kernel_specs:
        kernel_path = _kernel_file_path(kernel_root, spec)
        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        kernel_path.write_text(_render_kernel_stub(spec, symbol_values, config), encoding="utf-8")
        generated_kernel_files.append(kernel_path)
    return generated_kernel_files


def _compile_to_spirv(compiler: str, input_path: Path, output_path: Path) -> None:
    if Path(compiler).name == "glslangValidator":
        cmd = [compiler, "-V", "--target-env", "vulkan1.1", input_path.as_posix(), "-o", output_path.as_posix()]
    else:
        cmd = [compiler, "-fshader-stage=compute", "--target-env=vulkan1.1", input_path.as_posix(), "-o", output_path.as_posix()]
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


def _parfor_dump_path(parfor_dump_dir: Path, spec: VulkanKernelSpec) -> Path:
    src_basename = Path(spec.rel_path).name
    name_no_ext = src_basename.rsplit(".", 1)[0] if "." in src_basename else src_basename
    return parfor_dump_dir / f"offload_parfor_{name_no_ext}_{spec.lineno}.kernel"


def _kernel_compiler_python() -> str:
    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    return venv_python.as_posix() if venv_python.exists() else sys.executable


def _generate_kernel_compiler_artifacts(
    compiler: str | None,
    kernel_specs: list[VulkanKernelSpec],
    kernel_root: Path,
    parfor_dump_dir: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    repo_root = Path(__file__).resolve().parent
    compile_py = repo_root / "kernel_compiler" / "codegen" / "compile.py"
    generated_glsl: list[Path] = []
    generated_headers: list[Path] = []
    generated_spirv: list[Path] = []

    for spec in kernel_specs:
        dump = _parfor_dump_path(parfor_dump_dir, spec)
        glsl = _kernel_glsl_file_path(kernel_root, spec)
        header = _kernel_header_file_path(kernel_root, spec)
        spv = _spirv_file_path(kernel_root, spec)
        rel_spv = spv.relative_to(kernel_root).as_posix()
        glsl.parent.mkdir(parents=True, exist_ok=True)
        header.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            _kernel_compiler_python(),
            compile_py.as_posix(),
            "--vulkan",
            glsl.as_posix(),
            "--rllm-dispatch-stub",
            header.as_posix(),
            "--rllm-spv-path",
            rel_spv,
            dump.as_posix(),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"kernel_compiler failed for {dump}:\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        generated_glsl.append(glsl)
        generated_headers.append(header)
        if compiler:
            _compile_to_spirv(compiler, glsl, spv)
            generated_spirv.append(spv)

    umbrella = kernel_root / "rllm_vulkan_kernels.hpp"
    include_lines = ["#pragma once"]
    include_lines.extend(
        f'#include "{header.relative_to(kernel_root).as_posix()}"'
        for header in generated_headers
    )
    umbrella.write_text("\n".join(include_lines) + "\n", encoding="utf-8")
    generated_headers.append(umbrella)
    return generated_glsl, generated_headers, generated_spirv


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
    config = _load_vulkan_config(Path(__file__).resolve().with_name("vulkan-config.json"))
    kernel_specs: list[VulkanKernelSpec] = []
    symbol_values = resolve_symbol_values(args.enum_value_tool)

    # Do not clear kernel_root here: src/ and tests/ vulkanize targets run in
    # parallel and share this directory. Deleting it introduces a race where one
    # invocation removes files the other is about to compile.

    def collect_kernel(ctx: LoopContext) -> None:
        kernel_specs.append(
            VulkanKernelSpec(
                rel_path=ctx.rel_path,
                lineno=ctx.lineno,
                is_2d=ctx.is_2d,
                is_3d=ctx.is_3d,
                vars=list(ctx.vars),
                kernel_guard_expr=ctx.kernel_guard_expr,
                extra_params=ctx.extra_params,
                extra_param_types=ctx.extra_param_types,
                offload_param_lines=ctx.offload_param_lines,
                range_expr=ctx.range_expr,
                body_lines=list(ctx.body_lines),
                shared_vars=ctx.shared_vars,
            )
        )

    generated_files = transform_tree(
        src_dir,
        out_dir,
        backend_namespace="vulkan",
        include_line="#include <rllm_vulkan_kernels.hpp>",
        emit_named_kernels=True,
        on_emit_loop=collect_kernel,
        symbol_values=symbol_values,
        parfor_dump_dir=out_dir / "parfor_dumps",
    )
    parfor_dump_dir = out_dir / "parfor_dumps"
    generated_kernels, generated_headers, generated_spirv = _generate_kernel_compiler_artifacts(
        args.shader_compiler,
        kernel_specs,
        kernel_root,
        parfor_dump_dir,
    )

    if args.manifest:
        manifest = Path(args.manifest).resolve()
        write_manifest(manifest, "RLLM_VULKANIZED_SOURCES", generated_files)
        kernel_manifest = manifest.with_name("vulkan_kernel_sources.cmake")
        write_manifest(kernel_manifest, "RLLM_VULKAN_KERNEL_SOURCES", generated_kernels + generated_headers)
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
