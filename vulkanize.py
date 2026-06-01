#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess

from offloadize_common import LoopContext, resolve_symbol_values, transform_tree, write_manifest


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
_ATOMIC_INC_CALL_RE = re.compile(r"\bATOMIC_INC\s*\((.+)\)\s*;\s*$")
_OVERFLOW_CHECK_ADD_RE = re.compile(r"^\s*OVERFLOW_CHECK_ADD\s*\([^;]*\)\s*;?\s*$")
_COMMA_INDEX_RE = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\]")
# Scalar kernel args are passed via push constants, so we only match by-value
# float aliases here and leave reference/pointer forms on the buffer path.
_FLOAT_SCALAR_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*(?:float|rlmm_float(?:_small)?)\s*(?P<ref>[&*])?\s*$"
)


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
    vars: list[str]
    kernel_guard_expr: str | None
    extra_params: str | None
    extra_param_types: dict[str, str] | None
    offload_param_lines: list[str] | None
    range_expr: str
    body_lines: list[str]


@dataclass
class MatrixViewSpec:
    name: str
    rows: int
    cols: int
    is_const: bool
    flat_len: int


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
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*(?:fixed_size_matrix|flexible_rows_matrix|flexible_cols_matrix|flexible_rows_cols_matrix)\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<rows>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*(?P<ref>[&*])?\s*$"
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
_FIXED_SIZE_VECTOR_INT_RE = re.compile(
    r"^(?P<const>const\s+)?(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*int\s*,\s*[^>]+>\s*(?P<ref>[&*])?\s*$"
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
    matrix_match = _MATRIX_TYPE_RE.match(t)
    if matrix_match:
        rows_max = _resolve_enum_max(matrix_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(matrix_match.group("cols"), symbol_values)
        if rows_max is not None and cols_max is not None:
            is_const = matrix_match.group("const") is not None
            # Matrix-like tensors are exposed as SSBO bindings, not copied into function-local arrays.
            flat_len = rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
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

    fixed_size_vector_int_match = _FIXED_SIZE_VECTOR_INT_RE.match(t)
    if fixed_size_vector_int_match:
        is_const = fixed_size_vector_int_match.group("const") is not None
        return "", "", None, BufferViewSpec(name=name, glsl_scalar="int", is_const=is_const), None

    nested_fixed_size_vector_matrix_match = _NESTED_FIXED_SIZE_VECTOR_MATRIX_RE.match(t)
    if nested_fixed_size_vector_matrix_match:
        rows_max = _resolve_enum_max(nested_fixed_size_vector_matrix_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(nested_fixed_size_vector_matrix_match.group("cols"), symbol_values)
        if rows_max is not None and cols_max is not None:
            is_const = nested_fixed_size_vector_matrix_match.group("const") is not None
            flat_len = rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
                rows=rows_max,
                cols=cols_max,
                is_const=is_const,
                flat_len=flat_len,
            ), None, None

    nested_core_match = re.match(
        r"^(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?:[A-Za-z_]\w*::)*fixed_size_vector\s*<\s*(?P<scalar>[^,]+)\s*,\s*(?P<cols>[^>]+)\s*>\s*,\s*(?P<rows>[^>]+)\s*>$",
        core_type,
    )
    if nested_core_match:
        rows_max = _resolve_enum_max(nested_core_match.group("rows"), symbol_values)
        cols_max = _resolve_enum_max(nested_core_match.group("cols"), symbol_values)
        if rows_max is not None and cols_max is not None:
            flat_len = rows_max * cols_max
            return "", "", MatrixViewSpec(
                name=name,
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
    for name, spec in specs.items():
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


def _render_kernel_stub(spec: VulkanKernelSpec, symbol_values: dict[str, str], config: VulkanConfig) -> str:
    const_preamble: list[str] = []
    for line in spec.offload_param_lines or []:
        rendered = _sanitize_offload_param_line_for_glsl(line)
        if rendered is not None:
            const_preamble.append(rendered)
    const_block = ("\n".join(const_preamble) + "\n\n") if const_preamble else ""

    filtered_body_lines = [line for line in spec.body_lines if not _SIMD_REDUCTION_PLUS_RE.match(line)]
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
                readonly = "readonly " if matrix_view_spec.is_const else ""
                block_name = f"RllmBuffer_{name}"
                ssbo_decls.append(
                    f"layout(std430, set = 0, binding = {ssbo_binding}) {readonly}buffer {block_name} {{ float {name}[{matrix_view_spec.flat_len}]; }};"
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

    # Re-render body once matrix view specs are known.
    body_text = "\n".join(
        f"    {_rewrite_matrix_view_indexing(line.lstrip(), matrix_view_specs)}" if line.strip() else ""
        for line in filtered_body_lines
    )
    if spec.is_2d and len(spec.vars) >= 2:
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
    guard_block = ""
    if spec.kernel_guard_expr:
        guard_expr = _sanitize_kernel_line_for_glsl(spec.kernel_guard_expr)
        guard_block = f"    if ({guard_expr}) {{\n        return;\n    }}\n"
    call_arg_names.extend(
        name for name in extra_param_names if name not in matrix_view_specs and name not in buffer_view_specs
    )
    call_args = ", ".join(call_arg_names)
    ssbo_block = "\n".join(ssbo_decls)
    if ssbo_block:
        ssbo_block = ssbo_block + "\n\n"
    push_constant_block = ""
    if scalar_param_specs:
        # Vulkan gives us a compact scalar-argument path without inventing a
        # one-off SSBO for values like learning rates.
        fields = "\n".join(f"    {scalar.glsl_type} {scalar.name};" for scalar in scalar_param_specs)
        push_constant_block = f"layout(push_constant) uniform RllmPushConstants {{\n{fields}\n}} rllm_push;\n\n"
    return (
        "#version 450\n"
        "\n"
        f"layout(local_size_x = {config.workgroup_size_x}, local_size_y = {config.workgroup_size_y}, local_size_z = {config.workgroup_size_z}) in;\n"
        "\n"
        f"{const_block}"
        f"{push_constant_block}"
        f"{ssbo_block}"
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
                vars=list(ctx.vars),
                kernel_guard_expr=ctx.kernel_guard_expr,
                extra_params=ctx.extra_params,
                extra_param_types=ctx.extra_param_types,
                offload_param_lines=ctx.offload_param_lines,
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
    generated_kernels = _write_kernel_stubs(kernel_specs, kernel_root, symbol_values, config)
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