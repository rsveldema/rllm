"""C++ stub generator for Vulkan compute kernels.

Generates compilable C++ header with:
- Struct wrappers for matrix/vector types used in SSBOs
- Descriptor set layout generation code
- A dispatch function signature matching the kernel's parameters
"""

from typing import Dict, List, Optional, Set, Tuple
from .. import ast
from .visitor import Visitor


class VulkanCppStubVisitor(Visitor):
    """Transforms the parsed AST into a C++ stub for calling the kernel."""

    def __init__(self):
        self._lines: List[str] = []
        self._indent_level: int = 0
        # Maps (tag, elem) -> struct_name (FRM_float, SRM_float, etc.)
        self._matrix_structs: Dict[tuple, str] = {}
        self._kernel_name: str = ""

    # ── helpers ────────────────────────────────────────────────────────

    def _emit(self, line: str = "") -> None:
        self._lines.append("    " * self._indent_level + line)

    def _push(self) -> None:
        self._indent_level += 1

    def _pop(self) -> None:
        self._indent_level -= 1

    def result(self) -> str:
        return "\n".join(self._lines) + "\n"

    # ── type helpers (C++) ─────────────────────────────────────────────

    def _matrix_type_key(self, ty: ast.Type) -> tuple:
        """Returns ('flex_rows'|'fixed_rows', elem_cpp_type)."""
        elem = self._cpp_type_str(ty.elem_type) if getattr(ty, 'elem_type', None) else "float"
        is_flex = isinstance(ty, ast.FlexibleRowsMatrix)
        return (("flex_rows", elem) if is_flex else ("fixed_rows", elem))

    def _matrix_struct_name(self, key: tuple) -> str:
        tag, elem = key
        prefix = "FRM" if tag == "flex_rows" else "SRM"
        return f"{prefix}_{elem}"

    def _ensure_matrix_struct(self, ty: ast.Type) -> str:
        """Register a matrix type and return its struct name."""
        key = self._matrix_type_key(ty)
        if key not in self._matrix_structs:
            sname = self._matrix_struct_name(key)
            self._matrix_structs[key] = sname
        return self._matrix_structs[key]

    def _cpp_type_str(self, ty: ast.Type) -> str:
        if isinstance(ty, ast.Int):
            return "int32_t"
        if isinstance(ty, ast.Float):
            return "float"
        return "unknown_type"

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: ast.Type) -> str:
        return self._cpp_type_str(node)

    def visit_int(self, node: ast.Int) -> str:
        return "int32_t"

    def visit_float(self, node: ast.Float) -> str:
        return "float"

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector) -> str:
        elem = self._cpp_type_str(node.elem_type) if node.elem_type else "float"
        return f"{elem}"

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix) -> str:
        return self._ensure_matrix_struct(node)

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix) -> str:
        return self._ensure_matrix_struct(node)

    def visit_expression(self, node: ast.Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: ast.Number) -> str:
        return str(node.value)

    def visit_identifier(self, node: ast.Identifier) -> str:
        return node.name or "unknown"

    def visit_indexed_identifier(self, node: ast.IndexedIdentifier) -> str:
        base = node.base.accept(self) if node.base else ""
        indices = ", ".join(i.accept(self) for i in node.indices)
        return f"{base}[{indices}]"

    def visit_limit_expr(self, node: ast.LimitExpr) -> str:
        max_v = node.max_val.accept(self) if node.max_val else "?"
        body_v = node.body.accept(self) if node.body else "?"
        return f"limit<{max_v}>({body_v})"

    def visit_binary_expr(self, node: ast.BinaryExpr) -> str:
        left = node.left.accept(self) if node.left else "?"
        right = node.right.accept(self) if node.right else "?"
        op = node.op or "+"
        return f"({left} {op} {right})"

    def visit_cast_expr(self, node: ast.CastExpr) -> str:
        cast_type = self._cpp_type_str(node.cast_type) if node.cast_type else "int32_t"
        operand = node.operand.accept(self) if node.operand else "?"
        return f"static_cast<{cast_type}>({operand})"

    def visit_negation_expr(self, node: ast.NegationExpr) -> str:
        op = node.operand.accept(self) if node.operand else "?"
        return f"!{op}"

    # ── condition visitor ──────────────────────────────────────────────

    def visit_condition(self, node: ast.Condition) -> str:
        lhs = node.lhs.accept(self) if isinstance(node.lhs, ast.Expression) else (node.lhs or "")
        rhs = node.rhs.accept(self) if isinstance(node.rhs, ast.Expression) else (node.rhs or "")
        return f"{lhs} {node.op} {rhs}"

    # ── statement visitors ─────────────────────────────────────────────

    def visit_statement(self, node: ast.Statement) -> str:
        return node.accept(self)

    def visit_for(self, node: ast.For) -> str:
        raise NotImplementedError("visit_for: should be handled by accept")

    def visit_if(self, node: ast.If) -> str:
        raise NotImplementedError("visit_if: should be handled by accept")

    def visit_declaration(self, node: ast.Declaration) -> str:
        raise NotImplementedError("visit_declaration: should not be called directly")

    def visit_assignment(self, node: ast.Assignment) -> str:
        raise NotImplementedError("visit_assignment: should not be called directly")

    def visit_overflow_check(self, node: ast.OverflowCheck) -> str:
        raise NotImplementedError("visit_overflow_check: should not be called directly")

    def visit_shared_decl(self, node: ast.SharedDecl) -> str:
        raise NotImplementedError("visit_shared_decl: should not be called directly")

    # ── workgroup and program visitors ────────────────────────────────

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties) -> dict:
        return {
            'x': node.x_expr.accept(self) if node.x_expr else "8",
            'y': node.y_expr.accept(self) if node.y_expr else "8",
            'z': node.z_expr.accept(self) if node.z_expr else "1",
        }

    def visit_program(self, node: ast.Program) -> str:
        """Generate the complete C++ stub header."""
        self._lines = []
        self._matrix_structs = {}
        self._kernel_name = ""

        # Determine kernel name from header
        basename = "kernel"
        if node.header:
            parts = node.header.replace('"', '').split('/')
            basename = parts[-1].rsplit('.', 1)[0] if '.' in parts[-1] else parts[-1]
        self._kernel_name = f"{basename}_dispatch"

        # Classify params and collect matrix/vector structs
        matrix_params: List[Tuple[ast.Declaration, str]] = []     # (param, struct_name)
        vector_params: List[ast.Declaration] = []
        scalar_params: List[ast.Declaration] = []

        for param in node.params:
            if isinstance(param, ast.Declaration):
                vt = param.var_type
                if isinstance(vt, (ast.FlexibleRowsMatrix, ast.FixedSizeMatrix)):
                    sname = self._ensure_matrix_struct(vt)
                    matrix_params.append((param, sname))
                elif isinstance(vt, ast.FixedSizeVector):
                    vector_params.append(param)
                else:
                    scalar_params.append(param)

        # Collect dispatch dimensions
        has_2d = node.space_dim >= 2 and len(node.loop_vars) >= 2
        if has_2d:
            rows_param_name = node.loop_vars[0]
            cols_param_name = node.loop_vars[1]
        elif node.space_dim >= 1 and node.loop_vars:
            rows_param_name = node.loop_vars[0]
            cols_param_name = None
        else:
            rows_param_name = "dispatch_rows"
            cols_param_name = None

        # ── Emit C++ header ────────────────────────────────────────────

        func_params: List[str] = []
        func_params.append("    VkDevice device")
        func_params.append("    VkPipelineLayout pipeline_layout")
        func_params.append("    VkDescriptorSetLayout descriptor_set_layout")
        func_params.append("    VkCommandBuffer command_buffer")

        # Binding info comments
        binding = 0
        for _param, sname in matrix_params:
            self._emit(f"// Binding {binding}: SSBO (matrix) {sname}")
            binding += 1
        for param in vector_params:
            self._emit(f"// Binding {binding}: SSBO (vector) vec_{param.name}")
            binding += 1

        desc_set_name = "_desc_set"
        func_params.append(f"    VkDescriptorSet {desc_set_name}")

        if rows_param_name:
            func_params.append("    uint32_t dispatch_rows")
        if cols_param_name:
            func_params.append("    uint32_t dispatch_cols")

        # Matrix params — const reference (no redundant 'const' prefix)
        for param, sname in matrix_params:
            is_const = "const " if param.is_const else ""
            func_params.append(f"    {is_const}{sname}& {param.name}")

        # Vector params
        for param in vector_params:
            vec_name = f"vec_{param.name}"
            is_const = "const " if param.is_const else ""
            func_params.append(f"    {is_const}{vec_name}& {param.name}")

        # Scalar params (passed as push constant values)
        for param in scalar_params:
            ctype = self._cpp_type_str(param.var_type)
            is_const = "const " if param.is_const else ""
            init_val = ""
            if param.init_expr:
                init_val = f" = {param.init_expr.accept(self)}"
            func_params.append(f"    {is_const}{ctype} {param.name}{init_val}")

        # Workgroup sizes from workgroup properties
        wg_x, wg_y, wg_z = "8", "8", "1"
        for wg in node.workgroups:
            if isinstance(wg, ast.WorkgroupProperties):
                if wg.x_expr:
                    x_val = str(wg.x_expr.accept(self))
                    if x_val.isdigit():
                        wg_x = x_val
                if wg.y_expr:
                    y_val = str(wg.y_expr.accept(self))
                    if y_val.isdigit():
                        wg_y = y_val
                if wg.z_expr:
                    z_val = str(wg.z_expr.accept(self))
                    if z_val.isdigit():
                        wg_z = z_val

        # Emit file header and includes
        self._emit("")
        hdr_text = node.header.strip('"') if node.header else "unknown"
        self._emit(f"// ── Kernel dispatch stub: {hdr_text} ─────────────")
        self._emit("#include <cstdint>")
        self._emit("#include <vulkan/vulkan.h>")
        self._emit('')

        # Matrix structs (FRM for flexible-rows, SRM for fixed-size)
        for key in sorted(self._matrix_structs.keys(), key=lambda k: self._matrix_structs[k]):
            sname = self._matrix_structs[key]
            tag, elem = key
            is_flex = tag == "flex_rows"
            self._emit(f"struct {sname} {{")
            self._push()
            self._emit(f"{elem}* data;")
            if is_flex:
                self._emit(f"uint32_t rows;")
                self._emit(f"uint32_t cols;")
            else:
                self._emit(f"static constexpr uint32_t rows = 1;")
                self._emit(f"static constexpr uint32_t cols = 1;")
            self._pop()
            self._emit("};")

        # Vector structs
        for param in vector_params:
            vt = param.var_type
            elem = self._cpp_type_str(vt.elem_type) if vt.elem_type else "float"
            vec_name = f"vec_{param.name}"
            self._emit("")
            self._emit(f"struct {vec_name} {{")
            self._push()
            self._emit(f"{elem}* data;")
            self._emit(f"uint32_t size;")
            self._pop()
            self._emit("};")

        # Descriptor set struct
        if matrix_params or vector_params:
            self._emit("")
            self._emit(f"struct {self._kernel_name}_DescriptorSet {{")
            self._push()
            b = 0
            for param, sname in matrix_params:
                self._emit(f'{sname}& {param.name}; // binding {b}')
                b += 1
            for param in vector_params:
                vec_name = f"vec_{param.name}"
                self._emit(f'{vec_name}& {param.name}; // binding {b}')
                b += 1
            self._pop()
            self._emit("};")

        # Push constants struct for scalar params
        if scalar_params:
            pc_name = f"{self._kernel_name}_PushConstants"
            self._emit("")
            self._emit(f"struct {pc_name} {{")
            self._push()
            for param in scalar_params:
                ctype = self._cpp_type_str(param.var_type)
                is_const = "" if param.is_const else "const "
                self._emit(f'    {is_const}{ctype} {param.name};')
            self._pop()
            self._emit("};")

        # Dispatch function signature
        self._emit("")
        self._emit(f"inline void {self._kernel_name}(")
        for i, fp in enumerate(func_params):
            comma = "," if i < len(func_params) - 1 else ""
            self._emit(fp + comma)
        self._emit(') {')
        self._push()

        # vkCmdDispatch call
        x_wg = str(wg_x)
        y_wg = str(wg_y)
        z_wg = str(wg_z)

        if cols_param_name:
            self._emit(f'vkCmdDispatch(command_buffer, '
                       f'(dispatch_rows + {x_wg} - 1) / {x_wg}, '
                       f'(dispatch_cols + {y_wg} - 1) / {y_wg}, '
                       f'{z_wg});')
        else:
            self._emit(f'vkCmdDispatch(command_buffer, '
                       f'(dispatch_rows + {x_wg} - 1) / {x_wg}, '
                       f'1, 1);')

        self._pop()
        self._emit("}")

        # Workgroup size constants (if workgroup properties present)
        if any(isinstance(wg, ast.WorkgroupProperties) for wg in node.workgroups):
            self._emit("")
            self._emit(f'constexpr uint32_t {self._kernel_name}_WG_X = {wg_x};')
            self._emit(f'constexpr uint32_t {self._kernel_name}_WG_Y = {wg_y};')
            self._emit(f'constexpr uint32_t {self._kernel_name}_WG_Z = {wg_z};')

        return self.result()
