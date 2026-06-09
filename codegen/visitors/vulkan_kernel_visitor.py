"""Vulkan-compatible code generator (GLSL compute shader output)."""

from typing import Optional

from .. import ast
from .visitor import Visitor


class VulkanKernelVisitor(Visitor):
    """Transforms the parsed AST into a Vulkan GLSL compute shader string."""

    def __init__(self):
        self._lines: list[str] = []
        self._indent_level: int = 0
        self._param_counter: int = 0
        self._shared_counter: int = 0
        self._struct_names: set[str] = set()
        # Matrix parameter info for inline access expression generation
        self._matrix_params: list[dict] = []

    # ── helpers ────────────────────────────────────────────────────────

    def _indent(self) -> str:
        return "    " * self._indent_level

    def _emit(self, line: str) -> None:
        self._lines.append(f"{self._indent()}{line}")

    def _push(self) -> None:
        self._indent_level += 1

    def _pop(self) -> None:
        self._indent_level -= 1

    def result(self) -> str:
        return "\n".join(self._lines) + "\n"

    # ── type helpers (GLSL) ────────────────────────────────────────────

    def _type_str(self, ty: ast.Type) -> str:
        if isinstance(ty, ast.Int):
            return "uint"
        if isinstance(ty, ast.Float):
            return "float"
        if isinstance(ty, ast.FixedSizeVector):
            inner = self._type_str(ty.elem_type) if ty.elem_type else "float"
            size_expr = ty.size_expr
            if isinstance(size_expr, ast.Number):
                return f"{inner}[{size_expr.value}]"
            else:
                return f"{inner}[]"
        if isinstance(ty, (ast.FlexibleRowsMatrix, ast.FixedSizeMatrix)):
            return self._matrix_type_struct_name(ty)
        return "unknown_type"

    def _matrix_type_struct_name(self, ty: ast.Type) -> str:
        inner = self._type_str(ty.elem_type) if ty.elem_type else "float"
        is_flexible = isinstance(ty, ast.FlexibleRowsMatrix)
        tag = "flexible_rows_matrix" if is_flexible else "fixed_size_matrix"
        return f"{tag}_{inner}"

    def _matrix_type_key(self, ty: ast.Type) -> tuple:
        inner = self._type_str(ty.elem_type) if ty.elem_type else "float"
        is_flexible = isinstance(ty, ast.FlexibleRowsMatrix)
        tag = "flexible_rows_matrix" if is_flexible else "fixed_size_matrix"
        return (tag, inner)

    def _matrix_total_elements(self, ty: ast.Type):
        row_size = ty.row_size_expr
        col_size = ty.col_size_expr
        row_val = None
        col_val = None
        if isinstance(row_size, ast.Number):
            row_val = int(row_size.value)
        if isinstance(col_size, ast.Number):
            col_val = int(col_size.value)
        if row_val is not None and col_val is not None:
            return (row_val * col_val, True)
        return (None, False)

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: ast.Type) -> str:
        return self._type_str(node)

    def visit_int(self, node: ast.Int) -> str:
        return self._type_str(node)

    def visit_float(self, node: ast.Float) -> str:
        return self._type_str(node)

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector) -> str:
        inner = self._type_str(node.elem_type) if node.elem_type else "float"
        size_expr = node.size_expr
        if isinstance(size_expr, ast.Number):
            return f"{inner}[{size_expr.value}]"
        else:
            return f"{inner}[]"

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix) -> str:
        return self._matrix_type_struct_name(node)

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix) -> str:
        return self._matrix_type_struct_name(node)

    def visit_expression(self, node: ast.Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: ast.Number) -> str:
        val = node.value
        if isinstance(val, float):
            s = f"{val}"
            if "." not in s:
                s += ".0"
            return s
        return str(val)

    def visit_identifier(self, node: ast.Identifier) -> str:
        name = node.name
        unif_map = getattr(self, '_uniform_map', {})
        if name in unif_map:
            buf_type, instance_name, member = unif_map[name]
            return f"{instance_name}.{member}"
        ssbo_map = getattr(self, '_ssbo_map', {})
        if name in ssbo_map:
            buf_type, instance_name = ssbo_map[name]
            return f"{instance_name}.data"
        return node.name

    def _build_matrix_access(self, param_idx: int, idx1_node, idx2_node):
        """Build inline matrix data access expression for SSBO buffer."""
        mp = self._matrix_params[param_idx]
        instance_name = mp["instance"]
        
        def _to_str(e):
            if isinstance(e, ast.Number):
                return str(int(e.value))
            elif hasattr(e, 'accept'):
                return e.accept(self)
            else:
                return str(e)

        r_part = _to_str(idx1_node)
        if idx2_node is None or (isinstance(idx2_node, ast.Number) and idx2_node.value == 0):
            c_part = "uint(0)"
        else:
            c_part = _to_str(idx2_node)

        if mp.get("all_const") and mp.get("total") is not None:
            total = str(mp["total"])
            return f"{instance_name}.data[{r_part} * {total} + {c_part}]"
        # Handle vectors (1D) vs matrices (2D) differently
        if not mp.get("is_matrix", True):
            return f"{instance_name}.data[{r_part}]"
        return f"{instance_name}.data[{r_part} * {instance_name}.rows + {c_part}]"

    def visit_indexed_identifier(self, node: ast.IndexedIdentifier) -> str:
        base = node.base.name
        # Check if this base name maps to a matrix stored in an SSBO
        for param_idx, mp in enumerate(self._matrix_params):
            if base == mp["param_name"]:
                return self._build_matrix_access(param_idx, node.indices[0],
                                                 node.indices[1] if len(node.indices) > 1 else None)

        # Check SSBO map (for FixedSizeVector parameters)
        ssbo_map = getattr(self, '_ssbo_map', {})
        if base in ssbo_map:
            buf_type, instance_name = ssbo_map[base]
            parts = [f"{instance_name}.data"]
            for idx in node.indices:
                parts.append(f"[{idx.accept(self)}]")
            return "".join(parts)

        # Check uniform map (for scalar params in uniform block)
        unif_map = getattr(self, '_uniform_map', {})
        if base in unif_map:
            buf_type, instance_name, member = unif_map[base]
            parts = [f"{instance_name}.{member}"]
            for idx in node.indices:
                parts.append(f"[{idx.accept(self)}]")
            return "".join(parts)

        # Fallback to base name with indices
        parts = [node.base.name]
        for idx in node.indices:
            parts.append(f"[{idx.accept(self)}]")
        return "".join(parts)

    def visit_limit_expr(self, node: ast.LimitExpr) -> str:
        max_str = self.visit_number(node.max_val) if isinstance(node.max_val, ast.Number) else (node.max_val or "0")
        body_str = node.body.accept(self) if node.body else "0"
        return f"limit<{max_str}>({body_str})"

    def visit_binary_expr(self, node: ast.BinaryExpr) -> str:
        left_str = node.left.accept(self) if node.left else ""
        right_str = node.right.accept(self) if node.right else ""
        op = node.op or "+"
        return f"{left_str} {op} {right_str}"

    def visit_cast_expr(self, node: ast.CastExpr) -> str:
        cast_type = node.cast_type.accept(self) if isinstance(node.cast_type, (ast.Int, ast.Float)) else "uint"
        operand = node.operand.accept(self) if node.operand else ""
        return f"{cast_type}({operand})"

    def visit_negation_expr(self, node: ast.NegationExpr) -> str:
        operand = node.operand.accept(self) if node.operand else ""
        return f"!{operand}"

    def visit_condition(self, node: ast.Condition) -> str:
        lhs_str = node.lhs.accept(self) if isinstance(node.lhs, (ast.Expression, ast.Identifier)) else str(node.lhs)
        rhs_str = node.rhs.accept(self) if isinstance(node.rhs, ast.Expression) else str(node.rhs)
        op = node.op or ""
        return f"{lhs_str} {op} {rhs_str}"

    # ── statement visitors (emit to _lines) ───────────────────────────

    def visit_statement(self, node: ast.Statement) -> None:
        return node.accept(self)

    def visit_for(self, node: ast.For) -> None:
        indent = self._indent()
        loop_type = node.loop_var_type.accept(self) if isinstance(node.loop_var_type, (ast.Int, ast.Float)) else "uint"
        cond_str = node.condition.accept(self) if node.condition else ""
        body_lines = []
        for s in node.body_stmts:
            if hasattr(s, 'accept'):
                old_emit = self._emit
                old_lines = self._lines
                temp_lines = []

                def capture_emit(line):
                    temp_lines.append(f"{self._indent()}{line}")

                self._emit = capture_emit
                self._lines = temp_lines
                s.accept(self)
                body_lines.extend(temp_lines)
                self._emit = old_emit
                self._lines = old_lines
        self._emit(f"for ({loop_type} {node.loop_var_name}; "
                   f"{cond_str}; {node.increment_var}{node.increment_op}) {{")
        for line in body_lines:
            self._emit(line)
        self._emit("}")

    def visit_if(self, node: ast.If) -> None:
        indent = self._indent()
        cond_str = node.condition.accept(self) if node.condition else ""
        body_lines = []
        for s in node.body_stmts:
            if hasattr(s, 'accept'):
                old_emit = self._emit
                old_lines = self._lines
                temp_lines = []

                def capture_emit(line):
                    temp_lines.append(f"{self._indent()}{line}")

                self._emit = capture_emit
                self._lines = temp_lines
                s.accept(self)
                body_lines.extend(temp_lines)
                self._emit = old_emit
                self._lines = old_lines
        self._emit(f"if ({cond_str}) {{")
        for line in body_lines:
            self._emit(line)
        self._emit("}")

    def visit_declaration(self, node: ast.Declaration) -> None:
        indent = self._indent()
        prefix = "const " if node.is_const else ""
        init_str = f" = {node.init_expr.accept(self)}" if node.init_expr else ""
        var_type_str = node.var_type.accept(self)
        self._emit(f"{prefix}{var_type_str} {node.name}{init_str};")

    def visit_assignment(self, node: ast.Assignment) -> None:
        indent = self._indent()
        self._emit(f"{node.lvalue.accept(self)} {node.assign_op} {node.rvalue.accept(self)};")

    def visit_overflow_check(self, node: ast.OverflowCheck) -> None:
        indent = self._indent()
        self._emit(f"OVERFLOW_CHECK_ADD({node.lvalue.accept(self)}, {node.operand.accept(self)})")

    def visit_shared_decl(self, node: ast.SharedDecl) -> None:
        indent = self._indent()
        prefix = "const " if node.is_const else ""
        init_str = f" = {node.init_expr.accept(self)}" if node.init_expr else ""
        var_type_str = self._type_str(node.var_type)
        self._emit(f"shared {prefix}{var_type_str} {node.name}{init_str};")

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties) -> str:
        indent = self._indent()
        parts = []
        if node.x_expr is not None:
            parts.append(f"x: {node.x_expr.accept(self)}")
        if node.y_expr is not None:
            parts.append(f"y: {node.y_expr.accept(self)}")
        if node.z_expr is not None:
            parts.append(f"z: {node.z_expr.accept(self)}")
        return f"{indent}workgroup {{ {', '.join(parts)} }}"

    # ── program visitor ────────────────────────────────────────────────

    def visit_program(self, node: ast.Program) -> str:
        self._lines = []
        self._indent_level = 0
        self._param_counter = 0
        self._shared_counter = 0
        self._struct_names = set()
        self._matrix_params = []

        # Version + workgroup dispatch
        self._emit("#version 450")
        self._emit("")
        if node.workgroups and len(node.workgroups) > 0:
            wg = node.workgroups[0]
            parts = []
            if hasattr(wg, 'x_expr') and wg.x_expr is not None:
                x_val = wg.x_expr.accept(self) if isinstance(wg.x_expr, ast.Number) else "1"
                parts.append(f"x = {x_val}")
            if hasattr(wg, 'y_expr') and wg.y_expr is not None:
                y_val = wg.y_expr.accept(self) if isinstance(wg.y_expr, ast.Number) else "1"
                parts.append(f"y = {y_val}")
            if hasattr(wg, 'z_expr') and wg.z_expr is not None:
                z_val = wg.z_expr.accept(self) if isinstance(wg.z_expr, ast.Number) else "1"
                parts.append(f"z = {z_val}")
            if parts:
                self._emit(f"layout(local_size_x = {', '.join(parts)}) in;")

        # Shared declarations at top-level
        for item in node.workgroups:
            if isinstance(item, ast.SharedDecl):
                var_type = self._type_str(item.var_type)
                init = f" = {item.init_expr.accept(self)}" if item.init_expr else ""
                const_kw = "const " if item.is_const else ""
                self._emit(f"layout(location = {self._shared_counter}) shared {const_kw}{var_type} {item.name}{init};")
                self._shared_counter += 1

        # Classify parameters by storage type
        matrix_params = []          # Matrices -> dedicated SSBO with std430
        vector_params = []          # FixedSizeVector -> separate SSBO block  
        scalar_params = []          # Scalars (Int/Float) -> uniform block

        for param in node.params:
            if param is None or not isinstance(param, ast.Declaration):
                continue
            if isinstance(param.var_type, (ast.FlexibleRowsMatrix, ast.FixedSizeMatrix)):
                matrix_params.append(param)
            elif isinstance(param.var_type, ast.FixedSizeVector):
                vector_params.append(param)
            else:
                scalar_params.append(param)

        # ── 1. Emit SSBO buffer blocks for matrix parameters ──
        binding = 0
        for param in matrix_params:
            buf_name = f"_ssbo_{binding}"
            inner = "float"  # Matrix elements are float
            
            # Use fixed-size array when dimensions are known at compile-time
            total, all_const = self._matrix_total_elements(param.var_type)
            if all_const and total is not None:
                self._emit(f"layout(binding = {binding}, std430) buffer {buf_name} {{")
                self._push()
                self._emit(f"{inner}[{total}] data;")
                self._pop()
                self._emit(f"}} {param.name};")
            else:
                # Variable size: use dynamic array with row/cols tracking
                self._emit(f"layout(binding = {binding}, std430) buffer {buf_name} {{")
                self._push()
                self._emit(f"{inner}[] data;")
                self._emit(f"uint rows;")
                self._emit(f"uint cols;")
                self._pop()
                self._emit(f"}} {param.name};")

            mp_info = {
                "param_name": param.name,
                "instance": param.name,
                "is_matrix": True,
                "all_const": all_const,
                "total": total,
            }
            self._matrix_params.append(mp_info)

            if not hasattr(self, '_ssbo_map'):
                self._ssbo_map = {}
            self._ssbo_map[param.name] = (buf_name, param.name)
            binding += 1

        # ── 1b. Emit SSBO buffer blocks for FixedSizeVector parameters ──
        for param in vector_params:
            buf_name = f"_ssbo_{binding}"
            inner = self._type_str(param.var_type.elem_type) if param.var_type.elem_type else "float"
            
            # Use fixed-size array when size is known at compile-time
            size_expr = param.var_type.size_expr
            if isinstance(size_expr, ast.Number):
                total = int(size_expr.value)
                self._emit(f"layout(binding = {binding}, std430) buffer {buf_name} {{")
                self._push()
                self._emit(f"{inner}[{total}] data;")
                self._pop()
                self._emit(f"}} {param.name};")
            else:
                # Runtime size: dynamic array
                self._emit(f"layout(binding = {binding}, std430) buffer {buf_name} {{")
                self._push()
                self._emit(f"{inner}[] data;")
                self._pop()
                self._emit(f"}} {param.name};")

            mp_info = {
                "param_name": param.name,
                "instance": param.name,
                "is_matrix": False,
                "all_const": isinstance(size_expr, ast.Number),
                "total": int(size_expr.value) if isinstance(size_expr, ast.Number) else None,
            }
            self._matrix_params.append(mp_info)

            if not hasattr(self, '_ssbo_map'):
                self._ssbo_map = {}
            self._ssbo_map[param.name] = (buf_name, param.name)
            binding += 1

        # ── 2. Emit uniform block for scalar params ──
        unif_binding = binding

        if scalar_params:
            members = []
            for param in scalar_params:
                vtype = self._type_str(param.var_type)
                init = f" = {param.init_expr.accept(self)}" if param.init_expr else ""
                members.append((param.name, vtype, init))

            if members:
                buf_name = f"_uniform_{unif_binding}"
                instance_name = f"_ui_{unif_binding}"
                self._emit(f"layout(binding = {unif_binding}) uniform {buf_name} {{")
                self._push()
                for name, vtype, init in members:
                    self._emit(f"{vtype} {name}{init};")
                self._pop()
                self._emit(f"}} {instance_name};")
                if not hasattr(self, '_uniform_map'):
                    self._uniform_map = {}
                for name, _, _ in members:
                    self._uniform_map[name] = (buf_name, instance_name, name)

        # ── 3. Main function body --
        self._emit("")
        self._emit("void main() {")
        self._push()

        # Emit loop variable initialization from GLSL built-ins
        if node.space_dim >= 1 and node.loop_vars:
            for idx, var_name in enumerate(node.loop_vars):
                self._emit(
                    "uint " + var_name + " = gl_GlobalInvocationID."
                    + "xyz"[idx] + ";"
                )

        # Body statements
        body_lines = []
        for stmt in node.body_stmts:
            if hasattr(stmt, 'accept'):
                old_emit = self._emit
                old_lines = self._lines
                temp_lines = []

                def capture_emit(line):
                    temp_lines.append(f"{self._indent()}{line}")

                self._emit = capture_emit
                self._lines = temp_lines
                stmt.accept(self)
                body_lines.extend(temp_lines)
                self._emit = old_emit
                self._lines = old_lines

        for line in body_lines:
            self._emit(line)

        self._pop()
        self._emit("}")

        return self.result()
