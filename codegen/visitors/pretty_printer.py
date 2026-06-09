"""Pretty-printing visitor for AST nodes."""

from .visitor import (
    Visitor, Type, Int, Float, FixedSizeVector, Expression,
    Number, Identifier, IndexedIdentifier, LimitExpr, Condition,
    Statement, For, If, Declaration, Assignment, Program
)


class PrettyPrinter(Visitor):
    """Pretty-prints the AST to a string. Uses indentation to show hierarchy."""

    def __init__(self, indent: int = 0):
        self.indent = indent

    def _child_indent(self) -> "PrettyPrinter":
        return PrettyPrinter(self.indent + 2)

    def _indent_str(self) -> str:
        return " " * self.indent

    def visit_type(self, node: Type):
        return node.accept(self._child_indent())

    def visit_int(self, node: Int):
        return f"{self._indent_str()}int"

    def visit_float(self, node: Float):
        return f"{self._indent_str()}float"

    def visit_fixed_size_vector(self, node: FixedSizeVector):
        indent = self._indent_str()
        inner = ""
        if node.elem_type is not None:
            inner += indent + "  elem_type: " + node.elem_type.accept(PrettyPrinter()) + "\n"
        if node.size_expr is not None:
            inner += indent + "  size_expr: " + node.size_expr.accept(PrettyPrinter()) + "\n"
        return f"{indent}fixed_size_vector<\n{inner}{indent}>"

    def visit_expression(self, node: Expression):
        return node.accept(self._child_indent())

    def visit_number(self, node: Number):
        return f"{self._indent_str()}{node.value}"

    def visit_identifier(self, node: Identifier):
        return f"{self._indent_str()}{node.name}"

    def visit_indexed_identifier(self, node: IndexedIdentifier):
        indent = self._indent_str()
        parts = [node.base.name]
        for idx in node.indices:
            parts.append(f"[{idx.accept(PrettyPrinter())}]")
        return f"{self._indent_str()}{''.join(parts)}"

    def visit_limit_expr(self, node: LimitExpr):
        indent = self._indent_str()
        max_part = node.max_val.accept(PrettyPrinter())
        body_part = node.body.accept(PrettyPrinter())
        inner_indent = " " * (self.indent + 2)
        return (f"{indent}limit<\n"
                f"{inner_indent}{max_part}\n"
                f"{indent}>("
                f"\n{inner_indent}{body_part}"
                f"\n{indent})")

    def visit_condition(self, node: Condition):
        indent = self._indent_str()
        lhs_part = node.lhs.accept(PrettyPrinter())
        rhs_part = node.rhs.accept(PrettyPrinter())
        inner_indent = " " * (self.indent + 2)
        return (f"{indent}condition:\n"
                f"{inner_indent}lhs: {lhs_part}\n"
                f"{inner_indent}op: \"{node.op}\"\n"
                f"{inner_indent}rhs: {rhs_part}")

    def visit_statement(self, node: Statement):
        return node.accept(self._child_indent())

    def visit_for(self, node: For):
        indent = self._indent_str()
        inner_indent = " " * (self.indent + 2)
        parts = []
        if node.loop_var_type is not None:
            parts.append(f"{inner_indent}loop_var_type: {node.loop_var_type.accept(PrettyPrinter())}")
        parts.append(f"{inner_indent}loop_var_name: \"{node.loop_var_name}\"")
        if node.condition is not None:
            parts.append(f"{inner_indent}condition:\n{inner_indent}{node.condition.accept(PrettyPrinter())}")
        if node.increment_var is not None:
            parts.append(f"{inner_indent}increment: {node.increment_var} {node.increment_op}")
        if node.body_stmts:
            body_lines = []
            for s in node.body_stmts:
                body_lines.append(s.accept(PrettyPrinter()))
            parts.append(f"{inner_indent}body:\n" + "\n".join(" " * (self.indent + 4) + l.lstrip() for l in body_lines))
        return f"{indent}for (\n" + ",\n".join(parts) + f"\n{indent})"

    def visit_if(self, node: If):
        indent = self._indent_str()
        inner_indent = " " * (self.indent + 2)
        parts = []
        if node.condition is not None:
            parts.append(f"{inner_indent}condition:\n{inner_indent}{node.condition.accept(PrettyPrinter())}")
        if node.body_stmts:
            body_lines = []
            for s in node.body_stmts:
                body_lines.append(s.accept(PrettyPrinter()))
            parts.append(f"{inner_indent}body:\n" + "\n".join(" " * (self.indent + 4) + l.lstrip() for l in body_lines))
        return f"{indent}if (\n" + ",\n".join(parts) + f"\n{indent})"

    def visit_declaration(self, node: Declaration):
        indent = self._indent_str()
        inner_indent = " " * (self.indent + 2)
        const_prefix = "const " if node.is_const else ""
        return (f"{indent}{const_prefix}"
                f"type:\n{inner_indent}{node.var_type.accept(PrettyPrinter())}\n"
                f"{inner_indent}name: \"{node.name}\"")

    def visit_assignment(self, node: Assignment):
        indent = self._indent_str()
        inner_indent = " " * (self.indent + 2)
        return (f"{indent}{node.lvalue.accept(PrettyPrinter())}\n"
                f"{inner_indent}= {node.rvalue.accept(PrettyPrinter())}")

    def visit_program(self, node: Program):
        indent = self._indent_str()
        inner_indent = " " * (self.indent + 2)
        lines = [f'{indent}header: "{node.header}"',
                 f'{indent}space: {node.space}']
        if node.limit_expr is not None:
            limit_part = node.limit_expr.accept(PrettyPrinter())
            lines.append(f"{inner_indent}limit_expr:\n{limit_part}")
        if node.params:
            params_lines = [p.accept(PrettyPrinter()) for p in node.params]
            lines.append(f"{indent}params:")
            lines.extend(" " * (self.indent + 2) + l.lstrip() for l in params_lines)
        if node.body_stmts:
            body_lines = [s.accept(PrettyPrinter()) for s in node.body_stmts]
            lines.append(f"{indent}body_stmts:")
            lines.extend(" " * (self.indent + 2) + l.lstrip() for l in body_lines)
        return "\n".join(lines)
