"""StatementPrinter visitor for converting AST back to readable C++ code."""

from __future__ import annotations

from codegen.visitor import Visitor


class StatementPrinter(Visitor):
    """Visitor that prints AST statements back as C++ source text."""

    def visit_expression(self, node) -> str:
        if node.target:
            return f"  // {node.kind}: {node.target} = ..."
        return f"  // {node.kind}: (no target)"

    def visit_loop_statement(self, node) -> str:
        result = f"  // Loop: {node.header}"
        if node.body_lines:
            result += f"\n  //   body ({len(node.body_lines)} lines)"
        return result

    def visit_if_statement(self, node) -> str:
        result = f"  // If: {node.condition}"
        if node.then_body:
            result += f"\n  //   then ({len(node.then_body)} lines)"
        if node.else_body:
            result += f"\n  //   else ({len(node.else_body)} lines)"
        return result

    def visit_raw_line(self, node) -> str:
        return f"  // Raw: {node.text}"
