"""AST node: expression statement (assignments, increments, function calls)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Expression:
    """An expression statement."""
    kind: str
    target: str | None = None  # left-hand side or callee

    def visit(self, visitor: "Visitor") -> str:
        return visitor.visit_expression(self)
