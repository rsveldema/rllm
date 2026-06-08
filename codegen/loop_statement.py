"""AST node: for / while loop statement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopStatement:
    """A loop (for / while) with a header and body lines."""
    header: str          # the full ``for (...)`` part
    body_lines: list[str] | None = None  # inner content

    def visit(self, visitor: "Visitor") -> str:
        return visitor.visit_loop_statement(self)
