"""Abstract Visitor base for Statement tree traversal."""

from __future__ import annotations

import abc


class Visitor(abc.ABC):
    """Abstract visitor with one dispatch method per concrete Statement type."""

    @abc.abstractmethod
    def visit_expression(self, node: "Expression") -> str:
        ...

    @abc.abstractmethod
    def visit_loop_statement(self, node: "LoopStatement") -> str:
        ...

    @abc.abstractmethod
    def visit_if_statement(self, node: "IfStatement") -> str:
        ...

    @abc.abstractmethod
    def visit_raw_line(self, node: "RawLine") -> str:
        ...
