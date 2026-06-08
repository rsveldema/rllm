"""Marker base class for all AST statement nodes."""


class Statement:
    """Abstract marker for all AST statement nodes.

    Each concrete subclass provides its own ``visit(visitor)`` method
    that dispatches to the matching ``visit_*`` on *visitor*.
    """
    __slots__ = ()  # purely a marker; subclasses add dataclass fields
