"""Expression AST nodes for code generation."""


class Expression:
    """Base class for expression AST nodes."""
    def accept(self, visitor):
        return visitor.visit_expression(self)


class Number(Expression):
    def __init__(self, value=None):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_number(self)


class Identifier(Expression):
    def __init__(self, name=None):
        self.name = name

    def accept(self, visitor):
        return visitor.visit_identifier(self)


class IndexedIdentifier(Expression):
    """Represents an identifier with one or more index expressions (e.g., dst[i])."""
    def __init__(self, base=None, indices=None):
        self.base = base
        self.indices = indices or []

    def accept(self, visitor):
        return visitor.visit_indexed_identifier(self)


class LimitExpr(Expression):
    def __init__(self, max_val=None, body=None):
        self.max_val = max_val
        self.body = body

    def accept(self, visitor):
        return visitor.visit_limit_expr(self)


class BinaryExpr(Expression):
    """Represents a binary operation (e.g., a + b)."""
    def __init__(self, left=None, op=None, right=None):
        self.left = left
        self.op = op
        self.right = right

    def accept(self, visitor):
        return visitor.visit_binary_expr(self)


class CastExpr(Expression):
    """Represents a cast expression (e.g., int(x))."""
    def __init__(self, cast_type=None, operand=None):
        self.cast_type = cast_type
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_cast_expr(self)


class NegationExpr(Expression):
    """Represents a negation expression (e.g., !x)."""
    def __init__(self, operand=None):
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_negation_expr(self)


__all__ = [
    'Expression', 'Number', 'Identifier', 'IndexedIdentifier',
    'LimitExpr', 'BinaryExpr', 'CastExpr', 'NegationExpr',
]
