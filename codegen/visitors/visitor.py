"""Base visitor pattern: all AST node types and the Visitor base class."""


class Type:
    def accept(self, visitor):
        return visitor.visit_type(self)


class Int(Type):
    def __init__(self):
        pass

    def accept(self, visitor):
        return visitor.visit_int(self)


class Float(Type):
    def __init__(self):
        pass

    def accept(self, visitor):
        return visitor.visit_float(self)


class FixedSizeVector(Type):
    def __init__(self, elem_type: Type = None, size_expr = None):
        self.elem_type = elem_type
        self.size_expr = size_expr

    def accept(self, visitor):
        return visitor.visit_fixed_size_vector(self)


class FlexibleRowsMatrix(Type):
    """flexible_rows_matrix<elem_type, row_type, size_expr>&"""

    def __init__(self, elem_type: Type = None, row_type: Type = None, size_expr = None):
        self.elem_type = elem_type
        self.row_type = row_type
        self.size_expr = size_expr

    def accept(self, visitor):
        return visitor.visit_flexible_rows_matrix(self)


class FixedSizeMatrix(Type):
    """fixed_size_matrix<elem_type, row_type, col_type>&"""

    def __init__(self, elem_type: Type = None, row_type: Type = None, col_type: Type = None):
        self.elem_type = elem_type
        self.row_type = row_type
        self.col_type = col_type

    def accept(self, visitor):
        return visitor.visit_fixed_size_matrix(self)


class Expression:
    def accept(self, visitor):
        return visitor.visit_expression(self)


class Number(Expression):
    def __init__(self, value: int):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_number(self)


class Identifier(Expression):
    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor):
        return visitor.visit_identifier(self)


class IndexedIdentifier(Expression):
    """Represents an identifier with one or more index expressions (e.g., dst[i])."""

    def __init__(self, base: Identifier, indices: list = None):
        self.base = base
        self.indices = indices or []

    def accept(self, visitor):
        return visitor.visit_indexed_identifier(self)


class LimitExpr(Expression):
    def __init__(self, max_val: Expression, body: Expression):
        self.max_val = max_val
        self.body = body

    def accept(self, visitor):
        return visitor.visit_limit_expr(self)


class BinaryExpr(Expression):
    """Represents a binary operation (e.g., a + b)."""

    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op
        self.right = right

    def accept(self, visitor):
        return visitor.visit_binary_expr(self)


class CastExpr(Expression):
    """Represents a cast expression (e.g., int(x))."""

    def __init__(self, cast_type: Type, operand: Expression):
        self.cast_type = cast_type
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_cast_expr(self)


class NegationExpr(Expression):
    """Represents a negation expression (e.g., !x)."""

    def __init__(self, operand: Expression):
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_negation_expr(self)


class Condition:
    def __init__(self, lhs: Expression, op: str, rhs: Expression):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def accept(self, visitor):
        return visitor.visit_condition(self)


class Statement:
    def accept(self, visitor):
        return visitor.visit_statement(self)


class For(Statement):
    def __init__(self, loop_var_type, loop_var_name, condition,
                 increment_var, increment_op, body_stmts):
        self.loop_var_type = loop_var_type
        self.loop_var_name = loop_var_name
        self.condition = condition
        self.increment_var = increment_var
        self.increment_op = increment_op
        self.body_stmts = body_stmts

    def accept(self, visitor):
        return visitor.visit_for(self)


class If(Statement):
    def __init__(self, condition, body_stmts):
        self.condition = condition
        self.body_stmts = body_stmts

    def accept(self, visitor):
        return visitor.visit_if(self)


class Declaration(Statement):
    def __init__(self, is_const: bool, var_type: Type, name: str):
        self.is_const = is_const
        self.var_type = var_type
        self.name = name

    def accept(self, visitor):
        return visitor.visit_declaration(self)


class Assignment(Statement):
    def __init__(self, lvalue: Expression, assign_op: str, rvalue: Expression):
        self.lvalue = lvalue
        self.assign_op = assign_op
        self.rvalue = rvalue

    def accept(self, visitor):
        return visitor.visit_assignment(self)


class OverflowCheck(Statement):
    """OVERFLOW_CHECK_ADD(lvalue, operand) statement."""

    def __init__(self, lvalue: Expression, operand: Expression):
        self.lvalue = lvalue
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_overflow_check(self)


class Program:
    def __init__(self, header: str = "", space: str = "", limit_expr = None,
                 params: list = None, body_stmts: list = None):
        self.header = header
        self.space = space
        self.limit_expr = limit_expr
        self.params = params or []
        self.body_stmts = body_stmts or []

    def accept(self, visitor):
        return visitor.visit_program(self)

    def visit_children(self, visitor):
        """Visit this program and all its children nodes via the given visitor."""
        result = self.accept(visitor)
        for param in self.params:
            if hasattr(param, 'accept'):
                param.accept(visitor)
        for stmt in self.body_stmts:
            if hasattr(stmt, 'accept'):
                stmt.accept(visitor)
        return result


class Visitor:
    """Base visitor class. Subclasses override visit_* methods as needed."""

    def visit_type(self, node: Type):
        raise NotImplementedError

    def visit_int(self, node: Int):
        raise NotImplementedError

    def visit_float(self, node: Float):
        raise NotImplementedError

    def visit_fixed_size_vector(self, node: FixedSizeVector):
        raise NotImplementedError

    def visit_flexible_rows_matrix(self, node: FlexibleRowsMatrix):
        raise NotImplementedError

    def visit_fixed_size_matrix(self, node: FixedSizeMatrix):
        raise NotImplementedError

    def visit_expression(self, node: Expression):
        raise NotImplementedError

    def visit_number(self, node: Number):
        raise NotImplementedError

    def visit_identifier(self, node: Identifier):
        raise NotImplementedError

    def visit_indexed_identifier(self, node: IndexedIdentifier):
        raise NotImplementedError

    def visit_limit_expr(self, node: LimitExpr):
        raise NotImplementedError

    def visit_binary_expr(self, node: BinaryExpr):
        raise NotImplementedError

    def visit_cast_expr(self, node: CastExpr):
        raise NotImplementedError

    def visit_negation_expr(self, node: NegationExpr):
        raise NotImplementedError

    def visit_condition(self, node: Condition):
        raise NotImplementedError

    def visit_statement(self, node: Statement):
        raise NotImplementedError

    def visit_for(self, node: For):
        raise NotImplementedError

    def visit_if(self, node: If):
        raise NotImplementedError

    def visit_declaration(self, node: Declaration):
        raise NotImplementedError

    def visit_assignment(self, node: Assignment):
        raise NotImplementedError

    def visit_overflow_check(self, node: OverflowCheck):
        raise NotImplementedError

    def visit_program(self, node: Program):
        raise NotImplementedError
