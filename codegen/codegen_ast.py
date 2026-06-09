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


class LimitExpr(Expression):
    def __init__(self, max_val: Expression, body: Expression):
        self.max_val = max_val
        self.body = body

    def accept(self, visitor):
        return visitor.visit_limit_expr(self)


class Condition:
    def __init__(self, lhs: Expression, op: str = "<"):
        self.lhs = lhs
        self.op = op

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
    def __init__(self, lvalue: Identifier, rvalue: Expression):
        self.lvalue = lvalue
        self.rvalue = rvalue

    def accept(self, visitor):
        return visitor.visit_assignment(self)


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

    def visit_expression(self, node: Expression):
        raise NotImplementedError

    def visit_number(self, node: Number):
        raise NotImplementedError

    def visit_identifier(self, node: Identifier):
        raise NotImplementedError

    def visit_limit_expr(self, node: LimitExpr):
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

    def visit_program(self, node: Program):
        raise NotImplementedError


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
        inner_indent = " " * (self.indent + 2)
        return (f"{indent}condition:\n"
                f"{inner_indent}lhs: {lhs_part}\n"
                f"{inner_indent}op: \"{node.op}\"")

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
