"""Transform Lark trees to codegen_ast nodes.

Extracted from parser.py so that parser.py only contains grammar, parser,
parse() and a thin transform wrapper.
"""

from lark import Tree, Token
import codegen_ast


# ── helpers ────────────────────────────────────────────────────────

def _is_token(val):
    return isinstance(val, Token)


def _token_value(t):
    if isinstance(t, Token):
        return t.value
    return None


# ── extractors ─────────────────────────────────────────────────────

def extract_header(header_tree):
    """Extract string value from header tree."""
    string_tree = header_tree.children[0]
    return _token_value(string_tree.children[0])


def extract_space_name(space_tree):
    """Extract the loop/space variable name from parfor_space.

    For 1D: children = [IDENT, expression, parameters_list] → space = first IDENT
    For 2D: children = [IDENT_i, IDENT_j, expression(grid_name), parameters_list]
            → space = grid name from the expression child
    """
    if len(space_tree.children) >= 3:
        third = space_tree.children[2]
        if isinstance(third, Tree) and third.data == 'expression':
            # 2D case: extract the identifier from the expression
            return _extract_identifier_from_expr(third)
    # 1D case: first child is the IDENT loop variable
    if len(space_tree.children) >= 1:
        first = space_tree.children[0]
        if _is_token(first):
            return first.value
    return ""


def _extract_identifier_from_expr(expr_tree):
    """Extract an identifier name from an expression tree (single IDENT base)."""
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'base_expr':
            for sub in child.children:
                if isinstance(sub, Tree) and sub.data == 'lhs':
                    lhs_children = sub.children
                    if lhs_children:
                        first = lhs_children[0]
                        if _is_token(first):
                            return first.value
    return ""


def _find_expression_tree(space_tree):
    """Find the expression child in a parfor_space tree (works for 1D and 2D)."""
    for child in space_tree.children:
        if isinstance(child, Tree) and child.data == 'expression':
            return child
    return None


def extract_limit_expr(space_tree):
    """Extract limit expression from parfor_space."""
    expr_tree = _find_expression_tree(space_tree)
    if expr_tree is None:
        return None

    # Direct child check
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'limit_expr':
            return transform_expression(child)

    # Check inside base_expr wrapper
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'base_expr':
            for sub in child.children:
                if isinstance(sub, Tree) and sub.data == 'limit_expr':
                    return transform_expression(sub)

    return None


# ── expression / lvalue transforms ────────────────────────────────

def transform_expression(expr_tree):
    """Convert an expression Tree to an AST node."""
    # Direct limit_expr
    if expr_tree.data == 'limit_expr':
        max_val = transform_expression(expr_tree.children[0])
        body = transform_expression(expr_tree.children[1])
        return codegen_ast.LimitExpr(max_val, body)

    # 'expression' wrapper with binary ops: base_expr arith_operator expression
    if expr_tree.data == 'expression':
        children = expr_tree.children
        if len(children) == 3 and _is_token(children[1]):
            op_token = children[1]
            left = transform_expression(children[0])
            right = transform_expression(children[2])
            if left is not None and right is not None:
                return codegen_ast.BinaryExpr(left, op_token.value, right)

        # Simple expression — unwrap and process single child
        for child in children:
            if isinstance(child, Tree):
                result = transform_expression(child)
                if result is not None:
                    return result

    # Direct number tree
    if expr_tree.data == 'number':
        val_token = expr_tree.children[0]
        return codegen_ast.Number(int(val_token.value))

    # Handle base_expr directly (recursion target from expression or direct)
    if expr_tree.data == 'base_expr':
        return _transform_from_base(expr_tree)

    # Bare lhs trees directly
    if expr_tree.data == 'lhs':
        return _transform_lvalue(expr_tree)

    return None


def _transform_from_base(base_tree):
    """Transform a base_expr tree into an AST expression."""
    for child in base_tree.children:
        if isinstance(child, Tree):
            data = child.data
            if data == 'limit_expr':
                return transform_expression(child)
            elif data == 'lhs':
                return _transform_lvalue(child)
            elif data == 'cast':
                cast_type = _resolve_nested_type(child.children[0])
                operand = transform_expression(child.children[1])
                if operand is not None:
                    return codegen_ast.CastExpr(cast_type, operand)
            elif data == 'negation':
                operand = transform_expression(child.children[0])
                if operand is not None:
                    return codegen_ast.NegationExpr(operand)
            else:
                # Unknown Tree data — recurse (handles number, etc.)
                result = transform_expression(child)
                if result is not None:
                    return result

        # Direct Token base (e.g. IDENT in some contexts)
        elif _is_token(child):
            return codegen_ast.Identifier(child.value)

    return None


def _transform_lvalue(lhs_tree):
    """Transform an LHS tree into an AST lvalue expression.

    Handles:
      - Tree(data='lhs', children=[Token(IDENT)]) → Identifier
      - Tree(data='lhs', children=[Token(IDENT), expr, ...]) → IndexedIdentifier
      - Token(IDENT) → Identifier
      - base_expr → unwrapped via _transform_from_base
    """
    if _is_token(lhs_tree):
        return codegen_ast.Identifier(lhs_tree.value)

    data = lhs_tree.data

    # Simple identifier: lhs -> [Token(IDENT)]
    if data == 'lhs':
        first = lhs_tree.children[0]
        if _is_token(first) and len(lhs_tree.children) == 1:
            return codegen_ast.Identifier(first.value)

        # Indexed identifier: lhs -> [Token(IDENT), expr, ...]
        if _is_token(first):
            base = codegen_ast.Identifier(first.value)
            indices = []
            for child in lhs_tree.children[1:]:
                if isinstance(child, Tree):
                    expr_result = transform_expression(child)
                    if expr_result is not None:
                        indices.append(expr_result)
            return codegen_ast.IndexedIdentifier(base, indices)

        # Nested structure — recurse
        return _transform_lvalue(first)

    # base_expr (already handled by _transform_from_base in most cases)
    if data == 'base_expr':
        return _transform_from_base(lhs_tree)

    return None


# ── type transform ─────────────────────────────────────────────────

def _transform_type(type_tree):
    """Transform a 'type' Lark tree into an AST Type node."""
    data = type_tree.data
    children = type_tree.children
    if data != 'type' or len(children) < 2:
        return None

    elem_type = _resolve_nested_type(children[0])
    if elem_type is None:
        return None

    # 3-param types: distinguish by whether the third child is an expression.
    if len(children) == 3:
        third = children[2]
        if isinstance(third, Tree):
            if third.data == 'expression':
                row_type = _resolve_nested_type(children[1])
                size_expr = transform_expression(third)
                return codegen_ast.FlexibleRowsMatrix(elem_type, row_type, size_expr)
            else:
                # All three are type params → fixed_size_matrix
                col_type = _resolve_nested_type(third)
                row_type = _resolve_nested_type(children[1])
                return codegen_ast.FixedSizeMatrix(elem_type, row_type, col_type)

    # 2-param types
    elif len(children) == 2:
        second = children[1]
        if isinstance(second, Tree):
            if second.data == 'expression':
                return codegen_ast.FixedSizeVector(elem_type, transform_expression(second))
            else:
                second_type = _resolve_nested_type(second)
                if second_type is not None:
                    return codegen_ast.FlexibleRowsMatrix(elem_type, second_type, None)

    return None


def _resolve_nested_type(node):
    """Resolve a nested type from a tree node (int/float/type)."""
    if isinstance(node, Tree):
        data = node.data
        if data == 'int':
            return codegen_ast.Int()
        elif data == 'float':
            return codegen_ast.Float()
        elif data == 'type':
            return _transform_type(node)
    return None


# ── statement transform ────────────────────────────────────────────

def _is_overflow_check(stmt_tree):
    """Check if a statement tree represents an OVERFLOW_CHECK_ADD."""
    if stmt_tree.data != 'statement':
        return False
    if len(stmt_tree.children) != 2:
        return False
    lhs_child, expr_child = stmt_tree.children
    if not isinstance(lhs_child, Tree) or lhs_child.data != 'lhs':
        return False
    if not isinstance(expr_child, Tree):
        return False
    return True


def transform_statement(stmt_tree):
    """Transform a statement Tree to an AST node."""
    data = stmt_tree.data

    # Unwrap "statement" wrapper
    if data == 'statement':
        return transform_statement(stmt_tree.children[0])

    # Overflow check: 2 children [lhs, expression]
    if _is_overflow_check(stmt_tree):
        lhs_part = stmt_tree.children[0]
        expr_part = stmt_tree.children[1]
        lvalue = _transform_lvalue(lhs_part) if isinstance(lhs_part, Tree) else None
        operand = transform_expression(expr_part)
        if lvalue and operand:
            return codegen_ast.OverflowCheck(lvalue, operand)
        return None

    # Assignment: lhs assign_operator expression
    if data == 'assignment':
        lhs_part = stmt_tree.children[0]
        op_token = stmt_tree.children[1]
        rhs_part = stmt_tree.children[2]

        if isinstance(lhs_part, Tree):
            lvalue = _transform_lvalue(lhs_part)
        else:
            lvalue = codegen_ast.Identifier(
                _token_value(lhs_part) or str(lhs_part))

        assign_op = _token_value(op_token) or '='
        rvalue = transform_expression(rhs_part)
        return codegen_ast.Assignment(lvalue, assign_op, rvalue)

    # For/if statements — placeholder
    if data in ('for_statement', 'if_statement'):
        return None

    # Standalone declarations
    if data in ('decl', 'const_decl'):
        return transform_declaration(stmt_tree)

    return None


def transform_declaration(stmt_tree):
    """Transform a declaration tree into Declaration AST."""
    is_const = stmt_tree.data == 'const_decl'
    var_type = None
    name = ""

    for child in stmt_tree.children:
        if isinstance(child, Tree):
            if child.data == 'type':
                resolved = _resolve_nested_type(child)
                if resolved is not None:
                    var_type = resolved
                else:
                    var_type = _transform_type(child)
            elif child.data in ('index_expr', 'lhs'):
                lvalue = _transform_lvalue(child)
                if isinstance(lvalue, codegen_ast.Identifier):
                    name = lvalue.name
        elif isinstance(child, Token):
            # Could be the variable name token directly
            name = child.value

    return codegen_ast.Declaration(is_const, var_type or codegen_ast.Int(), name)


# ── public transform entry point ───────────────────────────────────

def transform(t: Tree) -> codegen_ast.Program:
    """Top-level: turn a parsed Lark tree into a Program AST."""
    p = codegen_ast.Program()

    # children: [header, parfor_space, params_list, body_list]
    p.header = extract_header(t.children[0])
    p.space = extract_space_name(t.children[1])
    p.limit_expr = extract_limit_expr(t.children[1])

    for decl_tree in t.children[2].children:
        p.params.append(transform_declaration(decl_tree))

    for stmt_tree in t.children[3].children:
        stmt = transform_statement(stmt_tree)
        if stmt is not None:
            p.body_stmts.append(stmt)

    return p
