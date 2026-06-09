"""Transform Lark trees to codegen_ast nodes.

Extracted from parser.py so that parser.py only contains grammar, parser,
parse() and a thin transform wrapper.
"""

from lark import Tree, Token
from . import ast as codegen_ast


# ── helpers ────────────────────────────────────────────────────────

def _is_token(val):
    return isinstance(val, Token)


def _token_value(t):
    if isinstance(t, Token):
        return t.value
    if isinstance(t, Tree) and len(t.children) == 1:
        child = t.children[0]
        if isinstance(child, Token):
            return child.value
    return None


# ── extractors ─────────────────────────────────────────────────────

def extract_header(header_tree):
    """Extract string value and workgroup properties from header tree.
    
    Header tree has: [string, workgroup_properties*, ...]
    Returns (header_string, list_of_workgroup_properties_trees)
    """
    header_str = ""
    workgroups = []
    for child in header_tree.children:
        if isinstance(child, Tree):
            if child.data == 'string':
                header_str = _token_value(child.children[0])
            elif child.data == 'workgroup_properties':
                workgroups.append(child)
        elif _is_token(child) and child.type == 'ESCAPED_STRING':
            header_str = _token_value(child)
    return (header_str, workgroups)


def extract_loop_vars_and_dim(space_tree):
    """Extract all loop variable names and dimensionality from parfor_space.

    For 1D (OFFLOAD_PARFOR_1D_PARAM(i, ...)): loop_vars=[i], dim=1
    For 2D (OFFLOAD_PARFOR_2D_PARAM(i, j, ...)): loop_vars=[i, j], dim=2
    """
    # IDENTs are the first N children where N = space_dim
    ids = []
    for c in space_tree.children[:2]:
        if _is_token(c) and c.type == 'IDENT':
            ids.append(c.value)

    dim = len(ids)
    if dim > 3:
        dim = 3

    return ids, dim


def _extract_grid_name_from_expr(expr_tree):
    """Extract a single identifier name from an expression tree (grid/extent name in 2D)."""
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'base_expr':
            for sub in child.children:
                if isinstance(sub, Tree) and sub.data == 'lhs':
                    lhs_children = sub.children
                    if lhs_children:
                        first = lhs_children[0]
                        if _is_token(first):
                            return first.value
    # Fallback: scan all children for a lone IDENT token
    for child in expr_tree.children:
        if _is_token(child) and child.type == 'IDENT':
            return child.value
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



def _transform_workgroup_properties(wg_tree):
    """Transform workgroup_properties tree to an AST node.
    
    Two grammar alternatives:
    1. workgroup { x: expr, y: expr, z: expr } -> WorkgroupProperties
    2. shared declaration ; -> SharedDecl
    """
    if len(wg_tree.children) == 0:
        return None
    
    # Detect shared vs workgroup structurally:
    # - workgroup { ... } has 3+ expression children (x, y, z expressions)
    # - shared declaration ; has 1 child (the declaration tree)
    if len(wg_tree.children) == 1 and isinstance(wg_tree.children[0], Tree):
        child = wg_tree.children[0]
        if child.data in ('declaration', 'decl', 'const_decl'):
            # This is the shared alternative
            decl_tree = child
            is_const = child.data == 'const_decl'
            var_type = codegen_ast.Int()
            name = ""
            init_expr = None
            for dc in decl_tree.children:
                if isinstance(dc, Tree):
                    # Check if this child looks like a type (either 'type' or a grammar-rewritten leaf name)
                    is_type_like = dc.data in ('type', 'int', 'float')
                    # Also check for complex types that start with their container name
                    if dc.data.startswith(('fixed_size_vector', 'flexible_rows_matrix', 'fixed_size_matrix')):
                        is_type_like = True
                    if is_type_like:
                        resolved = _resolve_nested_type(dc)
                        var_type = resolved or _transform_type(dc)
                    elif dc.data == 'assignment':
                        init_expr = transform_expression(dc)
                    elif dc.data == 'expression':
                        init_expr = dc
                elif isinstance(dc, Token):
                    name = dc.value
            return codegen_ast.SharedDecl(is_const, var_type, name, init_expr)
    
    # workgroup { x: ..., y: ..., z: ... } case
    expressions = []
    for child in wg_tree.children:
        if isinstance(child, Tree):
            expr_result = transform_expression(child)
            if expr_result is not None:
                expressions.append(expr_result)
    x_expr = expressions[0] if len(expressions) > 0 else None
    y_expr = expressions[1] if len(expressions) > 1 else None
    z_expr = expressions[2] if len(expressions) > 2 else None
    return codegen_ast.WorkgroupProperties(x_expr, y_expr, z_expr)

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
        op_val = None
        
        # Try token-based operator first (compare_operator style)
        if len(children) == 3 and _is_token(children[1]):
            op_val = children[1].value
        # Also handle Tree-based operator (arith_operator, assign_operator) with single Token child
        elif len(children) == 3 and isinstance(children[1], Tree) and len(children[1].children) == 1:
            inner = children[1].children[0]
            if isinstance(inner, Token):
                op_val = inner.value
        
        if op_val is not None:
            left = transform_expression(children[0])
            right = transform_expression(children[2])
            if left is not None and right is not None:
                return codegen_ast.BinaryExpr(left, op_val, right)

        # Simple expression — unwrap and process single child
        for child in children:
            if isinstance(child, Tree):
                result = transform_expression(child)
                if result is not None:
                    return result

    # Direct number tree
    if expr_tree.data == 'number':
        val_token = expr_tree.children[0]
        try:
            val_str = val_token.value
            if isinstance(val_str, str) and '.' in val_str:
                core = val_str.rstrip('fF')  # Remove trailing f/F suffix
                if core.endswith('.'):
                    # Trailing dot with no digits after, e.g. "0." or "16384."
                    core = core + '0'
                return codegen_ast.Number(int(float(core)))
            return codegen_ast.Number(int(val_str))
        except (ValueError, TypeError):
            return None

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
                elif _is_token(child):
                    indices.append(codegen_ast.Identifier(child.value))
            return codegen_ast.IndexedIdentifier(base, indices)

        # Nested structure — recurse
        return _transform_lvalue(first)

    # base_expr (already handled by _transform_from_base in most cases)
    if data == 'base_expr':
        return _transform_from_base(lhs_tree)

    return None


# ── type transform ─────────────────────────────────────────────────

def _transform_type(type_tree):
    """Transform a 'type' Lark tree into an AST Type node.

    Grammar rules for multi-param types:
      fixed_size_vector<type, expression>&
      flexible_rows_matrix<type, expression, expression>&
      fixed_size_matrix<type, expression, expression>&

    The type name token is consumed by the grammar; we infer the kind
    from the number of children and whether they are Types or Expressions.
    Note: flexible_rows_matrix and fixed_size_matrix produce identical trees
    (both have 3 children), so we default to FlexibleRowsMatrix for 3-param.
    """
    data = type_tree.data
    children = type_tree.children
    if data != 'type' or len(children) < 2:
        return None

    elem_type = _resolve_nested_type(children[0])
    if elem_type is None:
        return None

    # 3-param types: flexible_rows_matrix or fixed_size_matrix
    if len(children) == 3:
        first_is_expr = isinstance(children[1], Tree) and children[1].data == 'expression'
        second_is_expr = isinstance(children[2], Tree) and children[2].data == 'expression'

        if first_is_expr and second_is_expr:
            # Both are expression trees → flexible_rows_matrix or fixed_size_matrix
            row_size_expr = transform_expression(children[1])
            col_size_expr = transform_expression(children[2])
            return codegen_ast.FlexibleRowsMatrix(elem_type, row_size_expr, col_size_expr)

    # 2-param types: fixed_size_vector<type, expression>&
    elif len(children) == 2:
        second = children[1]
        if isinstance(second, Tree):
            if second.data == 'expression':
                return codegen_ast.FixedSizeVector(elem_type, transform_expression(second))
            else:
                # Fallback: treat as type (shouldn't happen with current grammar)
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
        op_tree = stmt_tree.children[1]
        rhs_part = stmt_tree.children[2]

        if isinstance(lhs_part, Tree):
            lvalue = _transform_lvalue(lhs_part)
        else:
            lvalue = codegen_ast.Identifier(
                _token_value(lhs_part) or str(lhs_part))

        # Extract operator value (now a Token wrapped in assign_operator tree)
        op_val = _token_value(op_tree)  # Now finds the inner Token via Tree with single child
        assign_op = op_val if op_val else '='
        rvalue = transform_expression(rhs_part)
        return codegen_ast.Assignment(lvalue, assign_op, rvalue)

    # For statement: for "(" for_loop_var ";" condition ";" increment ")" block
    if data == 'for_statement':
        return _transform_for(stmt_tree)

    # If statement: if "(" condition ")" block
    if data == 'if_statement':
        return _transform_if(stmt_tree)

    # Standalone declarations
    if data in ('decl', 'const_decl'):
        return transform_declaration(stmt_tree)

    return None


def _extract_condition(cond_tree):
    """Extract a Condition AST node from a condition tree.

    condition: IDENT compare_operator expression
    """
    lhs_expr = None
    op = ""
    rhs_expr = None

    for c in cond_tree.children:
        if isinstance(c, Token) and c.type == 'IDENT':
            lhs_expr = codegen_ast.Identifier(c.value)
        elif isinstance(c, Tree) and c.data == 'compare_operator':
            # compare_operator has one child: the operator token (LT/GT/etc.)
            for sub in c.children:
                if isinstance(sub, Token):
                    op = sub.value
                    break
        elif isinstance(c, Tree):
            rhs_expr = transform_expression(c)

    return codegen_ast.Condition(lhs_expr, op, rhs_expr)


def _transform_for(for_tree):
    """Transform a for_statement tree into a For AST node."""
    # Structure: [for_loop_var, condition, increment, block]
    for_loop_var_tree = for_tree.children[0]
    cond_tree = for_tree.children[1]
    inc_tree = for_tree.children[2]
    block_tree = for_tree.children[3]

    # Parse loop variable: type IDENT "=" expression
    loop_var_type = None
    loop_var_name = ""
    init_expr = None
    for child in for_loop_var_tree.children:
        if isinstance(child, Tree) and child.data == 'type':
            loop_var_type = _resolve_nested_type(child) or _transform_type(child)
        elif isinstance(child, Token):
            loop_var_name = child.value

    # Parse condition
    condition = _extract_condition(cond_tree)

    # Parse increment: supports both prefix (++i) and postfix (i++)
    increment_var = ""
    increment_op = ""
    if isinstance(inc_tree, Tree):
        inc_data = inc_tree.data
        if inc_data in ('increment', 'inc_postfix', 'inc_prefix'):
            tokens = [c for c in inc_tree.children if isinstance(c, Token)]
            for t in tokens:
                val = t.value
                if val in ('++', '--'):
                    increment_op = val
                elif t.type == 'IDENT':
                    increment_var = val

    # Parse body block
    body_stmts = []
    if isinstance(block_tree, Tree) and block_tree.data == 'block':
        for stmt_child in block_tree.children:
            stmt = transform_statement(stmt_child)
            if stmt is not None:
                body_stmts.append(stmt)

    return codegen_ast.For(loop_var_type, loop_var_name, condition,
                          increment_var, increment_op, body_stmts)


def _transform_if(if_tree):
    """Transform an if_statement tree into an If AST node."""
    cond_tree = if_tree.children[0]
    block_tree = if_tree.children[1]

    # Parse condition
    condition = _extract_condition(cond_tree)

    # Parse body block
    body_stmts = []
    if isinstance(block_tree, Tree) and block_tree.data == 'block':
        for stmt_child in block_tree.children:
            stmt = transform_statement(stmt_child)
            if stmt is not None:
                body_stmts.append(stmt)

    return codegen_ast.If(condition, body_stmts)


def transform_declaration(stmt_tree):
    """Transform a declaration tree into Declaration AST."""
    is_const = stmt_tree.data == 'const_decl'
    var_type = None
    name = ""
    init_expr = None

    for child in stmt_tree.children:
        if isinstance(child, Tree):
            # Simple types (int/float from grammar rewrite rules like "float" -> float)
            if child.data in ('int', 'float'):
                var_type = _resolve_nested_type(child) or codegen_ast.Int()
            elif child.data == 'type':
                resolved = _resolve_nested_type(child)
                if resolved is not None:
                    var_type = resolved
                else:
                    var_type = _transform_type(child)
            elif child.data == 'expression':
                init_expr = transform_expression(child)
            elif child.data in ('index_expr', 'lhs'):
                lvalue = _transform_lvalue(child)
                if isinstance(lvalue, codegen_ast.Identifier):
                    name = lvalue.name
        elif isinstance(child, Token):
            # Could be the variable name token directly
            name = child.value

    return codegen_ast.Declaration(is_const, var_type or codegen_ast.Int(), name, init_expr)


# ── public transform entry point ───────────────────────────────────


def transform(t: Tree) -> codegen_ast.Program:
    """Top-level: turn a parsed Lark tree into a Program AST."""
    p = codegen_ast.Program()

    # children: [header, parfor_space, params_list, body_list]
    header_str, wg_trees = extract_header(t.children[0])
    p.header = header_str
    
    for wg_tree in wg_trees:
        p.workgroups.append(_transform_workgroup_properties(wg_tree))
    
    # Extract all loop variable names, dimensionality, and grid name from parfor_space
    p.loop_vars, p.space_dim = extract_loop_vars_and_dim(t.children[1])
    
    # Try to get the grid/extent name (only relevant for 2D)
    if p.space_dim >= 2:
        for c in t.children[1].children:
            if isinstance(c, Tree) and c.data == 'expression':
                p.grid_name = _extract_grid_name_from_expr(c)
                break
    
    # Try to find dispatch size expression from the expression inside parfor_space
    # (e.g., the "length" part of limit<16384>(length))
    for c in t.children[1].children:
        if isinstance(c, Tree) and c.data == 'expression':
            p.dispatch_size_expr = transform_expression(c)
            break

    # Build name-to-declaration map from parameters_list
    param_names = []
    for c in t.children[1].children:
        if isinstance(c, Tree) and c.data == 'parfor_parameters_list':
            for pc in c.children:
                if isinstance(pc, Tree) and pc.data == 'parameter':
                    for tc in pc.children:
                        if isinstance(tc, Token) and tc.type == 'IDENT':
                            param_names.append(tc.value)

    # Build name-to-type map from the type declarations (for FixedSizeVector etc.)
    param_type_map = {}
    for decl_tree in t.children[2].children:
        var_name = ""
        if isinstance(decl_tree, Tree):
            for dc in decl_tree.children:
                if isinstance(dc, Token) and dc.type == 'IDENT':
                    # Find a type-decl pair: type token followed by IDENT
                    idx = decl_tree.children.index(dc)
                    if idx > 0:
                        prev = decl_tree.children[idx-1]
                        if isinstance(prev, Tree) and prev.data in ('type', 'int', 'float'):
                            var_name = dc.value
        param_type_map[var_name] = transform_declaration(decl_tree)

    p.params = [param_type_map.get(n) for n in param_names]

    p.limit_expr = extract_limit_expr(t.children[1])

    for stmt_tree in t.children[3].children:
        stmt = transform_statement(stmt_tree)
        if stmt is not None:
            p.body_stmts.append(stmt)

    return p
