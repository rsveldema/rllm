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
    """Extract all loop variable names and dimensionality from a parfor_space tree.

    For 1D (OFFLOAD_PARFOR_1D_PARAM(i, ...)): loop_vars=[i], dim=1
    For 2D (OFFLOAD_PARFOR_2D_PARAM(i, j, ...)): loop_vars=[i, j], dim=2
    For 3D triangular (OFFLOAD_PARFOR_3D_TRIANGULAR_PARAM(hi, i, j, ...)):
        loop_vars=[hi, i, j], dim=3
    """
    ids = []
    for c in space_tree.children:
        if _is_token(c) and c.type == 'IDENT':
            ids.append(c.value)
            if len(ids) >= 3:
                break

    dim = min(len(ids), 3)
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
    for child in expr_tree.children:
        if _is_token(child) and child.type == 'IDENT':
            return child.value
    return ""


def _find_expression_trees(space_tree):
    """Find ALL expression children in a parfor_space tree (before parfor_parameters_list)."""
    results = []
    for child in space_tree.children:
        if isinstance(child, Tree) and child.data == 'expression':
            results.append(child)
    return results


def _find_expression_tree(space_tree):
    """Find the first expression child in a parfor_space tree (works for 1D)."""
    trees = _find_expression_trees(space_tree)
    return trees[0] if trees else None


def extract_limit_expr(space_tree):
    """Extract limit or triangular bound expressions from parfor_space.

    For 1D/2D (single expression): returns a single expression AST node.
    For 3D triangular (two expressions): returns a tuple of (lower_bound, upper_bound).
    """
    expr_trees = _find_expression_trees(space_tree)
    if not expr_trees:
        return None

    if len(expr_trees) == 2:
        # Triangular case: two bounds (lower and upper)
        lower = transform_expression(expr_trees[0])
        upper = transform_expression(expr_trees[1])
        return (lower, upper) if lower is not None and upper is not None else None

    expr_tree = expr_trees[0]

    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'limit_expr':
            return transform_expression(child)

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
    2. shared declaration ";" -> SharedDecl
    """
    if wg_tree.data == 'workgroup':
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
    
    # shared declaration case
    if isinstance(wg_tree.children[0], Tree):
        decl_tree = wg_tree.children[0]
        is_const = decl_tree.data == 'const_decl'
        var_type = None
        name = ""
        init_expr = None
        
        for child in decl_tree.children:
            if isinstance(child, Tree):
                data = child.data
                if data in ('int', 'float'):
                    var_type = _resolve_nested_type(child) or codegen_ast.Int()
                elif data == 'type':
                    resolved = _resolve_nested_type(child)
                    if resolved is not None:
                        var_type = resolved
                    else:
                        var_type = _transform_type(child)
                elif data == 'expression':
                    init_expr = transform_expression(child)
                elif data in ('index_expr', 'lhs'):
                    lvalue = _transform_lvalue(child)
                    if isinstance(lvalue, codegen_ast.Identifier):
                        name = lvalue.name
            elif isinstance(child, Token):
                name = child.value
        
        return codegen_ast.SharedDecl(is_const, var_type or codegen_ast.Int(), name, init_expr)

    # workgroup { x: ..., y: ..., z: ... } case (from grammar tree)
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

    # 'expression' wrapper with binary ops
    if expr_tree.data == 'expression':
        children = expr_tree.children
        op_val = None
        
        if len(children) == 3 and _is_token(children[1]):
            op_val = children[1].value
        elif len(children) == 3 and isinstance(children[1], Tree) and len(children[1].children) == 1:
            inner = children[1].children[0]
            if isinstance(inner, Token):
                op_val = inner.value
        
        if op_val is not None:
            left = transform_expression(children[0])
            right = transform_expression(children[2])
            if left is not None and right is not None:
                return codegen_ast.BinaryExpr(left, op_val, right)

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
                core = val_str.rstrip('fF')
                if core.endswith('.'):
                    core = core + '0'
                return codegen_ast.Number(int(float(core)))
            return codegen_ast.Number(int(val_str))
        except (ValueError, TypeError):
            return None

    # Handle base_expr directly
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
                result = transform_expression(child)
                if result is not None:
                    return result

        elif _is_token(child):
            return codegen_ast.Identifier(child.value)

    return None


def _transform_lvalue(lhs_tree):
    """Transform an LHS tree into an AST lvalue expression."""
    if _is_token(lhs_tree):
        return codegen_ast.Identifier(lhs_tree.value)

    data = lhs_tree.data

    if data == 'lhs':
        first = lhs_tree.children[0]
        if _is_token(first) and len(lhs_tree.children) == 1:
            return codegen_ast.Identifier(first.value)

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

        return _transform_lvalue(first)

    if data == 'base_expr':
        return _transform_from_base(lhs_tree)

    return None


# ── type transform ─────────────────────────────────────────────────


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


def _transform_type(type_tree):
    """Transform a 'type' Lark tree into an AST Type node.

    After the grammar rewrite with explicit type name terminals (FSV, FRM, FSM,
    FSLRMC, FRCLM), each multi-param type produces a Tree(data='type', children=[...])
    where the first child is the type name token and remaining children are:
      elem_type + expression params.
    """
    data = type_tree.data
    children = type_tree.children
    if data != 'type' or len(children) < 2:
        return None

    first_child = children[0]

    # Check if this is a multi-param type (first child is terminal token)
    if _is_token(first_child):
        type_name = first_child.value
        
        expr_children = [c for c in children[1:] if isinstance(c, Tree) and c.data == 'expression']
        
        if 'fixed_size_levels_rows_cols_matrix' in type_name:
            # 4 params: elem_type + level + row + col
            if len(expr_children) >= 3:
                return codegen_ast.FixedSizeLevelsRowsColsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                    transform_expression(expr_children[2])
                )
        elif 'flexible_rows_cols_levels_matrix' in type_name:
            # 4 params: elem_type + level + row + col
            if len(expr_children) >= 3:
                return codegen_ast.FlexibleRowsColsLevelsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                    transform_expression(expr_children[2])
                )
        elif 'fixed_size_matrix' in type_name:
            # 3 params: elem_type + row + col
            if len(expr_children) >= 2:
                return codegen_ast.FixedSizeMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1])
                )
        elif 'flexible_rows_matrix' in type_name:
            # 3 params: elem_type + row + col
            if len(expr_children) >= 2:
                return codegen_ast.FlexibleRowsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1])
                )
        elif 'fixed_size_vector' in type_name:
            # 2 params: elem_type + size_expr
            if len(expr_children) >= 1:
                return codegen_ast.FixedSizeVector(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0])
                )
        return None

    # Simple types (data like 'int' or 'float')
    if first_child.data in ('int', 'float'):
        return codegen_ast.Int() if first_child.data == 'int' else codegen_ast.Float()

    if first_child.data == 'type':
        return _transform_type(first_child)

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


def transform_declaration(stmt_tree):
    """Transform a declaration tree into Declaration AST."""
    is_const = stmt_tree.data == 'const_decl'
    var_type = None
    name = ""
    init_expr = None
    
    extended_type_prefixes = (
        'fixed_size_vector', 'flexible_rows_matrix', 'fixed_size_matrix',
        'fixed_size_levels_rows_cols_matrix', 'flexible_rows_cols_levels_matrix'
    )

    for child in stmt_tree.children:
        if isinstance(child, Tree):
            data = child.data
            # Simple types (int/float from grammar rewrite rules)
            if data in ('int', 'float'):
                var_type = _resolve_nested_type(child) or codegen_ast.Int()
            elif data == 'type':
                resolved = _resolve_nested_type(child)
                if resolved is not None:
                    var_type = resolved
                else:
                    var_type = _transform_type(child)
            elif child.data.startswith(extended_type_prefixes):
                # Complex type as direct child (shouldn't happen normally, but handle it)
                var_type = _transform_type(child)
            elif data == 'expression':
                init_expr = transform_expression(child)
            elif data in ('index_expr', 'lhs'):
                lvalue = _transform_lvalue(child)
                if isinstance(lvalue, codegen_ast.Identifier):
                    name = lvalue.name
        elif isinstance(child, Token):
            # Could be the variable name token directly
            name = child.value

    return codegen_ast.Declaration(is_const, var_type or codegen_ast.Int(), name, init_expr)


def transform_statement(stmt_tree):
    """Transform a statement Tree to an AST node."""
    data = stmt_tree.data

    if data == 'statement':
        return transform_statement(stmt_tree.children[0])

    # Overflow check: 2 children [lhs, expression]
    if _is_overflow_check(stmt_tree):
        lhs_child, expr_child = stmt_tree.children
        lvalue = _transform_lvalue(lhs_child)
        operand = transform_expression(expr_child)
        if lvalue is not None and operand is not None:
            return codegen_ast.OverflowCheck(lvalue, operand)

    # block: "{" statement* "}"
    if data == 'block':
        body_stmts = []
        for stmt_child in stmt_tree.children:
            stmt = transform_statement(stmt_child)
            if stmt is not None:
                body_stmts.append(stmt)
        return codegen_ast.Statement() if not body_stmts else body_stmts[0]

    # for_statement
    if data == 'for_statement':
        # Two grammar alternatives:
        # 1. "for" "(" for_loop_var ";" condition ";" increment ")" (block | statement)
        # 2. "for" "(" "const" type IDENT ":" expression ")" (block | statement)
        body_stmts = []
        
        # Handle loop variable part (children[0] = for_loop_var or "const" type+IDENT)
        for_loop_part = stmt_tree.children[0]
        if isinstance(for_loop_part, Tree):
            for_lp_data = for_loop_part.data
            if for_lp_data == 'for_loop_var':
                # for_loop_var -> (const)? type IDENT "=" expression
                loop_var_type = None
                loop_var_name = ""
                init_expr_for_var = None
                
                for child in for_loop_part.children:
                    if isinstance(child, Tree):
                        data = child.data
                        if data == 'type':
                            loop_var_type = _resolve_nested_type(child) or _transform_type(child)
                        elif data == 'expression':
                            init_expr_for_var = transform_expression(child)
                    elif _is_token(child):
                        loop_var_name = child.value
                
                # condition from children[1]
                condition_tree = stmt_tree.children[1] if len(stmt_tree.children) > 1 else None
                condition = None
                if isinstance(condition_tree, Tree) and condition_tree.data == 'condition':
                    lhs_t = condition_tree.children[0]
                    rhs_t = condition_tree.children[2]
                    lhs = _transform_lvalue(lhs_t) if _is_token(lhs_t) or (isinstance(lhs_t, Tree) and lhs_t.data in ('lhs',)) else None
                    op_val = ""
                    for c in condition_tree.children:
                        if _is_token(c):
                            op_val = c.value
                        elif isinstance(c, Tree) and len(c.children) == 1:
                            inner = c.children[0]
                            if _is_token(inner):
                                op_val = inner.value
                    
                    rhs = transform_expression(rhs_t) if isinstance(rhs_t, Tree) else None
                    if lhs is not None and rhs is not None:
                        condition = codegen_ast.Condition(lhs, op_val, rhs)
                
                # increment from children[2]
                inc_tree = stmt_tree.children[2] if len(stmt_tree.children) > 2 else None
                inc_var = ""
                inc_op = ""
                if isinstance(inc_tree, Tree) and inc_tree.data == 'increment':
                    for c in inc_tree.children:
                        if _is_token(c) and c.type == 'IDENT':
                            inc_var = c.value
                        elif _is_token(c) and c.type == 'INC_OP':
                            inc_op = c.value
                
                # body from children[3] (block or statement)
                body_tree = stmt_tree.children[3] if len(stmt_tree.children) > 3 else None
                if isinstance(body_tree, Tree):
                    if body_tree.data == 'block':
                        for bc in body_tree.children:
                            s = transform_statement(bc)
                            if s is not None:
                                body_stmts.append(s)
                    else:
                        s = transform_statement(body_tree)
                        if s is not None:
                            body_stmts.append(s)
                
                return codegen_ast.For(loop_var_type, loop_var_name, condition, inc_var, inc_op, body_stmts)
            
            elif for_lp_data == 'for':
                # Alternative 2: "for" "(" "const" type IDENT ":" expression ")"
                for child in for_loop_part.children:
                    if isinstance(child, Tree):
                        data = child.data
                        if data == 'type':
                            loop_var_type = _resolve_nested_type(child) or _transform_type(child)
                        elif data == 'condition':
                            # Handle condition: IDENT compare_operator expression
                            pass
                        elif data == 'expression':
                            condition = transform_expression(child)
                    elif _is_token(child):
                        loop_var_name = child.value
                
                body_tree = stmt_tree.children[1] if len(stmt_tree.children) > 1 else None
                if isinstance(body_tree, Tree):
                    if body_tree.data == 'block':
                        for bc in body_tree.children:
                            s = transform_statement(bc)
                            if s is not None:
                                body_stmts.append(s)
                    else:
                        s = transform_statement(body_tree)
                        if s is not None:
                            body_stmts.append(s)
                
                return codegen_ast.For(loop_var_type, loop_var_name or "", condition, "", "", body_stmts)

    # if_statement
    if data == 'if_statement':
        condition_tree = stmt_tree.children[0] if len(stmt_tree.children) > 0 else None
        condition = None
        if isinstance(condition_tree, Tree) and condition_tree.data == 'condition':
            lhs_t = condition_tree.children[0]
            rhs_t = condition_tree.children[2]
            op_val = ""
            for c in condition_tree.children:
                if _is_token(c):
                    op_val = c.value
                elif isinstance(c, Tree) and len(c.children) == 1:
                    inner = c.children[0]
                    if _is_token(inner):
                        op_val = inner.value
            
            lhs = _transform_lvalue(lhs_t) if _is_token(lhs_t) or (isinstance(lhs_t, Tree) and (lhs_t.data == 'lhs' or True)) else None
            rhs = transform_expression(rhs_t) if isinstance(rhs_t, Tree) else None
            if lhs is not None and rhs is not None:
                condition = codegen_ast.Condition(lhs, op_val, rhs)

        body_stmts = []
        if len(stmt_tree.children) > 1:
            body_tree = stmt_tree.children[1]
            if isinstance(body_tree, Tree):
                if body_tree.data == 'block':
                    for bc in body_tree.children:
                        s = transform_statement(bc)
                        if s is not None:
                            body_stmts.append(s)
                else:
                    s = transform_statement(body_tree)
                    if s is not None:
                        body_stmts.append(s)

        return codegen_ast.If(condition, body_stmts)

    # declaration;
    if data == 'decl' or data == 'const_decl':
        decl = transform_declaration(stmt_tree)
        return decl

    # assignment;
    if data == 'assignment':
        assign_op = ""
        lvalue = None
        rvalue = None
        
        for child in stmt_tree.children:
            if isinstance(child, Tree):
                if child.data == 'lhs':
                    lvalue = _transform_lvalue(child)
                elif child.data == 'expression':
                    rvalue = transform_expression(child)
            elif _is_token(child):
                assign_op = child.value
        
        return codegen_ast.Assignment(lvalue, assign_op, rvalue)

    # SharedDecl from workgroup "shared" alternative (handled in _transform_workgroup_properties)
    # block statement within body
    if data == 'block':
        body_stmts = []
        for stmt_child in stmt_tree.children:
            stmt = transform_statement(stmt_child)
            if stmt is not None:
                body_stmts.append(stmt)
        return codegen_ast.Statement() if not body_stmts else body_stmts[0]

    return None


# ── public transform entry point ───────────────────────────────────


def transform(t: Tree) -> codegen_ast.Program:
    """Top-level: turn a parsed Lark tree into a Program AST."""
    p = codegen_ast.Program()

    # children: [header, parfor_space, params_list, body_list]
    header_str, wg_trees = extract_header(t.children[0])
    p.header = header_str
    
    for wg_tree in wg_trees:
        p.workgroups.append(_transform_workgroup_properties(wg_tree))
    
    p.loop_vars, p.space_dim = extract_loop_vars_and_dim(t.children[1])
    
    limit_result = extract_limit_expr(t.children[1])
    if isinstance(limit_result, tuple):
        p.lower_bound_expr = limit_result[0]
        p.upper_bound_expr = limit_result[1]

    if not hasattr(p, 'lower_bound_expr'):
        if p.space_dim >= 2:
            for c in t.children[1].children:
                if isinstance(c, Tree) and c.data == 'expression':
                    p.grid_name = _extract_grid_name_from_expr(c)
                    break

    if not hasattr(p, 'lower_bound_expr'):
        expr_tree = _find_expression_tree(t.children[1])
        if expr_tree is not None:
            p.dispatch_size_expr = transform_expression(expr_tree)

    # Build name-to-declaration map from parameters_list
    param_names = []
    for c in t.children[1].children:
        if isinstance(c, Tree) and c.data == 'parfor_parameters_list':
            for pc in c.children:
                if isinstance(pc, Tree) and pc.data == 'parameter':
                    for tc in pc.children:
                        if isinstance(tc, Token) and tc.type == 'IDENT':
                            param_names.append(tc.value)

    # Build name-to-type map from the type declarations
    extended_type_prefixes = (
        'fixed_size_vector', 'flexible_rows_matrix', 'fixed_size_matrix',
        'fixed_size_levels_rows_cols_matrix', 'flexible_rows_cols_levels_matrix'
    )
    
    param_type_map = {}
    for decl_tree in t.children[2].children:
        var_name = ""
        if isinstance(decl_tree, Tree):
            for dc in decl_tree.children:
                if isinstance(dc, Token) and dc.type == 'IDENT':
                    idx = decl_tree.children.index(dc)
                    if idx > 0:
                        prev = decl_tree.children[idx-1]
                        if isinstance(prev, Tree):
                            prev_data = prev.data
                            if prev_data in ('type', 'int', 'float') or prev_data.startswith(extended_type_prefixes):
                                var_name = dc.value
        param_type_map[var_name] = transform_declaration(decl_tree)

    p.params = [param_type_map.get(n) for n in param_names]

    if not isinstance(limit_result, tuple):
        p.limit_expr = limit_result

    for stmt_tree in t.children[3].children:
        stmt = transform_statement(stmt_tree)
        if stmt is not None:
            p.body_stmts.append(stmt)

    return p


