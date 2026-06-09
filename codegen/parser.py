from lark import Lark, Tree, Token
import codegen_ast


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def _is_token(val):
    return isinstance(val, Token)


def _token_value(t):
    if isinstance(t, Token):
        return t.value
    return None


def transform(t: Tree) -> codegen_ast.Program:
    p = codegen_ast.Program()

    # program children: [header, parfor_space, parfor_parameters, body]
    p.header = _extract_header(t.children[0])
    p.space = _extract_space_name(t.children[1])
    p.limit_expr = _extract_limit_expr(t.children[1])

    for decl_tree in t.children[2].children:
        p.params.append(_transform_declaration(decl_tree))

    for stmt_tree in t.children[3].children:
        stmt = _transform_statement(stmt_tree)
        if stmt is not None:
            p.body_stmts.append(stmt)

    return p


def _extract_header(header_tree):
    """Extract string value from header tree."""
    string_tree = header_tree.children[0]
    return _token_value(string_tree.children[0])  # ESCAPED_STRING token


def _extract_space_name(space_tree):
    """Extract loop variable name (first IDENT child of parfor_space)."""
    return space_tree.children[0].value


def _extract_limit_expr(space_tree):
    """Extract limit expression from parfor_space.

    The parfor_space tree has children [IDENT, expression(limit_expr...), parameters_list].
    Child[1] is the expression wrapper whose child is the limit_expr tree.
    """
    expr_tree = space_tree.children[1]
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'limit_expr':
            return _transform_expression(child)
    return None


def _transform_expression(expr_tree) -> codegen_ast.Expression:
    """Convert an expression Tree to an AST node.

    Handles:
      - limit_expr tree (from limit<max>(body)) -> LimitExpr
      - 'number' tree (NUMBER alias) -> Number
      - 'expression' wrapper (from various rules) -> unwrap and process child
      - lhs (identifier with optional index) -> Identifier or IndexedIdentifier
    """
    # Handle direct limit_expr (or from expression → limit_expr via recursion below)
    if expr_tree.data == 'limit_expr':
        max_child = expr_tree.children[0]
        body_child = expr_tree.children[1]
        max_val = _transform_expression(max_child)
        body = _transform_expression(body_child)
        return codegen_ast.LimitExpr(max_val, body)

    # Handle 'expression' wrapper (e.g., expression → number or expression → lhs)
    if expr_tree.data == 'expression':
        for child in expr_tree.children:
            if isinstance(child, Tree):
                result = _transform_expression(child)
                if result is not None:
                    return result

    # Check if this is a direct number tree (from NUMBER -> number)
    if expr_tree.data == 'number':
        val_token = expr_tree.children[0]
        return codegen_ast.Number(int(val_token.value))

    # Handle bare lhs trees directly (e.g., when expression expands to lhs)
    if expr_tree.data == 'lhs':
        return _transform_lvalue(expr_tree)

    # Check for lhs nested inside a wrapper tree
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == 'lhs':
            return _transform_lvalue(child)

    return None


def _extract_bracket_expressions(lhs_children):
    """Extract index expressions from bracket parts of an lhs tree.

    Skips string terminals like '[' and ',' and collects expression subtrees.
    """
    expressions = []
    i = 1  # Start after the base IDENT (children[0])
    while i < len(lhs_children):
        child = lhs_children[i]
        if isinstance(child, Tree) and child.data == 'expression':
            expr = _transform_expression(child)
            if expr is not None:
                expressions.append(expr)
        i += 1
    return expressions


def _transform_lvalue(lhs_tree) -> codegen_ast.Expression:
    """Convert a lhs Tree to an Expression.

    Returns:
      - Identifier for plain identifiers (e.g., "dst")
      - IndexedIdentifier for indexed access (e.g., "dst[i]")
    """
    base_name = None
    children = lhs_tree.children

    # Extract the base identifier name (first child is always IDENT)
    if children:
        first = children[0]
        if _is_token(first):
            base_name = first.value
        elif isinstance(first, Tree):
            base_name = str(first.data)
        else:
            base_name = str(first)

    if not base_name:
        return codegen_ast.Identifier("unknown")

    # Check for bracket expressions (index access)
    indices = _extract_bracket_expressions(children)

    if indices:
        return codegen_ast.IndexedIdentifier(
            codegen_ast.Identifier(base_name), indices
        )

    return codegen_ast.Identifier(base_name)


def _transform_declaration(decl_tree) -> codegen_ast.Declaration:
    """Transform a decl or const_decl Tree to an AST node.

    The grammar rule is: type IDENT -> decl
    For simple types like "int", the alias "-> int" means we get Tree(data='int') directly
    instead of Tree(data='type'). For complex types, we get Tree(data='type').
    """
    is_const = decl_tree.data == 'const_decl'

    var_type = None
    name_token = None

    for child in decl_tree.children:
        if isinstance(child, Tree):
            data = child.data
            if data == 'type':
                var_type = _transform_type(child)
            elif data == 'int':
                var_type = codegen_ast.Int()
            elif data == 'float':
                var_type = codegen_ast.Float()
        elif _is_token(child):
            name_token = child

    name = name_token.value if name_token else str(name_token)
    return codegen_ast.Declaration(is_const, var_type, name)


def _transform_type(type_tree) -> codegen_ast.Type:
    """Transform a type Tree to an AST Type.

    Handles the grammar rules:
      "int" -> int
      "float" -> float
      "fixed_size_vector" "<" type "," expression ">" "&"

    For base types like `int`/`float` appearing via aliasing, we get Tree(data='int') or
    Tree(data='float'). For complex types, the rule name is 'type' and its children are
    the nested type + size expression from fixed_size_vector.
    """
    data = type_tree.data
    if data == 'int':
        return codegen_ast.Int()
    elif data == 'float':
        return codegen_ast.Float()

    # Handle Tree(data='type') which comes from the fixed_size_vector alternative:
    # "fixed_size_vector" "<" type "," expression ">" "&"
    if data == 'type' and len(type_tree.children) >= 2:
        elem_type = None
        size_expr = None
        for sub in type_tree.children:
            if isinstance(sub, Tree):
                if sub.data in ('int', 'float'):
                    # This is the nested type (the element type of the vector)
                    if sub.data == 'int':
                        elem_type = codegen_ast.Int()
                    elif sub.data == 'float':
                        elem_type = codegen_ast.Float()
                elif sub.data == 'type':
                    elem_type = _transform_type(sub)
                elif sub.data == 'expression':
                    size_expr = _transform_expression(sub)
            elif _is_token(sub):
                if sub.value == 'fixed_size_vector':
                    # Terminal found - this is definitely a fixed_size_vector
                    pass

        # If we have both an element type and a size expression, it's a vector
        if elem_type is not None and size_expr is not None:
            return codegen_ast.FixedSizeVector(elem_type, size_expr)

    return None


def _transform_statement(stmt_tree) -> codegen_ast.Statement:
    """Transform a statement Tree to an AST node."""
    data = stmt_tree.data

    # Unwrap "statement" wrapper if present
    if data == 'statement':
        return _transform_statement(stmt_tree.children[0])

    # Assignment: lhs "=" expression (token "=" stripped by Lark)
    if data == 'assignment':
        lhs_part = stmt_tree.children[0]
        rhs_part = stmt_tree.children[1]

        if isinstance(lhs_part, Tree):
            lvalue = _transform_lvalue(lhs_part)
        else:
            lvalue = codegen_ast.Identifier(
                _token_value(lhs_part) or str(lhs_part))

        rvalue = _transform_expression(rhs_part)
        return codegen_ast.Assignment(lvalue, rvalue)

    # For/if statements - placeholder
    if data in ('for_statement', 'if_statement'):
        return None

    # Standalone declarations
    if data in ('decl', 'const_decl'):
        is_const = data == 'const_decl'
        var_type = codegen_ast.Int()
        name_token = None
        for child in stmt_tree.children:
            if _is_token(child):
                name_token = child
        name = name_token.value if name_token else str(name_token)
        return codegen_ast.Declaration(is_const, var_type, name)

    return None


grammar = read_file("grammar.lark")

parser = Lark(grammar, start="program")


def parse(filename: str):
    print(f"--------------- parsing: {filename} -----------------")

    text = read_file(filename)
    ret = parser.parse(text)
    #print(ret.pretty())

    program = transform(ret)

    from codegen_ast import PrettyPrinter
    printer = PrettyPrinter()
    output = program.visit_children(printer)
    print(output)


parse("testdata/single-assign.kernel")
parse("testdata/multi-arg.kernel")
