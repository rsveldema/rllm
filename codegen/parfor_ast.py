"""AST for parsed OFFLOAD_PARFOR dump files.

Dump files are produced by _write_parfor_dump() in offloadize_common.py
and look like:

    // === PARFOR block from src/File.cc:42 ===

    OFFLOAD_PARFOR_2D_PARAM(i, t, partial_grid, (x, row_sums))

    // === OFFLOAD_PARAMETERS ===
    const auto& x = ...;
    std::vector<float> row_sums;

    // === PARFOR BODY ===
    row_sums[t] = 0.0f;
    for (i = 0; i < n; ++i) {
        row_sums[t] += data[i][t];
    }

Usage:

    from parfor_ast import parse_dump_file
    dump = parse_dump_file("build/parfor_dumps/offload_parfor_OutputLayer_60.cc")
    print(dump.macro.iteration_vars)   # ['i', 't']
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Macro kind enumeration
# ---------------------------------------------------------------------------

class MacroKind(Enum):
    PARFOR_1D = "OFFLOAD_PARFOR_1D_PARAM"
    PARFOR_2D = "OFFLOAD_PARFOR_2D_PARAM"
    PARFOR_3D = "OFFLOAD_PARFOR_3D_PARAM"
    PARFOR_3D_TRIANGULAR = "OFFLOAD_PARFOR_3D_TRIANGULAR_PARAM"
    PARFOR_2D_TRIANGULAR = "OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM"
    PARFOR_2D_UPPER_TRIANGULAR = "OFFLOAD_PARFOR_2D_UPPER_TRIANGULAR_PARAM"

    @property
    def dimension(self) -> int:
        if self == MacroKind.PARFOR_1D:
            return 1
        elif self in (MacroKind.PARFOR_2D, MacroKind.PARFOR_2D_TRIANGULAR, MacroKind.PARFOR_2D_UPPER_TRIANGULAR):
            return 2
        else:
            return 3


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class ParforMacro:
    """Parsed OFFLOAD_PARFOR_*_PARAM invocation."""
    kind: MacroKind
    iteration_vars: list[str]           # e.g. ['i', 't'] or ['i', 'j', 'k']
    range_expression: str               # e.g. 'partial_grid' or 'enum_iterator<TokenID>()'
    extra_params: tuple[str, ...] = ()  # deps / arguments in the trailing paren group

    @property
    def dim(self) -> int:
        return self.kind.dimension


@dataclass
class OffloadParamDeclaration:
    """A single declaration from the OFFLOAD_PARAMETERS section."""
    type_str: str    # e.g. 'const auto&', 'std::vector<float>'
    name: str        # e.g. 'x', 'row_sums'
    init_expr: str | None = None  # optional initializer


@dataclass
class Statement:
    """Base class for body statements."""


@dataclass
class Expression(Statement):
    """An expression statement (typically an assignment or function call)."""
    source_line: int        # 1-based line in the dump file
    kind: str               # 'assignment', 'increment', 'decrement', 'call', 'expr'
    target: str | None = None  # for assignments: left-hand side

    def visit(self, visitor: "Visitor") -> str:
        """Dispatch to visitor.visit_expression(self)."""
        return visitor.visit_expression(self)
@dataclass
class LoopStatement(Statement):
    """for / while loop."""
    source_line: int
    header: str             # the full for/while ( ... ) part
    body_lines: list[str] = field(default_factory=list)

    def visit(self, visitor: "Visitor") -> str:
        """Dispatch to visitor.visit_loop_statement(self)."""
        return visitor.visit_loop_statement(self)


@dataclass
class IfStatement(Statement):
    """if / else if / else block."""
    source_line: int
    condition: str
    then_body: list[str] = field(default_factory=list)
    else_body: list[str] | None = None

    def visit(self, visitor: "Visitor") -> str:
        """Dispatch to visitor.visit_if_statement(self)."""
        return visitor.visit_if_statement(self)


@dataclass
class RawLine(Statement):
    """A line that could not be matched to a known pattern."""
    source_line: int
    text: str

    def visit(self, visitor: "Visitor") -> str:
        """Dispatch to visitor.visit_raw_line(self)."""
        return visitor.visit_raw_line(self)


# ---------------------------------------------------------------------------
# Top-level dump node
# ---------------------------------------------------------------------------

@dataclass
class ParforDump:
    """Root of the parsed AST."""
    source_file: Path          # original C++ source file path
    source_line: int           # line number in that file
    dump_file: Path            # path to the .cc dump file
    macro: ParforMacro
    params: list[OffloadParamDeclaration] = field(default_factory=list)
    body_statements: list[Statement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_MACRO_RE = re.compile(
    r'^\s*OFFLOAD_PARFOR_(\d+)D(_(?:TRIANGULAR|UPPER_TRIANGULAR)?)?'
    r'_PARAM\s*\(\s*'
    r'([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)'  # iteration vars
    r',\s*'
    r'(.*?)'                                      # range expression
    r',\s*\(([^)]*)\)\s*\)'                       # extra params group
)

_DECL_RE = re.compile(
    r'^([^;=]+?)'   # type (everything before first '=')
    r'(?:=\s*([^;]*))? '  # optional init expr
    r';'              # trailing semicolon
)

_LOOP_RE = re.compile(r'^(\s*)(for|while)\s*\(')

_IF_RE = re.compile(r'^\s*(if|else if)\s*\(')

# Matches [ ... ] on the left side of an assignment (array subscript or slice)
_ASSIGN_TARGET_RE = re.compile(r'^(.*?)\s*=\s*')

_INCREMENT_RE = re.compile(r'(?:\+\+|--)(?=$|\s*[;,])')


def _parse_macro(invocation_line: str) -> ParforMacro:
    """Parse a single OFFLOAD_PARFOR_*_PARAM line into a ParforMacro."""
    m = _MACRO_RE.match(invocation_line)
    if not m:
        raise ValueError(f"Cannot parse macro invocation: {invocation_line!r}")

    dim_str, tri, iter_vars_raw, range_expr, extra_params_raw = m.groups()
    dim = int(dim_str)

    kind_name = f"OFFLOAD_PARFOR_{dim}D"
    if tri in ("_TRIANGULAR", "_UPPER_TRIANGULAR"):
        kind_name += tri.upper()
    kind_name += "_PARAM"
    kind = MacroKind(kind_name)

    iter_vars = [v.strip() for v in iter_vars_raw.split(',')]
    extra_params = tuple(p.strip() for p in extra_params_raw.split(',') if p.strip())

    return ParforMacro(
        kind=kind,
        iteration_vars=iter_vars,
        range_expression=range_expr.strip(),
        extra_params=extra_params,
    )


def _find_eq_pos(s: str) -> int:
    """Return the position of the first '=' that is not inside '<...>', '(...)', or '{...}'."""
    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '<':
            depth_angle += 1
        elif c == '>':
            depth_angle -= 1
        elif c == '(':
            depth_paren += 1
        elif c == ')':
            depth_paren -= 1
        elif c == '{':
            depth_brace += 1
        elif c == '}':
            depth_brace -= 1
        elif c == '=' and depth_angle == 0 and depth_paren == 0 and depth_brace == 0:
            return i
        i += 1
    return -1


def _parse_declarations(lines: list[str]) -> list[OffloadParamDeclaration]:
    """Parse OFFLOAD_PARAMETERS lines into declarations."""
    decls: list[OffloadParamDeclaration] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == ';':
            continue

        # Strip trailing semicolon
        code = stripped.rstrip(';').strip()
        eq_pos = _find_eq_pos(code)

        if eq_pos != -1:
            type_and_name = code[:eq_pos].strip()
            init_expr = code[eq_pos + 1:].strip()
        else:
            type_and_name = code
            init_expr = None

        # Extract the name as the last token of 'type_and_name'
        tokens = type_and_name.split()
        if not tokens:
            continue
        name = tokens[-1]
        type_str = type_and_name[:type_and_name.rfind(name)].strip()

        decls.append(OffloadParamDeclaration(type_str=type_str, name=name, init_expr=init_expr))
    return decls


def _parse_body(body_lines: list[str], dump_path: Path) -> list[Statement]:
    """Parse body lines into structured Statements."""
    stmts: list[Statement] = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        lineno = i + 1  # approximate, not the original source line

        # Skip blank lines
        if not line.strip():
            i += 1
            continue

        # Detect loops: for(...) { ... } or while(...) { ... }
        loop_m = _LOOP_RE.match(line)
        if loop_m and (')' in line or '(' in line):  # Allow multi-line loops
            # Find the closing paren
            if ')' not in line:
                # Look ahead for closing paren
                j = i + 1
                while j < len(body_lines) and ')' not in body_lines[j]:
                    j += 1
                if j < len(body_lines):
                    # Found closing paren, combine lines for brace detection
                    combined = line
                    while j > i:
                        combined = body_lines[j] + ' ' + combined
                        j -= 1
                    has_close_paren = True
                else:
                    has_close_paren = False
            else:
                has_close_paren = True
                combined = line
            
            if has_close_paren and ('{' in combined or (i + 1 < len(body_lines) and '{' in body_lines[i + 1])):
                # Find the closing paren position
                close_paren_pos = combined.rfind(')') + 1
                header_start = combined.find('(')
                if header_start != -1 and close_paren_pos > header_start:
                    header = combined[:close_paren_pos]
                    brace_depth = combined[header_start:].count('{') - combined[header_start:].count('}')
                else:
                    brace_depth = 0
            else:
                brace_depth = 0
                
            if brace_depth <= 0:
                stmts.append(LoopStatement(source_line=lineno, header=header))
                i += 1
            else:
                body_lines_list = []
                while brace_depth > 0 and i + 1 < len(body_lines):
                    i += 1
                    bline = body_lines[i]
                    brace_depth += bline.count('{') - bline.count('}')
                    body_lines_list.append(bline)
                stmts.append(LoopStatement(source_line=lineno, header=header, body_lines=body_lines_list))
                i += 1
            continue

        # Rest of the function continues...
        # Detect if / else if
        if_m = _IF_RE.match(line)
        if if_m:
            cond_end = line.index(')') + 1
            condition = line[line.index('(') + 1:cond_end - 1]
            brace_depth = line.count('{') - line.count('}')
            if brace_depth <= 0:
                stmts.append(IfStatement(source_line=lineno, condition=condition))
                i += 1
            else:
                j = i + 1
                then_body = []
                while brace_depth > 0 and j < len(body_lines):
                    bline = body_lines[j]
                    brace_depth += bline.count('{') - bline.count('}')
                    then_body.append(bline)
                    j += 1
                else_body = None
                # Look for else / else if after the closing brace
                while j < len(body_lines):
                    eline = body_lines[j].strip()
                    if eline.startswith('else ') and '{' in eline:
                        ebrace_depth = eline.count('{') - eline.count('}')
                        elif_else_body = []
                        if ebrace_depth <= 0:
                            else_body = [eline]
                            j += 1
                        else:
                            j += 1
                            while ebrace_depth > 0 and j < len(body_lines):
                                ebline = body_lines[j]
                                ebrace_depth += ebline.count('{') - ebline.count('}')
                                elif_else_body.append(ebline)
                                j += 1
                            else_body = elif_else_body
                        break
                    elif eline.startswith('else {'):
                        else_body = [eline]
                        j += 1
                        break
                    else:
                        break
                stmts.append(IfStatement(source_line=lineno, condition=condition, then_body=then_body, else_body=else_body))
                i = j
            continue

        # Assignment: target = expr;
        if '=' in line and not line.strip().startswith(('for', 'while', 'if', '//')) and ';' in line:
            am = _ASSIGN_TARGET_RE.match(line.strip())
            if am:
                target = am.group(1).strip()
                stmts.append(Expression(source_line=lineno, kind='assignment', target=target))
                i += 1
                continue

        # Increment / decrement
        if _INCREMENT_RE.search(line):
            stmts.append(Expression(source_line=lineno, kind='increment'))
            i += 1
            continue

        # Check for function call pattern: name( ... );
        call_m = re.match(r'^\s*([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)?)\s*\(', line.strip())
        if call_m and ';' in line:
            stmts.append(Expression(source_line=lineno, kind='call', target=call_m.group(1)))
            i += 1
            continue

        # Raw fallback
        stmts.append(RawLine(source_line=lineno, text=line))
        i += 1

    return stmts
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dump_file(path: str | Path) -> ParforDump:
    """Parse a PARFOR dump .cc file into a ParforDump AST.

    Raises ValueError if the file cannot be parsed.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    return _parse_from_string(text, dump_path=path)


def _parse_from_string(text: str, dump_path: Path | None = None) -> ParforDump:
    """Parse dump content from a string."""
    lines = text.split('\n')

    # Extract source file and line from the header comment
    header_m = re.search(r'// === PARFOR block from\s+(.+?):(\d+)\s*===', lines[0] if lines else '')
    if not header_m:
        raise ValueError("Missing PARFOR header comment")
    source_file = Path(header_m.group(1))
    source_line = int(header_m.group(2))

    # Find sections
    macro_line: str | None = None
    param_lines: list[str] = []
    body_lines: list[str] = []

    in_params = False
    in_body = False

    for line in lines[1:]:  # skip header comment
        if line.startswith('// === OFFLOAD_PARAMETERS ==='):
            in_params = True
            in_body = False
            continue
        elif line.startswith('// === PARFOR BODY ==='):
            in_params = False
            in_body = True
            continue
        elif line.startswith('// === '):
            in_params = False
            in_body = False
            continue

        if in_params:
            param_lines.append(line)
        elif in_body:
            body_lines.append(line)
        else:
            # Try to capture the macro line (first non-blank, non-comment line before OFFLOAD_PARAMETERS)
            stripped = line.strip()
            if stripped and not stripped.startswith('//'):
                if macro_line is None:
                    macro_line = stripped

    if macro_line is None:
        raise ValueError("No PARFOR macro invocation found in dump file")

    # Parse into AST nodes
    macro = _parse_macro(macro_line)
    params = _parse_declarations(param_lines)
    body_statements = _parse_body(body_lines, dump_path or Path('.'))

    return ParforDump(
        source_file=source_file,
        source_line=source_line,
        dump_file=dump_path or Path('.'),
        macro=macro,
        params=params,
        body_statements=body_statements,
    )


# ---------------------------------------------------------------------------
# Pretty-printer for debugging / inspection
# ---------------------------------------------------------------------------

def dump_ast(dump: ParforDump, indent: int = 0) -> str:
    """Return a human-readable tree representation of the AST."""
    prefix = '  ' * indent
    result_lines: list[str] = []

    result_lines.append(f'{prefix}ParforDump(source={dump.source_file}:{dump.source_line})')
    result_lines.append(f'{prefix}  macro={dump.macro.kind.value}')
    result_lines.append(f'{prefix}  iteration_vars={dump.macro.iteration_vars}')
    result_lines.append(f'{prefix}  range_expr={dump.macro.range_expression!r}')
    result_lines.append(f'{prefix}  extra_params={dump.macro.extra_params!r}')

    result_lines.append(f'{prefix}  params:')
    for p in dump.params:
        result_lines.append(f'{prefix}    {p.type_str} {p.name} = {p.init_expr!r}')

    result_lines.append(f'{prefix}  body:')
    for stmt in dump.body_statements:
        result_lines.append(_format_stmt(stmt, indent + 2))

    return '\n'.join(result_lines)


def _format_stmt(stmt: Statement, indent: int) -> str:
    prefix = '  ' * indent
    if isinstance(stmt, Expression):
        if stmt.kind == 'assignment':
            return f'{prefix}Expr({stmt.kind}, target={stmt.target!r})'
        return f'{prefix}Expr({stmt.kind})'
    elif isinstance(stmt, LoopStatement):
        hdr = stmt.header[:50] + '...' if len(stmt.header) > 50 else stmt.header
        body_preview = _preview_body(stmt.body_lines)
        return f'{prefix}Loop(header={hdr!r}, body={body_preview})'
    elif isinstance(stmt, IfStatement):
        cond = stmt.condition[:40] + '...' if len(stmt.condition) > 40 else stmt.condition
        then_preview = _preview_body(stmt.then_body)
        return f'{prefix}If(cond={cond!r}, then={then_preview})'
    elif isinstance(stmt, RawLine):
        text = stmt.text[:50] + '...' if len(stmt.text) > 50 else stmt.text
        return f'{prefix}Raw({text!r})'
    return f'{prefix}?{type(stmt).__name__}'


def _preview_body(lines: list[str]) -> str:
    if not lines:
        return ''
    first = lines[0].strip()[:30] or (lines[-1].strip()[:30] if len(lines) > 1 else '')
    suffix = f', +{len(lines)-1} lines' if len(lines) > 1 else ''
    return f'{first!r}{suffix}'


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        dump = parse_dump_file(p)
        print(dump_ast(dump))
