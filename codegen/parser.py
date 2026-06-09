"""Compiler front-end: parse kernel files -> AST."""

from lark import Lark
from codegen_ast import Program
from visitors.pretty_printer import PrettyPrinter
from transforms import transform


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


grammar = read_file("grammar.lark")
parser = Lark(grammar, start="program")


def parse(filename: str):
    print(f"--------------- parsing: {filename} -----------------")
    text = read_file(filename)
    ret = parser.parse(text)
    program = transform(ret)
    return program

def prettyprint(program: Program):
    printer = PrettyPrinter()
    s = program.accept(printer)
    print(s)

# Quick smoke test
if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        program = parse(path)
        prettyprint(program)
