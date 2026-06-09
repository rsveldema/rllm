"""Compiler front-end: parse kernel files -> AST."""

from lark import Lark, Tree
import codegen_ast
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


# Quick smoke test
if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        parse(path)
