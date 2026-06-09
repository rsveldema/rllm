from lark import Lark

def read_file(filename: sstr) -> str:
    with open(filename) as f:
        return f.read()


grammar = read_file("grammar.lark")

parser = Lark(grammar, start="program")


def parse(filename: str):
    print(f"parsing: {filename}")

    text = read_file(filename)
    ret = parser.parse(text)
    print(f"got: {ret}")

parse("../tests/dump_matmul_A_B_C.cc")

