"""Unit tests for the codegen package.

Tests the AST parser, visitor pattern, and printer against realistic parfor dumps.
"""

import unittest
from pathlib import Path
from codegen.parfor_ast import (
    ParforDump,
    ParforMacro,
    OffloadParamDeclaration,
    Expression,
    LoopStatement,
    IfStatement,
    RawLine,
    parse_dump_file,
    dump_ast,
)
from codegen.visitor import Visitor
from codegen.printer import StatementPrinter


class TestParforDumpParser(unittest.TestCase):
    """Test parsing of parfor dump files into AST."""

    def test_matmul_dump_parsing(self):
        """Prove that the parser correctly handles a matmul parfor dump.

        This is the canonical test case: a 2D parfor with complex template types,
        loop nesting, and assignment patterns typical of matmul operations.
        """
        dump_path = Path(__file__).parent / "dump_matmul_A_B_C.cc"
        dump = parse_dump_file(dump_path)

        # Check top-level structure
        self.assertIsInstance(dump, ParforDump)
        self.assertEqual(dump.source_file, Path("src/vecmath.cc"))
        self.assertEqual(dump.source_line, 296)

        # Check macro parsing
        self.assertIsInstance(dump.macro, ParforMacro)
        self.assertEqual(dump.macro.kind.value, "OFFLOAD_PARFOR_2D_PARAM")
        self.assertEqual(dump.macro.dim, 2)
        self.assertEqual(dump.macro.iteration_vars, ["i", "j"])
        self.assertEqual(dump.macro.range_expression, "grid")
        self.assertEqual(dump.macro.extra_params, ("A", "B", "C"))

        # Check parameter parsing
        self.assertIsInstance(dump.params, list)
        self.assertEqual(len(dump.params), 3)

        # First param: A
        a_param = dump.params[0]
        self.assertIsInstance(a_param, OffloadParamDeclaration)
        self.assertEqual(a_param.type_str, "const rlmm::fixed_size_matrix<rlmm_float, EmbeddingDimension, Position>&")
        self.assertEqual(a_param.name, "A")
        self.assertEqual(a_param.init_expr, "A")

        # Second param: B
        b_param = dump.params[1]
        self.assertIsInstance(b_param, OffloadParamDeclaration)
        self.assertEqual(b_param.type_str, "const rlmm::fixed_size_matrix<rlmm_float, TokenID, Position>&")
        self.assertEqual(b_param.name, "B")

        # Third param: C
        c_param = dump.params[2]
        self.assertIsInstance(c_param, OffloadParamDeclaration)
        self.assertEqual(c_param.type_str, "rlmm::fixed_size_matrix<rlmm_float, PositionID, Token>&")
        self.assertEqual(c_param.name, "C")

        # Check body parsing
        self.assertIsInstance(dump.body_statements, list)
        self.assertGreater(len(dump.body_statements), 0)


class TestVisitorPattern(unittest.TestCase):
    """Test that the visitor pattern works correctly for all Statement types."""

    def setUp(self):
        """Create a minimal AST for testing."""
        dump_path = Path(__file__).parent / "dump_matmul_A_B_C.cc"
        self.dump = parse_dump_file(dump_path)

    def test_visitor_dispatch(self):
        """Prove that each statement type can be visited correctly."""

        class TestVisitor(Visitor):
            def __init__(self):
                self.expression_calls = 0
                self.loop_calls = 0
                self.if_calls = 0
                self.raw_calls = 0

            def visit_expression(self, node) -> str:
                self.expression_calls += 1
                return f"Expr:{node.kind}"

            def visit_loop_statement(self, node) -> str:
                self.loop_calls += 1
                return f"Loop:{node.header[:20]}"

            def visit_if_statement(self, node) -> str:
                self.if_calls += 1
                return f"If:{node.condition[:20]}"

            def visit_raw_line(self, node) -> str:
                self.raw_calls += 1
                return f"Raw:{node.text[:20]}"

        visitor = TestVisitor()

        # Visit each statement in the body
        for stmt in self.dump.body_statements:
            result = stmt.visit(visitor)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

        # Check that at least some statements were visited
        total = visitor.expression_calls + visitor.loop_calls + visitor.if_calls + visitor.raw_calls
        self.assertGreater(total, 0, "At least one statement should have been visited")

    def test_visitor_returns_string(self):
        """Prove that visit() returns a string for each statement type."""
        class LocalVisitor(Visitor):
            def visit_expression(self, node) -> str:
                return "expr"
            def visit_loop_statement(self, node) -> str:
                return "loop"  
            def visit_if_statement(self, node) -> str:
                return "if"
            def visit_raw_line(self, node) -> str:
                return "raw"
        
        local_visitor = LocalVisitor()
        for stmt in self.dump.body_statements:
            result = stmt.visit(local_visitor)
            self.assertIsInstance(result, str)


class TestPrinter(unittest.TestCase):
    """Test that the StatementPrinter visitor works correctly."""

    def setUp(self):
        """Create a minimal AST for testing."""
        dump_path = Path(__file__).parent / "dump_matmul_A_B_C.cc"
        self.dump = parse_dump_file(dump_path)

    def test_printer_produces_output(self):
        """Prove that the printer produces readable output for all statements."""
        printer = StatementPrinter()

        # Print the entire AST
        full_output = dump_ast(self.dump)
        self.assertIsInstance(full_output, str)
        self.assertGreater(len(full_output), 0)

        # Print individual statements
        printed_stmts = []
        for stmt in self.dump.body_statements:
            result = stmt.visit(printer)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            printed_stmts.append(result)

        # Verify the output is human-readable (contains meaningful content)
        combined_output = "\n".join(printed_stmts)
        self.assertTrue(any(keyword in combined_output for keyword in ["assignment", "call", "Raw"]))  # Various statement types
        self.assertIn("Loop", combined_output)  # Loop statements should be printed


class TestDumpASTFormat(unittest.TestCase):
    """Test the format and structure of dump_ast output."""

    def test_dump_ast_format(self):
        """Prove that dump_ast() produces properly formatted AST tree."""
        dump_path = Path(__file__).parent / "dump_matmul_A_B_C.cc"
        dump = parse_dump_file(dump_path)

        output = dump_ast(dump)
        lines = output.split('\n')

        # First line should be the ParforDump header
        self.assertTrue(lines[0].startswith("ParforDump(source="))
        self.assertIn("src/vecmath.cc:296", lines[0])

        # Second line should contain macro info
        self.assertTrue(any("macro=OFFLOAD_PARFOR_2D_PARAM" in line for line in lines))

        # Should contain iteration vars
        self.assertTrue(any("iteration_vars=['i', 'j']" in line for line in lines))

        # Should contain params section
        self.assertTrue(any(line.strip() == "params:" for line in lines))

        # Should contain body section
        self.assertTrue(any(line.strip() == "body:" for line in lines))


if __name__ == '__main__':
    unittest.main()
