import training_postprocessor as postprocessor


def test_python_indentation_converts_four_space_groups_to_tabs():
    source = (
        "def example():\n"
        "    first = 1\n"
        "        second = 2\n"
        "            third = 3\n"
    )

    assert postprocessor.normalize_python_indentation(source) == (
        "def example():\n"
        "\tfirst = 1\n"
        "\t\tsecond = 2\n"
        "\t\t\tthird = 3\n"
    )


def test_python_indentation_preserves_remainder_and_interior_spaces():
    source = "      value = \"four    interior spaces\"\n"

    assert postprocessor.normalize_python_indentation(source) == (
        "\t  value = \"four    interior spaces\"\n"
    )


def test_python_indentation_preserves_existing_tabs():
    source = "\t    value = 1\n"

    assert postprocessor.normalize_python_indentation(source) == "\t\tvalue = 1\n"


def test_process_file_only_applies_python_indentation_to_python(tmp_path):
    python_file = tmp_path / "sample.py"
    text_file = tmp_path / "sample.data"
    python_file.write_text("    value = 1\n", encoding="utf-8")
    text_file.write_text("    value = 1\n", encoding="utf-8")

    assert postprocessor.process_file(python_file)
    assert not postprocessor.process_file(text_file)
    assert python_file.read_text(encoding="utf-8") == "\tvalue = 1\n"
    assert text_file.read_text(encoding="utf-8") == "    value = 1\n"


def test_inspection_copy_materializes_runtime_abstraction(tmp_path):
    source_file = tmp_path / "sample.cpp"
    source_file.write_text(
        'auto result = std::println("hello", name);\n', encoding="utf-8")

    assert postprocessor.process_file(source_file, abstract_symbols=True)
    assert source_file.read_text(encoding="utf-8") == (
        "auto <IDENTIFIER> = "
        "<MCP><IDENTIFIER>::<IDENTIFIER></MCP>(<STRING>, <IDENTIFIER>);\n"
    )


def test_strip_c_cpp_comments_preserves_literals_and_newlines():
    source = (
        'const char* url = "https://example.test/a/*b*/"; // trailing\n'
        "char slash = '/'; /* first\nsecond */ int value = 3;\n"
    )

    assert postprocessor.strip_c_cpp_comments(source) == (
        'const char* url = "https://example.test/a/*b*/"; \n'
        "char slash = '/'; \n int value = 3;\n"
    )


def test_strip_python_comments_preserves_hash_in_string():
    source = 'value = "# literal"  # trailing\n# whole line\nnext_value = 2\n'

    stripped = postprocessor.strip_python_comments(source)

    assert '"# literal"' in stripped
    assert "# trailing" not in stripped
    assert "# whole line" not in stripped
    assert "next_value = 2" in stripped


def test_process_file_strips_comments_only_when_requested(tmp_path):
    source_file = tmp_path / "sample.cpp"
    source_file.write_text("int value = 1; // explanation\n", encoding="utf-8")

    assert postprocessor.process_file(source_file, strip_comments=True)
    assert source_file.read_text(encoding="utf-8") == "int value = 1;\n"


def test_abstracts_identifiers_strings_calls_and_cpp_namespace_access():
    source = 'auto result = std::println("hello {}", user_name);\nhelper(result);\n'

    assert postprocessor.abstract_code_symbols(source) == (
        "auto <IDENTIFIER> = <MCP><IDENTIFIER>::<IDENTIFIER></MCP>(<STRING>, <IDENTIFIER>);\n"
        "<MCP><IDENTIFIER></MCP>(<IDENTIFIER>);\n"
    )


def test_field_access_uses_the_shared_identifier_token():
    source = "auto value = object.field; auto other = pointer->member;\n"
    assert postprocessor.abstract_code_symbols(source, ".cpp") == (
        "auto <IDENTIFIER> = <MCP><IDENTIFIER>.<IDENTIFIER></MCP>; "
        "auto <IDENTIFIER> = <MCP><IDENTIFIER>-><IDENTIFIER></MCP>;\n"
    )


def test_abstraction_preserves_keywords_comments_and_punctuation():
    source = "for (int index = 0; index < count; ++index) { // useful prose\nreturn index;\n}\n"

    assert postprocessor.abstract_code_symbols(source) == (
        "for (int <IDENTIFIER> = <INTEGER>; <IDENTIFIER> < <IDENTIFIER>; ++<IDENTIFIER>) { // useful prose\n"
        "return <IDENTIFIER>;\n}\n"
    )


def test_identifiers_share_a_token_across_syntax_roles():
    source = (
        "void sort(int count, Widget value) {\n"
        "auto local = count;\n"
        "for (int index = 0; index < count; ++index) local = external;\n"
        "}\n"
    )
    assert postprocessor.abstract_code_symbols(source, ".cpp") == (
        "void <MCP><IDENTIFIER></MCP>(int <IDENTIFIER>, "
        "<IDENTIFIER> <IDENTIFIER>) {\n"
        "auto <IDENTIFIER> = <IDENTIFIER>;\n"
        "for (int <IDENTIFIER> = <INTEGER>; <IDENTIFIER> < <IDENTIFIER>; "
        "++<IDENTIFIER>) <IDENTIFIER> = <IDENTIFIER>;\n"
        "}\n"
    )


def test_identifier_tokens_do_not_depend_on_category_or_count():
    source = ";".join(f"name{index}" for index in range(100))
    assert postprocessor.abstract_code_symbols(source) == ";".join(["<IDENTIFIER>"] * 100)


def test_integer_spelling_is_abstracted_but_floats_are_preserved():
    assert postprocessor.abstract_code_symbols("123 0xff 0b10 1'000 42ULL 1.25 2e-3") == (
        "<INTEGER> <INTEGER> <INTEGER> <INTEGER> <INTEGER> 1.25 2e-3"
    )


def test_abstraction_is_idempotent_and_marks_includes_as_mcp():
    source = '#include <iostream>\n#include "local.hpp"\nstd::println("hello");\n'
    once = postprocessor.abstract_code_symbols(source)

    assert once == (
        "#include <MCP><STRING></MCP>\n"
        "#include <MCP><STRING></MCP>\n"
        "<MCP><IDENTIFIER>::<IDENTIFIER></MCP>(<STRING>);\n"
    )
    assert postprocessor.abstract_code_symbols(once) == once


def test_imported_library_name_is_an_mcp_access():
    assert postprocessor.abstract_code_symbols("import requests\n", ".py") == (
        "import <MCP><IDENTIFIER></MCP>\n"
    )


def test_python_java_and_rust_files_keep_source_spelling(tmp_path):
    cases = {
        "sample.py": (
            'import requests\nvalue = requests.get("https://example.test")\n',
            'import requests\nvalue = requests.get("https://example.test")\n',
        ),
        "Sample.java": (
            'class Sample { void run() { System.out.println("hello"); } }\n',
            'class Sample { void run() { System.out.println("hello"); } }\n',
        ),
        "sample.rs": (
            'use std::fs;\nfn run() { std::fs::read_to_string("input.txt"); }\n',
            'use std::fs;\nfn run() { std::fs::read_to_string("input.txt"); }\n',
        ),
    }

    for filename, (source, expected) in cases.items():
        path = tmp_path / filename
        path.write_text(source, encoding="utf-8")
        postprocessor.process_file(path)
        assert path.read_text(encoding="utf-8") == expected


def test_only_the_active_languages_keywords_are_preserved():
    assert postprocessor.abstract_code_symbols("fn = value;\n", ".java") == (
        "<IDENTIFIER> = <IDENTIFIER>;\n"
    )
