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
        "auto <LOCAL><STI_0> = "
        "<MCP><GLOBAL><STI_1>::<FIELD><STI_2></MCP>(<STRING><STI_3>, <GLOBAL><STI_4>);\n"
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
        "auto <LOCAL><STI_0> = <MCP><GLOBAL><STI_1>::<FIELD><STI_2></MCP>(<STRING><STI_3>, <GLOBAL><STI_4>);\n"
        "<MCP><GLOBAL><STI_5></MCP>(<LOCAL><STI_0>);\n"
    )


def test_field_access_uses_a_dedicated_identifier_token():
    source = "auto value = object.field; auto other = pointer->member;\n"
    assert postprocessor.abstract_code_symbols(source, ".cpp") == (
        "auto <LOCAL><STI_0> = <MCP><GLOBAL><STI_1>.<FIELD><STI_2></MCP>; "
        "auto <LOCAL><STI_3> = <MCP><GLOBAL><STI_4>-><FIELD><STI_5></MCP>;\n"
    )


def test_abstraction_preserves_keywords_comments_and_punctuation():
    source = "for (int index = 0; index < count; ++index) { // useful prose\nreturn index;\n}\n"

    assert postprocessor.abstract_code_symbols(source) == (
		"for (int <LOOP><STI_0> = <INTEGER><ITI_0>; <LOOP><STI_0> < <GLOBAL><STI_1>; ++<LOOP><STI_0>) { // useful prose\n"
        "return <LOOP><STI_0>;\n}\n"
    )


def test_identifier_categories_cover_parameters_locals_loops_and_unknowns():
    source = (
        "void sort(int count, Widget value) {\n"
        "auto local = count;\n"
        "for (int index = 0; index < count; ++index) local = external;\n"
        "}\n"
    )
    assert postprocessor.abstract_code_symbols(source, ".cpp") == (
        "void <MCP><GLOBAL><STI_0></MCP>(int <PARAM><STI_1>, "
        "<GLOBAL><STI_2> <PARAM><STI_3>) {\n"
        "auto <LOCAL><STI_4> = <PARAM><STI_1>;\n"
		"for (int <LOOP><STI_5> = <INTEGER><ITI_0>; <LOOP><STI_5> < <PARAM><STI_1>; "
		"++<LOOP><STI_5>) <LOCAL><STI_4> = <GLOBAL><STI_6>;\n"
        "}\n"
    )


def test_identifier_categories_use_plain_tokens():
    scopes: list[dict[str, str]] = [{}]
    pending: dict[str, str] = {}
    string_table = postprocessor.StringTable()
    local_tokens = [postprocessor._assigned_identifier_token(
        f"local{index}", "local", scopes, pending, string_table) for index in range(17)]
    param_tokens = [postprocessor._assigned_identifier_token(
        f"param{index}", "param", scopes, pending, string_table) for index in range(17)]
    global_tokens = [postprocessor._assigned_identifier_token(
        f"global{index}", "global", scopes, pending, string_table) for index in range(17)]
    loop_tokens = [postprocessor._assigned_identifier_token(
        f"loop{index}", "loop", scopes, pending, string_table) for index in range(9)]

    assert {token.split("<STI_")[0] for token in local_tokens} == {"<LOCAL>"}
    assert {token.split("<STI_")[0] for token in param_tokens} == {"<PARAM>"}
    assert {token.split("<STI_")[0] for token in global_tokens} == {"<GLOBAL>"}
    assert {token.split("<STI_")[0] for token in loop_tokens} == {"<LOOP>"}
    assert len(set(local_tokens + param_tokens + global_tokens + loop_tokens)) == 60


def test_identifier_slots_are_reused_after_scope_exit():
    source = "{ auto first = 1; { auto second = 2; } auto third = 3; } { auto fourth = 4; }"
    assert postprocessor.abstract_code_symbols(source, ".cpp") == (
		"{ auto <LOCAL><STI_0> = <INTEGER><ITI_0>; { auto <LOCAL><STI_1> = <INTEGER><ITI_1>; } auto <LOCAL><STI_2> = <INTEGER><ITI_2>; } "
		"{ auto <LOCAL><STI_3> = <INTEGER><ITI_3>; }"
	)


def test_numeric_constants_are_atomic_typed_tokens_with_exact_payloads():
	source = "a = 13245; b = 3.14; c = 0xffu; d = 6.02e23; e = 0x1.fp3;"
	assert postprocessor.abstract_code_symbols(source, ".cpp") == (
		"<LOCAL><STI_0> = <INTEGER><ITI_0>; <LOCAL><STI_1> = <FLOAT><FTI_0>; "
		"<LOCAL><STI_2> = <INTEGER><ITI_1>; <LOCAL><STI_3> = <FLOAT><FTI_1>; "
		"<LOCAL><STI_4> = <FLOAT><FTI_2>;"
	)


def test_parameter_slots_restart_at_zero_for_each_function():
    source = (
        "void declared(int stale); void first(int alpha) { } "
        "void second(int beta) { }"
    )
    assert postprocessor.abstract_code_symbols(source, ".cpp") == (
        "void <MCP><GLOBAL><STI_0></MCP>(int <PARAM><STI_1>); "
        "void <MCP><GLOBAL><STI_2></MCP>(int <PARAM><STI_3>) { } "
        "void <MCP><GLOBAL><STI_4></MCP>(int <PARAM><STI_5>) { }"
    )


def test_global_slots_restart_for_free_functions_but_not_methods():
    assert postprocessor.abstract_code_symbols(
        "void first() { } void second() { }", ".cpp"
    ) == (
        "void <MCP><GLOBAL><STI_0></MCP>() { } "
        "void <MCP><GLOBAL><STI_1></MCP>() { }"
    )
    assert postprocessor.abstract_code_symbols(
        "class Widget { void first() { } void second() { } };", ".cpp"
    ) == (
        "class <CLASS_NAME><STI_0> { void <MCP><GLOBAL><STI_1></MCP>() { } "
        "void <MCP><GLOBAL><STI_2></MCP>() { } };"
    )


def test_class_names_keep_their_category_on_later_references():
    assert postprocessor.abstract_code_symbols(
        "class Widget {}; Widget value;", ".cpp"
    ) == "class <CLASS_NAME><STI_0> {}; <CLASS_NAME><STI_0> <GLOBAL><STI_1>;"


def test_abstraction_is_idempotent_and_marks_includes_as_mcp():
    source = '#include <iostream>\n#include "local.hpp"\nstd::println("hello");\n'
    once = postprocessor.abstract_code_symbols(source)

    assert once == (
        "#include <MCP><STRING><STI_0></MCP>\n"
        "#include <MCP><STRING><STI_1></MCP>\n"
        "<MCP><GLOBAL><STI_2>::<FIELD><STI_3></MCP>(<STRING><STI_4>);\n"
    )
    assert postprocessor.abstract_code_symbols(once) == once


def test_imported_library_name_is_an_mcp_access():
    assert postprocessor.abstract_code_symbols("import requests\n", ".py") == (
        "import <MCP><GLOBAL><STI_0></MCP>\n"
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
        "<LOCAL><STI_0> = <GLOBAL><STI_1>;\n"
    )
