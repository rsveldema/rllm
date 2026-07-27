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
