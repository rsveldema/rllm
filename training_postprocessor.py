#!/usr/bin/env python3
"""Normalize files in the selected training-data directory before training.

Rules:
1. Remove non-ASCII characters from every processed file.
2. Leave source spelling intact. Runtime language-aware tokenization abstracts
   identifiers, literals, and MCP accesses without rewriting training files.
3. For C/C++ files:
   - normalize line endings to '\n'
   - trim leading whitespace
   - drop empty lines
   - move inline comments to their own next line
4. For Python files:
   - replace each group of four leading spaces with one tab
5. For .md/.txt files:
	- split sentence punctuation ('.', '!', '?') into line breaks
	- keep 'etc.' inside sentences
"""

from __future__ import annotations

import argparse
import bisect
import io
from pathlib import Path
import re
import shutil
import tokenize

from create_tokenizer_map import (
	CPP_KEYWORDS,
	JAVA_KEYWORDS,
	PREPROCESSOR_KEYWORDS,
	PYTHON_KEYWORDS,
	RUST_KEYWORDS,
	SHELL_KEYWORDS,
)


C_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
PYTHON_EXTENSIONS = {".py"}
TEXT_EXTENSIONS = {".md", ".txt"}
CODE_EXTENSIONS = C_EXTENSIONS | PYTHON_EXTENSIONS | {".rs", ".java", ".sh"}

LOOP_TOKENS = tuple(f"<LOOP_{index}>" for index in range(8))
LOOP_OVERFLOW_TOKEN = "<LOOP_OVERFLOW>"
LOCAL_TOKENS = tuple(f"<LOCAL_{index}>" for index in range(16))
PARAM_TOKENS = tuple(f"<PARAM_{index}>" for index in range(16))
GLOBAL_TOKENS = tuple(f"<GLOBAL_{index}>" for index in range(16))
LOCAL_OVERFLOW_TOKEN = "<LOCAL_OVERFLOW>"
PARAM_OVERFLOW_TOKEN = "<PARAM_OVERFLOW>"
GLOBAL_OVERFLOW_TOKEN = "<GLOBAL_OVERFLOW>"
FIELD_ACCESS_TOKEN = "<FIELD_ACCESS_IDENT>"
LEGACY_IDENTIFIER_TOKEN = "<IDENTIFIER>"
STRING_TOKEN = "<STRING>"
MCP_START_TOKEN = "<MCP>"
MCP_END_TOKEN = "</MCP>"

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_QUALIFIED_IDENTIFIER_RE = re.compile(
	r"[A-Za-z_][A-Za-z_0-9]*(?:\s*(?:::|->|\.)\s*[A-Za-z_][A-Za-z_0-9]*)+"
)
_PREPROCESSOR_RE = re.compile(
	r"#\s*(?:" + "|".join(re.escape(word.removeprefix("#")) for word in PREPROCESSOR_KEYWORDS) + r")\b"
)
_KEYWORDS_BY_SUFFIX = {
	".c": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".cc": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".cpp": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".cxx": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".h": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".hh": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".hpp": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".hxx": set(CPP_KEYWORDS) | set(PREPROCESSOR_KEYWORDS),
	".py": set(PYTHON_KEYWORDS),
	".rs": set(RUST_KEYWORDS),
	".java": set(JAVA_KEYWORDS),
	".sh": set(SHELL_KEYWORDS),
}


def _next_non_whitespace(text: str, position: int) -> str:
	while position < len(text) and text[position].isspace():
		position += 1
	return text[position] if position < len(text) else ""


def _identifier_token(
	text: str, start: int, end: int, newline_positions: list[int]
) -> str:
	word = text[start:end]
	if word[:1].isupper():
		return "global"
	line_index = bisect.bisect_left(newline_positions, start)
	line_start = 0 if line_index == 0 else newline_positions[line_index - 1] + 1
	line_end = newline_positions[line_index] if line_index < len(newline_positions) else len(text)
	prefix = text[line_start:start]
	suffix = text[end:line_end]
	for_position = max(prefix.rfind("for ("), prefix.rfind("for("), prefix.rfind("for "))
	if for_position >= 0 and prefix.rfind(")") < for_position:
		if ";" not in prefix[for_position:] and re.match(r"\s*(?:=|:|in\b|;)", suffix):
			return "loop"
	open_paren = prefix.rfind("(")
	if open_paren >= 0 and prefix.rfind(")") < open_paren:
		declaration_prefix = prefix[:open_paren]
		if re.search(r"\b(?:def|fn)\s+\w+\s*$", declaration_prefix) or re.search(
			r"(?:\b\w+\s+){1,}\w+\s*$", declaration_prefix
		):
			return "param"
	if re.match(r"\s*=", suffix) or re.search(r"\b(?:auto|let|var)\s+$", prefix):
		return "local"
	return "global"


def _is_function_declaration_name(
	text: str, start: int, end: int, suffix: str | None
) -> bool:
	if _next_non_whitespace(text, end) != "(":
		return False
	line_start = text.rfind("\n", 0, start) + 1
	prefix = text[line_start:start]
	if suffix == ".py":
		return re.search(r"\bdef\s+$", prefix) is not None
	if suffix == ".rs":
		return re.search(r"\bfn\s+$", prefix) is not None
	if suffix in C_EXTENSIONS or suffix == ".java" or suffix is None:
		declaration = re.split(r"[;{}]", prefix)[-1]
		return (
			bool(_IDENTIFIER_RE.search(declaration))
			and "=" not in declaration
			and not re.search(r"\b(?:return|throw)\s+$", declaration)
		)
	return False


def _assigned_identifier_token(
	word: str,
	category: str,
	scopes: list[dict[str, str]],
	pending_parameters: dict[str, str],
) -> str:
	if category == "param" and word in pending_parameters:
		return pending_parameters[word]
	if category in {"local", "loop"} and word in scopes[-1]:
		return scopes[-1][word]
	if category == "global":
		if word in pending_parameters:
			return pending_parameters[word]
		for scope in reversed(scopes):
			if word in scope:
				return scope[word]
	tokens, overflow = {
			"loop": (LOOP_TOKENS, LOOP_OVERFLOW_TOKEN),
			"local": (LOCAL_TOKENS, LOCAL_OVERFLOW_TOKEN),
			"param": (PARAM_TOKENS, PARAM_OVERFLOW_TOKEN),
		"global": (GLOBAL_TOKENS, GLOBAL_OVERFLOW_TOKEN),
	}[category]
	used = {token for scope in scopes for token in scope.values()}
	used.update(pending_parameters.values())
	token = next((candidate for candidate in tokens if candidate not in used), overflow)
	if category == "global":
		scopes[0][word] = token
	elif category == "param":
		pending_parameters[word] = token
	else:
		scopes[-1][word] = token
	return token


def abstract_code_symbols(text: str, suffix: str | None = None) -> str:
	"""Keep language syntax while hiding source names and literal contents.

	A qualified name or call target is an external capability boundary. Its
	spelling is abstracted as identifiers and enclosed by MCP markers. Comments
	remain prose and are intentionally left alone.
	"""
	keywords = _KEYWORDS_BY_SUFFIX.get(suffix, set().union(*_KEYWORDS_BY_SUFFIX.values()))
	out: list[str] = []
	identifier_scopes: list[dict[str, str]] = [{}]
	pending_parameters: dict[str, str] = {}
	indentation_levels = [0]
	class_scopes = [False]
	function_scopes = [False]
	pending_class_scope = False
	pending_function_scope = False
	newline_positions = [position for position, char in enumerate(text) if char == "\n"]
	i = 0
	expect_library_name = False
	while i < len(text):
		if suffix == ".py" and (i == 0 or text[i - 1] == "\n"):
			line_position = i
			indentation = 0
			while line_position < len(text) and text[line_position] in " \t":
				indentation += 4 if text[line_position] == "\t" else 1
				line_position += 1
			if line_position < len(text) and text[line_position] not in "#\n":
				while len(indentation_levels) > 1 and indentation < indentation_levels[-1]:
					indentation_levels.pop()
					identifier_scopes.pop()
					class_scopes.pop()
					function_scopes.pop()
				if indentation > indentation_levels[-1]:
					indentation_levels.append(indentation)
					identifier_scopes.append(pending_parameters)
					pending_parameters = {}
					class_scopes.append(pending_class_scope)
					function_scopes.append(pending_function_scope)
					pending_class_scope = False
					pending_function_scope = False
		control_token = next(
			(token for token in (*LOOP_TOKENS, LOOP_OVERFLOW_TOKEN, *LOCAL_TOKENS, LOCAL_OVERFLOW_TOKEN,
			 *PARAM_TOKENS, PARAM_OVERFLOW_TOKEN, *GLOBAL_TOKENS, GLOBAL_OVERFLOW_TOKEN,
			 FIELD_ACCESS_TOKEN, STRING_TOKEN, MCP_START_TOKEN, MCP_END_TOKEN)
			 if text.startswith(token, i)),
			None,
		)
		if control_token is not None:
			out.append(control_token)
			i += len(control_token)
			continue
		if text.startswith(LEGACY_IDENTIFIER_TOKEN, i):
			out.append(GLOBAL_OVERFLOW_TOKEN)
			i += len(LEGACY_IDENTIFIER_TOKEN)
			continue
		if text.startswith("//", i):
			end = text.find("\n", i)
			end = len(text) if end < 0 else end
			out.append(text[i:end])
			i = end
			continue
		if text.startswith("/*", i):
			end = text.find("*/", i + 2)
			end = len(text) if end < 0 else end + 2
			out.append(text[i:end])
			i = end
			continue
		if text[i] == "#":
			match = _PREPROCESSOR_RE.match(text, i)
			if match:
				out.append(match.group())
				i = match.end()
				if re.sub(r"\s", "", match.group()) == "#include":
					while i < len(text) and text[i] in " \t":
						out.append(text[i])
						i += 1
					if i < len(text) and text[i] == "<" and not text.startswith(MCP_START_TOKEN, i):
						end = text.find(">", i + 1)
						if end >= 0:
							out.extend((MCP_START_TOKEN, STRING_TOKEN, MCP_END_TOKEN))
							i = end + 1
					elif i < len(text) and text[i] == '"':
						end = text.find('"', i + 1)
						if end >= 0:
							out.extend((MCP_START_TOKEN, STRING_TOKEN, MCP_END_TOKEN))
							i = end + 1
				continue
			end = text.find("\n", i)
			end = len(text) if end < 0 else end
			out.append(text[i:end])
			i = end
			continue
		if text[i] in {'"', "'", '`'}:
			quote = text[i]
			triple = text.startswith(quote * 3, i)
			delimiter = quote * (3 if triple else 1)
			end = i + len(delimiter)
			while end < len(text):
				if text.startswith(delimiter, end):
					end += len(delimiter)
					break
				if text[end] == "\\" and end + 1 < len(text):
					end += 2
				else:
					end += 1
			out.append(STRING_TOKEN)
			i = end
			continue
		qualified = _QUALIFIED_IDENTIFIER_RE.match(text, i)
		if qualified:
			qualified_text = qualified.group()
			if _is_function_declaration_name(text, i, qualified.end(), suffix):
				pending_function_scope = True
			def abstract_qualified_part(match: re.Match[str]) -> str:
				prefix = qualified_text[:match.start()].rstrip()
				cpp_scope_access = prefix.endswith("::") and (
					suffix is None or suffix in C_EXTENSIONS
				)
				if prefix.endswith(".") or prefix.endswith("->") or cpp_scope_access:
					return FIELD_ACCESS_TOKEN
				return _assigned_identifier_token(
					match.group(), "global", identifier_scopes, pending_parameters)
			spelling = re.sub(
				_IDENTIFIER_RE,
				abstract_qualified_part,
				qualified_text,
			)
			out.extend((MCP_START_TOKEN, spelling, MCP_END_TOKEN))
			i = qualified.end()
			expect_library_name = False
			continue
		identifier = _IDENTIFIER_RE.match(text, i)
		if identifier:
			word = identifier.group()
			if word in keywords:
				out.append(word)
				expect_library_name = word in {"import", "from", "use"}
				if word in {"class", "struct"}:
					pending_class_scope = True
			elif expect_library_name:
				out.extend((MCP_START_TOKEN, _assigned_identifier_token(
					word, "global", identifier_scopes, pending_parameters), MCP_END_TOKEN))
				expect_library_name = False
			elif _next_non_whitespace(text, identifier.end()) == "(":
				if _is_function_declaration_name(text, i, identifier.end(), suffix):
					if not any(class_scopes) and not any(function_scopes):
						identifier_scopes[0].clear()
					pending_function_scope = True
				out.extend((MCP_START_TOKEN, _assigned_identifier_token(
					word, "global", identifier_scopes, pending_parameters), MCP_END_TOKEN))
			else:
				category = _identifier_token(text, i, identifier.end(), newline_positions)
				out.append(_assigned_identifier_token(
					word, category, identifier_scopes, pending_parameters))
			i = identifier.end()
			continue
		if text[i] == "{":
			identifier_scopes.append(pending_parameters)
			pending_parameters = {}
			class_scopes.append(pending_class_scope)
			function_scopes.append(pending_function_scope)
			pending_class_scope = False
			pending_function_scope = False
		elif text[i] == "}" and len(identifier_scopes) > 1:
			identifier_scopes.pop()
			class_scopes.pop()
			function_scopes.pop()
		elif text[i] == ";":
			pending_parameters = {}
			pending_class_scope = False
			pending_function_scope = False
		out.append(text[i])
		i += 1
	return "".join(out)


def remove_non_ascii(text: str) -> str:
	return "".join(ch for ch in text if ord(ch) < 128)


def move_inline_comments_to_next_line(line: str) -> list[str]:
	"""Split `code // comment` or `code /* comment` into separate lines."""
	slash_idx = line.find("//")
	block_idx = line.find("/*")

	indices = [i for i in (slash_idx, block_idx) if i != -1]
	if not indices:
		return [line]

	comment_start = min(indices)
	if comment_start == 0:
		return [line]

	code = line[:comment_start].rstrip()
	comment = line[comment_start:].strip()
	if not code:
		return [comment]
	return [code, comment]

def multi_replace(text: str, old: str, new: str) -> str:
	while old in text:
		text = text.replace(old, new)
	return text

def normalize_c_cpp(text: str) -> str:
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	normalized_lines: list[str] = []

	for raw_line in text.split("\n"):
		line = raw_line.lstrip()
		if not line or len(line) == 0:
			continue

		line = multi_replace(line, "---", "-")
		line = multi_replace(line, "\t", " ")
		line = multi_replace(line, "  ", " ")
		line = multi_replace(line, "====", "-")
		line = multi_replace(line, "+++", "+")
		line = multi_replace(line, "~~~", "~")
		line = multi_replace(line, "###", "#")

		if line == "{" and len(normalized_lines) > 0:
			# when seeing:
			#    for ()
			#      {
			# change to:
			#   for () {
			
			normalized_lines[-1] += "" + line
			continue


		split_lines = move_inline_comments_to_next_line(line)
		for split_line in split_lines:
			if split_line:
				normalized_lines.append(split_line)

	return "\n".join(normalized_lines) + ("\n" if normalized_lines else "")


def normalize_python_indentation(text: str) -> str:
	"""Convert complete four-space groups in leading whitespace to tabs."""
	normalized_lines: list[str] = []
	for line in text.splitlines(keepends=True):
		content = line.lstrip(" \t")
		indent_length = len(line) - len(content)
		indentation = line[:indent_length]
		normalized_lines.append(indentation.replace("    ", "\t") + content)
	return "".join(normalized_lines)


def strip_c_cpp_comments(text: str) -> str:
	"""Remove C/C++ comments while preserving literals and line boundaries."""
	out: list[str] = []
	i = 0
	state = "code"
	while i < len(text):
		ch = text[i]
		next_ch = text[i + 1] if i + 1 < len(text) else ""
		if state == "line_comment":
			if ch == "\n":
				out.append(ch)
				state = "code"
			i += 1
			continue
		if state == "block_comment":
			if ch == "*" and next_ch == "/":
				state = "code"
				i += 2
			elif ch == "\n":
				out.append(ch)
				i += 1
			else:
				i += 1
			continue
		if state == "code":
			if ch == "/" and next_ch == "/":
				state = "line_comment"
				i += 2
				continue
			if ch == "/" and next_ch == "*":
				state = "block_comment"
				i += 2
				continue
			if ch == '"':
				state = "string"
			elif ch == "'":
				state = "character"
			out.append(ch)
			i += 1
			continue
		out.append(ch)
		if ch == "\\" and i + 1 < len(text):
			out.append(text[i + 1])
			i += 2
			continue
		if (state == "string" and ch == '"') or (state == "character" and ch == "'"):
			state = "code"
		i += 1
	return "".join(out)


def strip_python_comments(text: str) -> str:
	"""Remove Python COMMENT tokens without treating '#' in strings as comments."""
	tokens = tokenize.generate_tokens(io.StringIO(text).readline)
	return tokenize.untokenize(token for token in tokens if token.type != tokenize.COMMENT)


def drop_blank_lines(text: str) -> str:
	"""Remove lines left empty after their comments were stripped."""
	lines = [line for line in text.splitlines() if line.strip()]
	return "\n".join(lines) + ("\n" if lines else "")


def is_etc_abbreviation(text: str, dot_index: int) -> bool:
	start = dot_index - 1
	while start >= 0 and text[start].isalpha():
		start -= 1
	word = text[start + 1 : dot_index].lower()
	return word == "etc"


def split_text_sentences(text: str) -> str:
	out_chars: list[str] = []
	i = 0
	while i < len(text):
		ch = text[i]
		out_chars.append(ch)

		if ch in {".", "!", "?"}:
			if ch == "." and is_etc_abbreviation(text, i):
				i += 1
				continue

			j = i + 1
			while j < len(text) and text[j] in {" ", "\t"}:
				j += 1

			if j < len(text) and text[j] not in {"\n", "\r"}:
				out_chars.append("\n")
				i = j
				continue

		i += 1

	return "".join(out_chars)


def process_file(
	path: Path,
	strip_comments: bool = False,
	abstract_symbols: bool = False,
) -> bool:
	original = path.read_text(encoding="utf-8", errors="ignore")
	updated = remove_non_ascii(original)

	suffix = path.suffix.lower()
	if abstract_symbols and suffix in CODE_EXTENSIONS:
		updated = abstract_code_symbols(updated, suffix)
	if suffix in C_EXTENSIONS:
		updated = normalize_c_cpp(updated)
		if strip_comments:
			updated = drop_blank_lines(strip_c_cpp_comments(updated))
	elif suffix in PYTHON_EXTENSIONS:
		updated = normalize_python_indentation(updated)
		if strip_comments:
			updated = drop_blank_lines(strip_python_comments(updated))
	elif suffix in TEXT_EXTENSIONS:
		updated = split_text_sentences(updated)

	if updated == original:
		return False

	path.write_text(updated, encoding="utf-8")
	return True


def main() -> int:
	parser = argparse.ArgumentParser(description="Normalize training data files.")
	parser.add_argument(
		"--dir",
		default="training_data2",
		help="Directory containing training files (default: training_data2)",
	)
	parser.add_argument(
		"--output-dir",
		help="Write a processed copy to this directory instead of modifying --dir",
	)
	parser.add_argument(
		"--strip-comments",
		action="store_true",
		help="Remove comments from C/C++ and Python source files",
	)
	parser.add_argument(
		"--abstract-symbols",
		action="store_true",
		help="Materialize the runtime identifier/string/MCP representation",
	)
	parser.add_argument(
		"--replace-output-dir",
		action="store_true",
		help="Replace an existing --output-dir before writing the processed copy",
	)
	args = parser.parse_args()

	root = Path(args.dir)
	if not root.is_dir():
		raise SystemExit(f"Directory not found: {root}")
	if args.output_dir:
		output_root = Path(args.output_dir)
		if output_root.exists():
			if not args.replace_output_dir:
				raise SystemExit(f"Output directory already exists: {output_root}")
			shutil.rmtree(output_root)
		shutil.copytree(root, output_root)
		root = output_root

	changed_count = 0
	file_count = 0
	for path in sorted(root.rglob("*")):
		if not path.is_file():
			continue
		file_count += 1
		if process_file(
			path,
			strip_comments=args.strip_comments,
			abstract_symbols=args.abstract_symbols,
		):
			changed_count += 1

	print(f"Processed {file_count} files in {root}; updated {changed_count}.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
