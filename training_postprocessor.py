#!/usr/bin/env python3
"""Normalize training files in training_data1 before training.

Rules:
1. Remove non-ASCII characters from every processed file.
2. For C/C++ files:
   - normalize line endings to '\n'
   - trim leading whitespace
   - drop empty lines
   - move inline comments to their own next line
3. For .md/.txt files:
	- split sentence punctuation ('.', '!', '?') into line breaks
	- keep 'etc.' inside sentences
"""

from __future__ import annotations

import argparse
from pathlib import Path


C_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
TEXT_EXTENSIONS = {".md", ".txt"}


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


def process_file(path: Path) -> bool:
	original = path.read_text(encoding="utf-8", errors="ignore")
	updated = remove_non_ascii(original)

	suffix = path.suffix.lower()
	if suffix in C_EXTENSIONS:
		updated = normalize_c_cpp(updated)
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
		default="training_data1",
		help="Directory containing training files (default: training_data1)",
	)
	args = parser.parse_args()

	root = Path(args.dir)
	if not root.is_dir():
		raise SystemExit(f"Directory not found: {root}")

	changed_count = 0
	file_count = 0
	for path in sorted(root.rglob("*")):
		if not path.is_file():
			continue
		file_count += 1
		if process_file(path):
			changed_count += 1

	print(f"Processed {file_count} files in {root}; updated {changed_count}.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
