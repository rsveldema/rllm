#!/usr/bin/env python3
"""Generate a token-safe concept graph from a seed graph and source code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


DECLARATIONS = re.compile(
    r"\b(class|struct|union|enum(?:\s+class)?|interface|concept|def|function)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
KINDS = {"def": "function", "enum class": "enum"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java", ".py", ".rs"}
NON_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|#[^\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def generate(base: Path, token_map_path: Path, sources: list[Path]) -> dict[str, object]:
    document = json.loads(base.read_text(encoding="utf-8"))
    concepts = document.get("concepts")
    relationships = document.get("relationships")
    if not isinstance(concepts, list) or not all(isinstance(item, str) for item in concepts):
        raise ValueError("base graph concepts must be an array of strings")
    if not isinstance(relationships, list):
        raise ValueError("base graph relationships must be an array")
    vocabulary = set(json.loads(token_map_path.read_text(encoding="utf-8")))
    concept_set = set(concepts)
    initial_concept_count = len(concept_set)
    relation_set = {
        (entry[0], entry[1], float(entry[2]))
        for entry in relationships
        if isinstance(entry, list) and len(entry) == 3
    }

    scanned_files = 0
    discovered_facts: set[tuple[str, str, str, bool]] = set()
    for source in sources:
        files = (path for path in source.rglob("*") if path.is_file()) if source.is_dir() else (source,)
        for path in files:
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            scanned_files += 1
            code = NON_CODE.sub(lambda match: "\n" * match.group(0).count("\n"), text)
            for match in DECLARATIONS.finditer(code):
                kind = KINDS.get(match.group(1), match.group(1))
                identifier = match.group(2)
                concept_kind = "class-name" if kind in {"class", "struct", "union", "interface"} else kind
                discovered_facts.add((identifier, "is-a", concept_kind, identifier in vocabulary))
                # Subword rows are shared by unrelated names, so only exact vocabulary
                # entries are safe targets for static embedding initialization.
                if concept_kind not in concept_set or identifier not in vocabulary or len(identifier) < 2:
                    continue
                if identifier not in concept_set:
                    concepts.append(identifier)
                    concept_set.add(identifier)
                relation_set.add((identifier, concept_kind, 0.9))

    document["concepts"] = concepts
    document["relationships"] = [list(item) for item in sorted(relation_set)]
    document["generated"] = {
        "scanned_files": scanned_files,
        "exact_token_identifiers": len(concept_set) - initial_concept_count,
        "discovered_facts": len(discovered_facts),
    }
    document["discovered_facts"] = [
        {"subject": subject, "relation": relation, "object": object_, "exact_token": exact_token}
        for subject, relation, object_, exact_token in sorted(discovered_facts)
    ]
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--token-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = generate(args.base, args.token_map, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote embedding prefill graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
