import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TEST_TRAINING_TEXT = 'Neural Network hpp Neural Network hpp'

import create_tokenizer_map as ctm


def _greedy_tokenize(text: str, sorted_tokens: list[str]) -> tuple[list[str], list[str]]:
    """Greedy longest-match tokenizer. Returns (matched_tokens, skipped_chars).

    Mirrors the C++ tokenizer in Corpus::get_token_ids: iterate the token map
    (sorted longest-first) and take the first match at each position.
    """
    matched, skipped = [], []
    ix = 0
    while ix < len(text):
        for tok in sorted_tokens:
            if text[ix : ix + len(tok)] == tok:
                matched.append(tok)
                ix += len(tok)
                break
        else:
            skipped.append(text[ix])
            ix += 1
    return matched, skipped


def _runtime_greedy_tokenize(
    text: str, tokenizer_map: dict[str, int]
) -> tuple[list[str], list[str]]:
    """Mirror generated C++ token spellings and end-of-word checks."""
    token_specs = []
    for token, token_id in tokenizer_map.items():
        is_eow = token.endswith(ctm.EOW_MARKER)
        spelling = token.removesuffix(ctm.EOW_MARKER) if is_eow else token
        if spelling and spelling != " ":
            token_specs.append((token_id, spelling, is_eow))
    token_specs.sort()

    matched, skipped = [], []
    ix = 0
    while ix < len(text):
        for _, spelling, is_eow in token_specs:
            if not text.startswith(spelling, ix):
                continue
            next_ix = ix + len(spelling)
            if (
                is_eow
                and next_ix < len(text)
                and (text[next_ix].isalnum() or text[next_ix] == "_")
            ):
                continue
            matched.append(spelling)
            ix = next_ix
            break
        else:
            if not text[ix].isspace():
                skipped.append(text[ix])
            ix += 1
    return matched, skipped


@pytest.fixture(scope="session")
def token_vocab():
    """Build a deterministic sorted token list once per session."""
    tm = ctm.create_tokenizer_map(TEST_TRAINING_TEXT)
    return sorted(tm.keys(), key=lambda t: -len(t))


# ---------------------------------------------------------------------------
# Tests for: #include "NeuralNetwork.hpp"
# ---------------------------------------------------------------------------

def test_neural_network_hpp_no_skipped_chars(token_vocab):
    """Every character in '"NeuralNetwork.hpp"' must be covered by a token."""
    _, skipped = _greedy_tokenize('"NeuralNetwork.hpp"', token_vocab)
    assert skipped == [], f"Characters not covered by any token: {skipped!r}"


def test_neural_network_hpp_token_sequence(token_vocab):
    """'"NeuralNetwork.hpp"' must tokenize to the expected sequence."""
    matched, _ = _greedy_tokenize('"NeuralNetwork.hpp"', token_vocab)
    assert matched == ['"', 'Neural', 'Network', '.', 'hpp', '"']


def test_hash_prefixed_word_is_learned_as_single_token():
    """Repeated hash-prefixed words should survive preprocessing as one token."""
    tokenizer_map = ctm.create_tokenizer_map("#xxxx #xxxx")
    assert "#xxxx" in tokenizer_map

    token_vocab = sorted(tokenizer_map.keys(), key=lambda t: -len(t))
    matched, skipped = _greedy_tokenize("#xxxx", token_vocab)
    assert skipped == []
    assert matched == ["#xxxx"]


def test_invalid_token_is_reserved_even_when_absent_from_training_text():
    tokenizer_map = ctm.create_tokenizer_map("abc abc")
    assert "INVALID" in tokenizer_map


@pytest.mark.parametrize("keyword", ["while", "for", "if", "switch", "return"])
def test_cpp_keywords_are_guaranteed_atomic_tokens(keyword):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert keyword + ctm.EOW_MARKER in tokenizer_map

    matched, skipped = _runtime_greedy_tokenize(keyword + " (value)", tokenizer_map)
    assert skipped == []
    assert matched[0] == keyword
    assert len(_runtime_greedy_tokenize(keyword, tokenizer_map)[0]) == 1


@pytest.mark.parametrize(
    "keyword",
    ["#define", "#include", "#ifdef", "#ifndef", "#pragma"],
)
def test_preprocessor_keywords_are_guaranteed_atomic_tokens(keyword):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert keyword + ctm.EOW_MARKER in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(keyword, tokenizer_map)
    assert skipped == []
    assert matched == [keyword]


@pytest.mark.parametrize(
    "keyword",
    ["def", "import", "lambda", "nonlocal", "yield"],
)
def test_python_keywords_are_guaranteed_atomic_tokens(keyword):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert keyword + ctm.EOW_MARKER in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(keyword, tokenizer_map)
    assert skipped == []
    assert matched == [keyword]


@pytest.mark.parametrize(
    "keyword",
    ["then", "fi", "esac", "done", "function"],
)
def test_shell_keywords_are_guaranteed_atomic_tokens(keyword):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert keyword + ctm.EOW_MARKER in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(keyword, tokenizer_map)
    assert skipped == []
    assert matched == [keyword]


@pytest.mark.parametrize(
    "keyword",
    ["fn", "impl", "match", "mut", "unsafe", "where"],
)
def test_rust_keywords_are_guaranteed_atomic_tokens(keyword):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert keyword + ctm.EOW_MARKER in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(keyword, tokenizer_map)
    assert skipped == []
    assert matched == [keyword]


@pytest.mark.parametrize(
    "keyword",
    ["interface", "instanceof", "record", "sealed", "synchronized", "throws"],
)
def test_java_keywords_are_guaranteed_atomic_tokens(keyword):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert keyword + ctm.EOW_MARKER in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(keyword, tokenizer_map)
    assert skipped == []
    assert matched == [keyword]


@pytest.mark.parametrize("prefix", ["w", "wh", "whi", "whil"])
def test_partial_keyword_prefixes_remain_tokenizable(prefix):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    if len(prefix) >= 2:
        assert prefix in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(prefix, tokenizer_map)
    assert skipped == []
    assert "".join(matched) == prefix
    assert "while" not in matched
    if len(prefix) >= 2:
        assert matched == [prefix]


@pytest.mark.parametrize("prefix", ["#d", "#de", "#def", "#defi", "#defin"])
def test_partial_preprocessor_prefixes_use_dedicated_tokens(prefix):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert prefix in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(prefix, tokenizer_map)
    assert skipped == []
    assert matched == [prefix]


@pytest.mark.parametrize("prefix", ["im", "imp", "impl", "inter", "interf"])
def test_partial_rust_and_java_prefixes_use_dedicated_tokens(prefix):
    tokenizer_map = ctm.create_tokenizer_map("unrelated corpus")
    assert prefix in tokenizer_map
    matched, skipped = _runtime_greedy_tokenize(prefix, tokenizer_map)
    assert skipped == []
    assert matched == [prefix]


def test_keyword_token_requires_identifier_boundary():
    tokenizer_map = ctm.create_tokenizer_map("meanwhile meanwhile")
    matched, skipped = _runtime_greedy_tokenize("while_value", tokenizer_map)
    assert skipped == []
    assert "".join(matched) == "while_value"
    assert matched[0] != "while"
