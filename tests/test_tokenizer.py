import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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


@pytest.fixture(scope="session")
def token_vocab():
    """Build the sorted token list once per session (longest tokens first)."""
    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        text = ctm.read_all_files_in_directory("corpus")
        tm = ctm.create_tokenizer_map(text)
    finally:
        os.chdir(old_cwd)
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
