from collections import defaultdict
import argparse
import re
import os

# End-of-word marker appended to every word before BPE training.
# This lets BPE learn subword units that are aware of word boundaries
# (e.g. "ing·" means the suffix "ing" at the end of a word).
# The marker is also emitted as its own token so the C++ tokenizer can
# recognise word-end positions in the input stream.
EOW_MARKER = '·'  # U+00B7 MIDDLE DOT

# read the sample data and create a dictionary of longest
# sequences of characters that appear in the data, and assign each sequence a unique ID.
# We then generate a C++ table for this for use in the neural network implementation. The
# table is sorted by sequence length, longest first, to ensure that we match the longest
# possible sequence when tokenizing input text.

training_file_extensions = [".cpp", ".cc", ".h", ".txt", ".text", ".md", ".json", ".yaml", ".yml"]

def read_all_files_in_directory(directory):
    all_text = ""
    for filename in os.listdir(directory):
        if filename.startswith("."):
            continue # skip hidden files
        for ext in training_file_extensions:
            if filename.endswith(ext):
                print(f"Reading file: {os.path.join(directory, filename)}")
                with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                    all_text += f.read()
                break
    return all_text


def get_unique_words(text: list[str]) -> list[str]:
    unique_words = set()
    for word in text:
        if word:
            unique_words.add(word)
    return list(unique_words)

def get_unique_words_with_frequency(text: list[str]) -> dict[str, int]:
    unique_words = defaultdict(int)
    for word in text:
        if word:
            unique_words[word] += 1
    return unique_words



def split_text_using_seperators(text, seperators):
    # split the text into words on the seperators, and also include the seperators as tokens:
    regex_pattern = '|'.join(map(re.escape, seperators))
    split = re.split(regex_pattern, text)
    return split

def compute_bpe_tokens(split: dict[str, int], num_merges: int = 500) -> list[str]:
    """
    Byte Pair Encoding: iteratively merge the most frequent adjacent character
    pair in the weighted word vocabulary until `num_merges` merges have been
    performed or no pair appears more than once.

    Returns the list of learned multi-character subword tokens (the merge results),
    not including single characters (those are already added elsewhere).
    """
    # Represent every word as a tuple of its characters, weighted by frequency.
    # The EOW_MARKER is appended to each word so BPE learns boundary-aware tokens.
    vocab: dict[tuple, int] = {}
    for word, freq in split.items():
        if len(word) >= 2:          # single chars are already in the token set
            key = tuple(word) + (EOW_MARKER,)
            vocab[key] = vocab.get(key, 0) + freq

    def get_pair_freqs(vocab: dict[tuple, int]) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = defaultdict(int)
        for symbols, freq in vocab.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_pair(pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
        merged = pair[0] + pair[1]
        new_vocab: dict[tuple, int] = {}
        for symbols, freq in vocab.items():
            new_syms: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_syms)] = freq
        return new_vocab

    learned: list[str] = []
    seen: set[str] = set()

    for _ in range(num_merges):
        pairs = get_pair_freqs(vocab)
        if not pairs:
            break
        best, best_freq = max(pairs.items(), key=lambda kv: kv[1])
        if best_freq < 2:
            break
        merged = best[0] + best[1]
        if merged not in seen:
            seen.add(merged)
            learned.append(merged)
        vocab = merge_pair(best, vocab)

    return learned

def is_all_uppercase(word: str) -> bool:
    return all(ch.isupper() for ch in word if ch.isalpha())

def is_all_lowercase(word: str) -> bool:
    return all(ch.islower() for ch in word if ch.isalpha())

def split_camel_case_words(split : list[str]) -> list[str]:
    # split camel case words into their components, and also include the original word as a token:
    split_camel_case = []
    for word in split:
        if not word:
            continue

        if is_all_uppercase(word):
            split_camel_case.append(word)
            continue

        if is_all_lowercase(word):
             split_camel_case.append(word)
             continue

        # if the first letter is uppercase and the rest are lowercase,
        # it's likely a proper noun or class name, so keep it as a single token:
        if word[0].isupper() and all(ch.islower() for ch in word[1:]):
             split_camel_case.append(word)
             continue

        # mixture of lower/uppercase, likely camel case, so split it:
        components = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word)
        split_camel_case.extend(components)
    return split_camel_case


def create_tokenizer_map(text, support_extra_latin_characters: bool = False) -> dict[str, int]:
    # create a combination of all sequences of characters that appear in the text, up to a certain length,
    # and count the frequency of each sequence. We will use this to create a tokenizer map that maps each sequence to a unique ID, sorted by length and frequency.
    seperators = ' \n\t(){}[]<>.,;:"\'`~?!@#$%^&*-_=+|\\/0123456789'

    # all seperators are tokens:
    tokens = []
    for ch in seperators:
        tokens.append(ch)
    tokens.append(EOW_MARKER)  # end-of-word marker used by BPE

    # Multi-character operators — added explicitly so longest-match-first
    # tokenization prefers them over their single-character constituents.
    # Without these, "==" would match only the first "=" and the second "="
    # would be skipped as unmatched, shortening the sequence and causing NaN loss.
    MULTI_CHAR_OPS = [
        '==', '!=', '<=', '>=',
        '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
        '<<', '>>', '<<=', '>>=',
        '->', '::', '...',
        '&&', '||', '!!',
    ]
    tokens.extend(MULTI_CHAR_OPS)

    # a-z, A-Z and accented latin characters (U+00C0–U+024F) as single-char tokens
    # so that every letter can always be tokenized without skipping.
    import unicodedata
    for cp in range(ord('A'), ord('Z') + 1):
        tokens.append(chr(cp))
    for cp in range(ord('a'), ord('z') + 1):
        tokens.append(chr(cp))

    if support_extra_latin_characters:
        for cp in range(0x00C0, 0x0250):   # Latin Extended-A/B + Latin-1 Supplement letters
            ch = chr(cp)
            if unicodedata.category(ch).startswith('L'):
                tokens.append(ch)


    # split the text into words on the seperators,
    # and also include the seperators as tokens:
    split = split_text_using_seperators(text, seperators)
    split = split_camel_case_words(split)
    split = get_unique_words_with_frequency(split)
    bpe_tokens = compute_bpe_tokens(split)
    tokens.extend(bpe_tokens)

    # Create a dictionary to count the frequency of each token.
    token_freq = defaultdict(int)
    for token in tokens:
        token_freq[token] += 1

    # Sort tokens by length (longest first) and then by frequency (most common first).
    sorted_tokens = sorted(token_freq.keys(), key=lambda x: (-len(x), -token_freq[x]))

    # Create a mapping of token to unique ID.
    tokenizer_map = {token: idx for idx, token in enumerate(sorted_tokens)}

    return tokenizer_map

def generate_cpp_table_impl(tokenizer_map) -> str:
    cpp_table = "// This file is generated by create_tokenizer_map.py. Do not edit manually.\n\n"
    cpp_table += "// Tokenizer map: maps string tokens to their corresponding TokenID enum values.\n"
    cpp_table += "// Sorted by token length (longest first) to ensure longest match during tokenization.\n"
    cpp_table += "#include <map>\n"
    cpp_table += "#include \"tokenizer_map.hpp\"\n\n"
    cpp_table += "namespace rllm {\n\n"
    cpp_table += "std::map<TokenID, TokenInfo> tokenizer_map = {\n"
    for token, idx in tokenizer_map.items():
        if token == " ":
            continue # skip space, we will handle it as a special case in the tokenizer
        is_eow = token.endswith(EOW_MARKER)
        stripped = token[:-len(EOW_MARKER)] if is_eow else token
        if not stripped:
            continue # skip the lone EOW_MARKER token
        if stripped == "\t":
            display_token = "\\t"
        elif stripped == "\n":
            display_token = "\\n"
        elif stripped == "\\":
            display_token = "\\\\"
        elif stripped == "\"":
            display_token = "\\\""
        else:
            display_token = stripped
        is_eow_str = "true" if is_eow else "false"
        cpp_table += f'    {{TokenID::TOK_{idx}, {{"{display_token}", {is_eow_str}}}}},\n'
    cpp_table += "};\n"
    cpp_table += "\n} // namespace rllm\n"
    return cpp_table

def generate_cpp_table_header(tokenizer_map) -> str:
    cpp_table = "#pragma once\n\n"
    cpp_table += "// This file is generated by create_tokenizer_map.py. Do not edit manually.\n\n"
    cpp_table += "#include <map>\n#include <cstring>\n\n"
    cpp_table += "namespace rllm {\n\n"
    cpp_table += "enum class TokenID {\n"
    for _token, idx in tokenizer_map.items():
        if _token == " ":
            continue # skip space, we will handle it as a special case in the tokenizer
        cpp_table += f'    TOK_{idx} = {idx},\n'
        if _token == "\n":
            cpp_table += f'    TOK_NEWLINE = TOK_{idx},\n'
        if _token == "\t":
            cpp_table += f'    TOK_TAB = TOK_{idx},\n'
        if _token == "0":
            cpp_table += f'    TOK_ZERO = TOK_{idx},\n'
        if _token == "1":
            cpp_table += f'    TOK_ONE = TOK_{idx},\n'
        if _token == "2":
            cpp_table += f'    TOK_TWO = TOK_{idx},\n'
        if _token == "3":
            cpp_table += f'    TOK_THREE = TOK_{idx},\n'
        if _token == "4":
            cpp_table += f'    TOK_FOUR = TOK_{idx},\n'
        if _token == "5":
            cpp_table += f'    TOK_FIVE = TOK_{idx},\n'
        if _token == "6":
            cpp_table += f'    TOK_SIX = TOK_{idx},\n'
        if _token == "7":
            cpp_table += f'    TOK_SEVEN = TOK_{idx},\n'
        if _token == "8":
            cpp_table += f'    TOK_EIGHT = TOK_{idx},\n'
        if _token == "9":
            cpp_table += f'    TOK_NINE = TOK_{idx},\n'
        if _token == ".":
            cpp_table += f'    TOK_DOT = TOK_{idx},\n'
        if _token == ",":
            cpp_table += f'    TOK_COMMA = TOK_{idx},\n'
        if _token == EOW_MARKER:
            cpp_table += f'    TOK_EOW = TOK_{idx},\n'
    cpp_table += "    START = TOK_0,\n"
    cpp_table += f"    MAX = {len(tokenizer_map)},\n"
    cpp_table += "    UNKNOWN_TOKEN_ID = -1\n"
    cpp_table += "};\n\n"
    cpp_table += "struct TokenInfo {\n"
    cpp_table += "    const char* str;\n"
    cpp_table += "    bool end_of_word;\n"
    cpp_table += "};\n\n"
    cpp_table += "extern std::map<TokenID, TokenInfo> tokenizer_map;\n"
    cpp_table += "\n} // namespace rllm\n"
    return cpp_table

def generate_cpp_table(tokenizer_map, cc_out: str, hpp_out: str):
    os.makedirs(os.path.dirname(cc_out),  exist_ok=True)
    os.makedirs(os.path.dirname(hpp_out), exist_ok=True)
    with open(cc_out, "w", encoding="utf-8") as f:
        cpp_table = generate_cpp_table_impl(tokenizer_map)
        f.write(cpp_table)
    with open(hpp_out, "w", encoding="utf-8") as f:
        cpp_table = generate_cpp_table_header(tokenizer_map)
        f.write(cpp_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc-out",  default="src/tokenizer_map.cc")
    parser.add_argument("--hpp-out", default="include/tokenizer_map.hpp")
    parser.add_argument("--support-extra-latin-characters", action="store_true", help="Include additional Latin characters (beyond basic ASCII) as single-character tokens")
    args = parser.parse_args()

    conatenated_text = read_all_files_in_directory("tokenizer_training_data")
    print(f"Total characters in corpus: {len(conatenated_text)}")
    tokenizer_map = create_tokenizer_map(conatenated_text, support_extra_latin_characters=args.support_extra_latin_characters)

    print(f"Total unique tokens: {len(tokenizer_map)}")

    generate_cpp_table(tokenizer_map, args.cc_out, args.hpp_out)