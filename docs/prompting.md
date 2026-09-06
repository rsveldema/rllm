# Prompt Mode

The language-aware tokenizer normalizes interactive source prompts and training
source in memory. Concrete identifiers receive indexed `<LOOP_n>`, `<LOCAL_n>`,
`<PARAM_n>`, or `<GLOBAL_n>` tokens according to their known scope; quoted literals
become `<STRING>`, and calls, imports, includes, or qualified accesses are
marked with `<MCP>` and `</MCP>`. Source text does not need to contain these
tokens explicitly; already-normalized markers are still accepted.

Interactive prompt mode keeps an in-memory history of submitted inputs. Use
the up and down arrow keys to navigate backward and forward through entries.
History lasts for the current prompt session and is not written to disk.
