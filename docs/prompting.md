# Prompt Mode

The language-aware tokenizer normalizes interactive source prompts and training
source in memory. Concrete identifiers receive plain `<LOOP>`, `<LOCAL>`,
`<PARAM>`, `<FIELD>`, or `<GLOBAL>` tokens according to their known scope;
quoted literals become `<STRING>`. Each string-bearing marker is followed by an
explicit virtual `<STI>` token whose metadata points into the prompt's
`string_table_value`; prompt rendering shows this as `<LOCAL: name>` or
`<STRING: "value">`. Calls, imports, includes, or qualified accesses are marked
with `<MCP>` and `</MCP>`. Source text does not need to contain these tokens
explicitly; already-normalized markers are still accepted.

Interactive prompt mode keeps an in-memory history of submitted inputs. Use
the up and down arrow keys to navigate backward and forward through entries.
History lasts for the current prompt session and is not written to disk.
