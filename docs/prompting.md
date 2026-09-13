# Prompt Mode

The language-aware tokenizer normalizes interactive source prompts and training
source in memory. Identifiers use `<IDENTIFIER>`, quoted literals use `<STRING>`,
and integer literals use `<INTEGER>`. A separate source-local table stores their
spellings. Calls, imports, includes, and qualified accesses are marked with
`<MCP>` and `</MCP>`. Already-normalized markers are still accepted, but carry no
concrete value unless a table index is supplied.

The model predicts both token type and table index. Generated value tokens resolve
through the prompt's table; the model can select an existing entry but cannot
create a new spelling through the index head. See [tokenizer details](tokenizer.md).

Interactive prompt mode keeps an in-memory history of submitted inputs. Use
the up and down arrow keys to navigate backward and forward through entries.
History lasts for the current prompt session and is not written to disk.
