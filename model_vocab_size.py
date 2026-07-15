#!/usr/bin/env python3

import json
import pathlib
import struct
import sys


path = pathlib.Path(sys.argv[1])
try:
    if path.suffix in {".st", ".safetensors"}:
        with path.open("rb") as file:
            header_len = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_len))
        print(header.get("__metadata__", {}).get("tokenizer_vocab_size", ""))
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as file:
            header = json.load(file)
        print(header.get("tokenizer_vocab_size", ""))
    else:
        print("")
except Exception as exc:
    print(f"ERROR:{exc}")
