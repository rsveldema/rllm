#!/usr/bin/env python3

import re
import sys


text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"\bMAX\s*=\s*(\d+)", text)
if not match:
    raise SystemExit(f"Could not find TokenID::MAX in {sys.argv[1]}")
print(match.group(1))
