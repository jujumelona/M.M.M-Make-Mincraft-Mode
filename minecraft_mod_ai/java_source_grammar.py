"""Native raw-Java decoding from the same structural grammar as admission."""
from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _resource() -> dict[str, Any]:
    path = Path(__file__).with_name("data") / "java_source_grammar.json"
    return json.loads(path.read_text(encoding="utf-8"))


def native_java_source_grammar(frame: Mapping[str, Any]) -> str:
    resource = _resource()
    entry = resource["entries"][frame["kind"]]
    prefix = json.dumps(frame["prefix"], ensure_ascii=False)
    suffix = json.dumps(frame["suffix"], ensure_ascii=False)
    root = f"root ::= {prefix} {entry} {suffix}"
    if frame.get("empty_value") is not None:
        root += " | " + json.dumps(frame["empty_value"], ensure_ascii=False)
    grammar = resource["grammar"]
    # The generated PEG contains 244 Java identifier suffix productions of
    # [A-Za-z0-9_$]*. A 9B sampler can legally produce AliasHolder thousands
    # of times without ever reaching a statement or the fixed method suffix.
    # Bound only identifier continuation, not source statements or literals.
    unbounded = "[A-Za-z0-9_$]*"
    if unbounded not in grammar:
        raise ValueError("NATIVE_JAVA_GRAMMAR_IDENTIFIER_CONTRACT_MISSING")
    grammar = grammar.replace(unbounded, "[A-Za-z0-9_$]{0,127}")
    return root + "\n" + grammar
