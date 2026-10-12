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
    return root + "\n" + resource["grammar"]
