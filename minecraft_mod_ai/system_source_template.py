"""Render package-owned Java source templates from immutable text resources."""
from __future__ import annotations

import codecs
from functools import lru_cache
from string import Formatter

from .task_template_catalog import RUNTIME_TEMPLATE_ROOT

_ROOT = (RUNTIME_TEMPLATE_ROOT / "implementation" / "system_pack").resolve()
_ALLOWED = frozenset({
    "persistent_store",
    "config_loader",
    "class_skill",
    "economy",
    "party_guild",
    "quest",
    "gui_networking",
})


@lru_cache(maxsize=None)
def _source(name: str) -> str:
    if name not in _ALLOWED:
        raise ValueError(f"SYSTEM_SOURCE_TEMPLATE: unknown template {name!r}")
    path = (_ROOT / f"{name}.java.fmt").resolve()
    if not path.is_relative_to(_ROOT) or not path.is_file():
        raise ValueError(f"SYSTEM_SOURCE_TEMPLATE: missing template {name!r}")
    return path.read_text(encoding="utf-8")


def render_system_source(name: str, **values: str) -> str:
    raw = _source(name)
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(raw)
        if field_name is not None
    }
    if fields != set(values):
        raise ValueError(
            f"SYSTEM_SOURCE_TEMPLATE: {name} expects {sorted(fields)}, got {sorted(values)}"
        )
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError(f"SYSTEM_SOURCE_TEMPLATE: {name} values must be non-empty strings")
    rendered = raw.format(**values)
    # Resource text preserves the original static Python-literal escaping (not code).
    # Decode those escapes after placeholder substitution so generated Java is byte-for-byte
    # equivalent to the former f-string output without executing or storing Python source.
    return codecs.decode(rendered.encode("utf-8"), "unicode_escape")


__all__ = ["render_system_source"]
