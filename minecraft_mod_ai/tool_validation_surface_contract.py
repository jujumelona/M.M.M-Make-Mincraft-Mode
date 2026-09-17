from __future__ import annotations

"""Validation-surface helpers for the core-owned native tool path.

The model-visible tool frontier and the host-authorized validation surface are
intentionally different capabilities. Hidden/stale schemas may be used to parse and
validate a tool name that already exists in the transcript, but they never become
model-visible and never grant execution authority. Execution remains guarded by the
current HostRunState phase/tool allowlist.

Native llama tool parsing owns this distinction directly. This module therefore
contains only deterministic schema-surface helpers plus a runtime assertion; it must
not monkey-patch completion, continuation, parser, retry, or transport functions.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _assert_unique_schema_names(
    schemas: Sequence[Any],
    *,
    surface: str,
) -> None:
    """Reject ambiguous same-name ownership inside one schema surface."""

    seen: set[str] = set()
    for schema in schemas:
        name = _tool_name(schema)
        if not name:
            continue
        if name in seen:
            raise RuntimeError(
                f"duplicate tool schema name {name!r} in {surface} surface"
            )
        seen.add(name)


def _validation_surface(
    visible: Sequence[Any],
    authorized: Sequence[Any],
) -> tuple[Any, ...]:
    """Merge parse-only schemas without overriding schemas shown this turn."""

    _assert_unique_schema_names(visible, surface="model-visible")
    _assert_unique_schema_names(authorized, surface="authorized-validation")
    result = list(visible)
    visible_names = {
        name
        for schema in visible
        if (name := _tool_name(schema))
    }
    for schema in authorized:
        name = _tool_name(schema)
        if name and name in visible_names:
            continue
        result.append(schema)
    return tuple(result)


def install() -> None:
    """Assert native core ownership; never mutate the live adapter or transport."""

    from .model_adapters import llama_cpp_adapter

    owner = getattr(llama_cpp_adapter, "_request_tool_schema_map", None)
    if not callable(owner) or not getattr(
        owner, "_mmm_core_validation_surface", False
    ):
        raise RuntimeError(
            "native llama adapter does not own the authorized tool validation surface"
        )


__all__ = [
    "_assert_unique_schema_names",
    "_validation_surface",
    "install",
]
