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

from .tool_validation_surface import (
    assert_unique_schema_names,
    tool_name,
    validation_surface,
)





# Compatibility aliases for callers that imported the historical private helpers.
_tool_name = tool_name
_assert_unique_schema_names = assert_unique_schema_names
_validation_surface = validation_surface


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
