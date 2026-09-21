from __future__ import annotations

"""Separate retrieval evidence from host-selected mutation targeting."""

from collections.abc import Collection, Mapping
from typing import Any


def observed_context_may_bind(
    observed_context: Any,
    *,
    binding_enabled: bool,
) -> bool:
    return bool(binding_enabled and observed_context is not None)


def context_is_host_pinned(context: Any) -> bool:
    return bool(context is not None and getattr(context, "target_pinned", False))


def materialized_create_context(
    path: str,
    operation: str,
    arguments: Mapping[str, Any],
    create_operations: Collection[str],
    context_factory: Any,
) -> Any:
    if not path or operation not in create_operations:
        return None
    content = arguments.get("content")
    return context_factory(
        target_path=path,
        source_body=content if isinstance(content, str) else None,
        is_new_file=False,
        evidence_source="mutation_receipt",
        writable_paths=(path,),
        target_pinned=True,
    )
