from __future__ import annotations

"""Separate retrieval evidence from host-selected mutation targeting."""

from typing import Any


def observed_context_may_bind(
    observed_context: Any,
    *,
    binding_enabled: bool,
) -> bool:
    return bool(binding_enabled and observed_context is not None)


def context_is_host_pinned(context: Any) -> bool:
    return bool(context is not None and getattr(context, "target_pinned", False))
