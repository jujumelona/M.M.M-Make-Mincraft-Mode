from __future__ import annotations

"""Lossless context-compaction binding for the model tool loop.

The adapter owns message compaction; this contract only binds that adapter to the
router once. Keeping the wrapper here prevents package bootstrap from containing
method-replacement implementation details.
"""

from functools import wraps
from typing import Any

from .small_model_compacting_adapter import CompactingAdapter


def install(model_router_module: Any) -> None:
    """Wrap the router tool loop with lossless context compaction exactly once."""
    cls = model_router_module.ModelRouter
    current = cls._generate_with_tools
    if getattr(current, "_mmm_lossless_context_compaction", False):
        return

    @wraps(current)
    def generate_with_compaction(
        self: Any,
        *,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        return current(
            self,
            adapter=CompactingAdapter(adapter),
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    generate_with_compaction._mmm_lossless_context_compaction = True  # type: ignore[attr-defined]
    cls._generate_with_tools = generate_with_compaction


__all__ = ["install"]
