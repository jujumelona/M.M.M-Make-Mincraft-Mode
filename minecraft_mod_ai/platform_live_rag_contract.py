from __future__ import annotations

"""Compatibility entrypoint for the former live-RAG installer.

Target-optional official retrieval is now owned directly by :mod:`minecraft_mod_ai.retrieval`.
Keeping this no-op entrypoint preserves bootstrap compatibility without mutating retrieval,
central research, or parallel research functions at runtime.
"""

from typing import Any


def install(*, retrieval_module: Any) -> None:
    # Kept for callers that still invoke the historical bootstrap hook.  The module
    # argument is intentionally unused: target resolution and corpus reuse are native.
    _ = retrieval_module


__all__ = ["install"]
