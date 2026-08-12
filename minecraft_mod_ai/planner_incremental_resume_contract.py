from __future__ import annotations

from functools import wraps
from typing import Any


def install(incremental_module: Any) -> None:
    """Finalize a resumed pending queue as soon as its saved work is complete."""

    current = incremental_module._process_pending_batches
    if getattr(current, "_mmm_finalize_pending_queue", False):
        return

    @wraps(current)
    def process_and_finalize(*args: Any, **kwargs: Any) -> None:
        current(*args, **kwargs)
        checkpoint_path = kwargs.get("checkpoint_path")
        checkpoint_state = kwargs.get("checkpoint_state")
        saved_batches = kwargs.get("saved_batches")
        if checkpoint_path is None or not isinstance(checkpoint_state, dict):
            return
        if checkpoint_state.get("pending_batches"):
            return
        checkpoint_state["saved_batches"] = list(saved_batches or [])
        checkpoint_state["pending_patch"] = None
        checkpoint_state["status"] = (
            "complete" if checkpoint_state.get("page_complete") else "page_complete"
        )
        incremental_module._save_checkpoint(checkpoint_path, checkpoint_state)

    process_and_finalize._mmm_finalize_pending_queue = True  # type: ignore[attr-defined]
    incremental_module._process_pending_batches = process_and_finalize


__all__ = ["install"]
