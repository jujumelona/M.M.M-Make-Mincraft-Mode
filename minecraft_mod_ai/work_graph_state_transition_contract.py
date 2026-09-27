from __future__ import annotations

"""Validate source-owned monotonic durable-work transitions.

The transition semantics now live directly on DurableWorkLedger.  This compatibility
surface deliberately performs no runtime rebinding so package/test import order cannot
change ledger behavior.
"""

from typing import Any


def install(work_graph_module: Any) -> None:
    cls = work_graph_module.DurableWorkLedger
    required = (
        "succeed",
        "fail",
        "begin_checkpoint",
        "fail_checkpoint",
    )
    for name in required:
        value = getattr(cls, name)
        if getattr(value, "__module__", "") != work_graph_module.__name__:
            raise RuntimeError(
                f"Work-graph transition owner must be source-owned: {name}"
            )
        if not getattr(value, "_mmm_fenced_transition", False):
            raise RuntimeError(
                f"Work-graph transition fence is missing from source-owned {name}"
            )
    if not callable(getattr(cls, "invalidate_checkpoint", None)):
        raise RuntimeError("Work-graph checkpoint invalidation owner is missing.")


__all__ = ["install"]
