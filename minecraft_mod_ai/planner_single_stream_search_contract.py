from __future__ import annotations

import os
from functools import wraps
from typing import Any


_MARKER = "_mmm_single_stream_plan_search"


def _single_stream_active() -> bool:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip()
    if not raw:
        return False
    try:
        return int(raw) <= 1
    except ValueError:
        return True


def install(agentic_module: Any) -> None:
    """Do not serialize Best-of-N planner candidates onto one native decode slot.

    Explicit agentic-search mode remains authoritative. In the default ``auto`` mode,
    once the managed local llama runtime reports one active decode slot, generating
    two or three planner candidates only multiplies latency and token cost because the
    candidates cannot overlap. Repair search is intentionally untouched.
    """

    current = agentic_module._planner_candidate_count
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def planner_candidate_count(request: Any, stage: str) -> int:
        mode = agentic_module._mode()
        if mode == "on":
            return current(request, stage)
        if mode == "auto" and _single_stream_active():
            return 1
        return current(request, stage)

    setattr(planner_candidate_count, _MARKER, True)
    agentic_module._planner_candidate_count = planner_candidate_count


__all__ = ["_single_stream_active", "install"]
