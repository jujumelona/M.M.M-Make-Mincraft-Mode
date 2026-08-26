from __future__ import annotations

"""Compatibility verifier for trajectory-memory performance ownership.

The indexed SQLite/FTS hot path lives in trajectory_memory itself. This module must
never install runtime replacements again.
"""

from collections.abc import Callable
from typing import Any

_MARKER = "_mmm_research_memory_performance_v1"


def harden(
    trajectory_memory_module: Any,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    append = trajectory_memory_module.append_trajectory
    relevant = trajectory_memory_module.relevant_trajectories
    missing = [
        name
        for name, function in (
            ("append_trajectory", append),
            ("relevant_trajectories", relevant),
        )
        if not getattr(function, _MARKER, False)
    ]
    if missing:
        raise RuntimeError(
            "Trajectory memory performance must be implemented natively; missing: "
            + ", ".join(missing)
        )
    return append, relevant


__all__ = ["harden"]
