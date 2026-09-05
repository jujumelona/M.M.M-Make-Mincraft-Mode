from __future__ import annotations

"""Compatibility hook for the retired planner graph mutation layer.

Task-DAG construction, prerequisite gates, and dependency validation are owned directly
by ``evidence_first_planning``.  Runtime finalization still calls ``install`` for stable
composition ordering, but this package deliberately imports or mutates no planner
callables.
"""

_INSTALLED = False


def install() -> None:
    """Mark the retired compatibility layer installed without runtime mutation."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True


__all__ = ["install"]
