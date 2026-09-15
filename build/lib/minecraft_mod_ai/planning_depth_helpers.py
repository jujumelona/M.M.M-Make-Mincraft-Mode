from __future__ import annotations

"""Compatibility installer for retired production-depth planner mutations.

The canonical evidence-first planner now owns semantic task templates and all
request-to-task dependency binding directly.  This module intentionally performs no
runtime rebinding; it remains only because the finalized runtime still calls its legacy
installer through the planner-integrity compatibility layer.
"""

_INSTALLED = False


def install() -> None:
    """Mark the retired compatibility layer installed without mutating planner APIs."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True


__all__ = ["install"]
