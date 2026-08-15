from __future__ import annotations

"""Deprecated compatibility import.

Target-specific implementation evidence is now owned by platform_planning_contract
after the single platform selection owner has attached ``_platform_selection``.
This module deliberately installs nothing and cannot create a targetless fallback RAG.
"""

from typing import Any


def install(*, complete_planner_module: Any, central_module: Any, retrieval_module: Any) -> None:
    del complete_planner_module, central_module, retrieval_module


__all__ = ["install"]
