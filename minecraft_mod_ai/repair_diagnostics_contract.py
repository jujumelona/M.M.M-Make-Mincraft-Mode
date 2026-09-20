from __future__ import annotations

"""Repair diagnostic compatibility markers for source-owned RepairEngine behavior.

RepairEngine._signature and RepairEngine._context are implemented directly in
repair_engine.py. This module must never replace those functions at runtime: doing so
made reviewed source differ from executed behavior and silently discarded newer
repository-grounding/build-log logic.
"""

from typing import Any


def install(repair_module: Any) -> None:
    """Assert and annotate the source-owned repair implementation without rebinding it."""

    cls = repair_module.RepairEngine
    signature = cls._signature
    context = cls._context

    if not callable(signature) or not callable(context):
        raise RuntimeError(
            "Repair diagnostics contract requires source-owned _signature and _context."
        )
    if getattr(signature, "__module__", "") != repair_module.__name__:
        raise RuntimeError(
            "RepairEngine._signature was rebound outside repair_engine before "
            "repair diagnostics installation."
        )
    if getattr(context, "__module__", "") != repair_module.__name__:
        raise RuntimeError(
            "RepairEngine._context was rebound outside repair_engine before "
            "repair diagnostics installation."
        )
    if hasattr(signature, "__wrapped__") or hasattr(context, "__wrapped__"):
        raise RuntimeError(
            "Repair diagnostic source owners must not be hidden behind runtime wrappers."
        )

    signature._mmm_flattened_jdt = True  # type: ignore[attr-defined]
    context._mmm_flattened_jdt = True  # type: ignore[attr-defined]
    context._mmm_reuses_repair_project_index = True  # type: ignore[attr-defined]


__all__ = ["install"]
