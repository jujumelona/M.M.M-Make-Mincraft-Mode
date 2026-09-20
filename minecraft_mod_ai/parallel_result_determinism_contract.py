from __future__ import annotations

"""Compatibility assertion for source-owned generation result ordering."""

from typing import Any


def install(*, orchestrator_module: Any) -> None:
    current = orchestrator_module.CompleteProductionOrchestrator._execute_generation_work
    if getattr(current, "__module__", "") != orchestrator_module.__name__:
        raise RuntimeError(
            "Parallel result determinism requires source-owned _execute_generation_work."
        )
    if hasattr(current, "__wrapped__"):
        raise RuntimeError(
            "Parallel result determinism cannot hide the generation owner behind a wrapper."
        )
    current._mmm_parallel_result_determinism = True  # type: ignore[attr-defined]


__all__ = ["install"]
