from __future__ import annotations

"""Compatibility adapter for the retired standalone scheduler fairness contract.

Fairness, lease ownership, lane capacity, and shared-GPU exclusion are implemented
only by scheduler_parallel_safety_contract. This module preserves the historical
one-argument installer for older callers without adding another claim wrapper.
"""

from typing import Any


def install(work_graph_module: Any) -> None:
    from . import complete_orchestrator
    from .scheduler_parallel_safety_contract import install as install_scheduler_safety

    install_scheduler_safety(
        work_graph_module=work_graph_module,
        orchestrator_module=complete_orchestrator,
    )


__all__ = ["install"]
