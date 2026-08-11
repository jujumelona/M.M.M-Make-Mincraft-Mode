from __future__ import annotations

from typing import Any

from .atomic_requirement_contract import install as _install_atomic_requirements
from .clean_room_verification_contract import install as _install_clean_room
from .repair_diagnostics_contract import install as _install_repair_diagnostics


def install(
    *,
    complete_planner_module: Any,
    orchestrator_module: Any,
    repair_module: Any,
    quality_evidence_module: Any,
    validation_module: Any,
) -> None:
    """Install MMM's final deterministic-control / narrow-agent architecture."""

    _install_atomic_requirements(
        complete_planner_module,
        orchestrator_module,
    )
    _install_repair_diagnostics(
        repair_module,
        validation_module,
    )
    _install_clean_room(
        orchestrator_module,
        quality_evidence_module,
        validation_module,
    )
