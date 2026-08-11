from __future__ import annotations

from typing import Any

from . import atomic_requirement_contract as _atomic_module
from .atomic_efficiency_contract import install as _install_atomic_efficiency
from .atomic_quality_binding_contract import install as _install_atomic_quality
from .atomic_requirement_contract import install as _install_atomic_requirements
from .build_input_scope_contract import install as _install_build_input_scope
from .clean_room_verification_contract import install as _install_clean_room
from .repair_diagnostics_contract import install as _install_repair_diagnostics
from .semantic_reviewer_role_contract import install as _install_reviewer_role


def install(
    *,
    complete_planner_module: Any,
    orchestrator_module: Any,
    repair_module: Any,
    quality_evidence_module: Any,
    validation_module: Any,
) -> None:
    """Install MMM's final deterministic-control / narrow-agent architecture."""

    _install_build_input_scope(validation_module)
    _install_atomic_efficiency(_atomic_module)
    _install_reviewer_role(_atomic_module)
    _install_atomic_requirements(
        complete_planner_module,
        orchestrator_module,
    )
    _install_atomic_quality(
        _atomic_module,
        quality_evidence_module,
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
