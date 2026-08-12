from __future__ import annotations

from typing import Any

from . import agentic_optimization_contract as _agentic_module
from . import atomic_requirement_contract as _atomic_module
from . import custom_module_generator as _custom_module_generator_module
from . import production_contract as _production_contract_module
from . import work_graph as _work_graph_module
from .atomic_efficiency_contract import install as _install_atomic_efficiency
from .atomic_evidence_routing_contract import install as _install_atomic_routes
from .atomic_execution_policy_contract import install as _install_atomic_execution
from .atomic_planner_policy_contract import install as _install_atomic_planner_policy
from .atomic_playtest_evidence_contract import install as _install_atomic_playtest
from .atomic_quality_binding_contract import install as _install_atomic_quality
from .atomic_requirement_contract import install as _install_atomic_requirements
from .build_input_scope_contract import install as _install_build_input_scope
from .clean_room_verification_contract import install as _install_clean_room
from .custom_generation_search_contract import install as _install_custom_generation_search
from .orchestrator_jdt_gate_contract import install as _install_orchestrator_jdt_gate
from .planner_json_runtime_contract import install as _install_planner_json_runtime
from .planner_pagination_safety_contract import install as _install_planner_pagination_safety
from .planner_parser_safety_contract import install as _install_planner_parser_safety
from .repair_diagnostics_contract import install as _install_repair_diagnostics
from .repair_memory_budget_contract import install as _install_repair_memory_budget
from .required_gate_compatibility_contract import install as _install_gate_compatibility
from .scheduler_fairness_contract import install as _install_scheduler_fairness
from .semantic_reviewer_role_contract import install as _install_reviewer_role
from .visual_acceptance_scope_contract import install as _install_visual_scope


def install(
    *,
    complete_planner_module: Any,
    orchestrator_module: Any,
    repair_module: Any,
    validation_module: Any,
    quality_evidence_module: Any | None = None,
    quality_module: Any | None = None,
    game_design_module: Any | None = None,
    work_graph_module: Any | None = None,
    production_contract_module: Any | None = None,
    services_module: Any | None = None,
) -> None:
    """Install MMM's final deterministic-control / narrow-agent architecture.

    ``performance_final_tuning`` historically calls this installer with the compact
    five-module contract while ``integrated_contract_bootstrap`` supplies the wider
    package integration set. Both paths are active during normal package import, so
    the installer intentionally accepts both forms and resolves their shared modules
    here instead of relying on import-order-specific signatures.
    """

    del game_design_module, services_module
    quality = quality_evidence_module or quality_module
    if quality is None:
        raise TypeError("final architecture install requires a quality evidence module")
    graph = work_graph_module or _work_graph_module
    production = production_contract_module or _production_contract_module

    _install_build_input_scope(validation_module)
    _install_atomic_efficiency(_atomic_module)
    _install_atomic_routes(
        _atomic_module,
        production,
    )
    _install_reviewer_role(_atomic_module)
    _install_atomic_requirements(
        complete_planner_module,
        orchestrator_module,
    )
    _install_atomic_planner_policy(
        _atomic_module,
        complete_planner_module,
    )
    _install_atomic_execution(
        _atomic_module,
        orchestrator_module,
    )
    _install_atomic_quality(
        _atomic_module,
        quality,
        orchestrator_module,
    )
    _install_atomic_playtest(
        _atomic_module,
        quality,
        orchestrator_module,
    )
    _install_repair_diagnostics(
        repair_module,
        validation_module,
    )
    _install_orchestrator_jdt_gate(orchestrator_module)
    _install_clean_room(
        orchestrator_module,
        quality,
        validation_module,
    )
    _install_planner_json_runtime(complete_planner_module)
    _install_planner_parser_safety(complete_planner_module)
    _install_planner_pagination_safety(complete_planner_module)
    _agentic_module.install(
        complete_planner_module=complete_planner_module,
        repair_module=repair_module,
        work_graph_module=graph,
    )
    _install_custom_generation_search(_custom_module_generator_module)
    _install_repair_memory_budget(_agentic_module)
    _install_scheduler_fairness(graph)
    _install_visual_scope(orchestrator_module)
    _install_gate_compatibility(orchestrator_module)
