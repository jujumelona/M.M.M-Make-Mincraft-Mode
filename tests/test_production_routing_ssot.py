from __future__ import annotations

import inspect

from minecraft_mod_ai import complete_orchestrator, validator, work_graph
from minecraft_mod_ai import production_generation_preflight as generation_preflight


def test_scheduler_dispatch_validator_share_canonical_routing_contract() -> None:
    scheduler = inspect.getsource(work_graph.build_production_work_plan)
    dispatch = inspect.getsource(
        complete_orchestrator.CompleteProductionOrchestrator._execute_generation_work
    )
    validator_source = inspect.getsource(validator._complete_production_routing)
    preflight = inspect.getsource(
        generation_preflight.validate_production_generation_project
    )

    assert "compile_production_routing(" in scheduler
    assert "production_routing_sha256" in scheduler
    assert "compile_production_routing(" in dispatch
    assert "STALE_PRODUCTION_ROUTING" in dispatch
    assert "compile_production_routing(" in validator_source
    assert "routing: ProductionRoutingSnapshot | None" in inspect.signature(
        generation_preflight.validate_production_generation_project
    ).__str__() or "routing" in preflight


def test_orchestrator_does_not_rederive_artifact_ownership() -> None:
    dispatch = inspect.getsource(
        complete_orchestrator.CompleteProductionOrchestrator._execute_generation_work
    )

    assert "artifact_owner_module_ids(" not in dispatch
    assert 'route.owner == "artifact_graph"' in dispatch
    assert "ARTIFACT_JOB_ROUTING_MISMATCH" in dispatch
