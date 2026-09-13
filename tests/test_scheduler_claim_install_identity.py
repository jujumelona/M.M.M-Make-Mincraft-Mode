from pathlib import Path

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.scheduler_parallel_safety_contract import install
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


def _node(node_id: str, stage: str) -> WorkNode:
    return WorkNode(
        node_id=node_id,
        stage=stage,
        input_hash=f"sha256:{node_id}",
        dependencies=(),
        payload={"kind": "test", "resource_class": "cpu_io"},
        resource_class="cpu_io",
    )


def test_install_keeps_direct_orchestrator_delegation_and_stage_serialization(
    tmp_path: Path,
) -> None:
    install(work_graph_module=work_graph_module, orchestrator_module=orchestrator_module)

    # Lane-aware orchestration is owned by DurableWorkLedger.claim_ready itself.  The
    # safety installer must not replace the method dynamically or rely on wrapper markers.
    assert "claim_orchestrator_ready" in DurableWorkLedger.claim_ready.__code__.co_names

    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:identity-reinstall",
        graph_hash="sha256:identity-reinstall-graph",
        module_count=0,
        nodes=(
            _node("a-content", "generate:content"),
            _node("b-content", "generate:content"),
            _node("c-system", "generate:system"),
        ),
    )
    ledger = DurableWorkLedger(tmp_path / "identity.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)
    stages = ("generate:content", "generate:system")

    first = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
    second = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)

    assert first is not None and first["node_id"] == "a-content"
    # CPU generation stages mutate shared stage state, so another content node must not
    # be admitted while a-content is running. A different serial stage remains eligible.
    assert second is not None and second["node_id"] == "c-system"
    assert ledger.task("b-content")["state"] == "pending"
