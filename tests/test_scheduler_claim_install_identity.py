from functools import wraps
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


def test_install_reclaims_outermost_claim_after_wraps_copies_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install(work_graph_module=work_graph_module, orchestrator_module=orchestrator_module)
    installed = DurableWorkLedger.claim_ready
    bypass_target = getattr(installed, "__wrapped__", installed)

    @wraps(installed)
    def late_overlay(self, worker_id, *, stages=(), lease_seconds=900):
        return bypass_target(
            self,
            worker_id,
            stages=stages,
            lease_seconds=lease_seconds,
        )

    # functools.wraps copies the old marker. Marker-only idempotence therefore
    # cannot distinguish this bypassing overlay from the installed safety owner.
    assert getattr(late_overlay, "_mmm_parallel_lane_claim_version", 0) >= 2
    monkeypatch.setattr(DurableWorkLedger, "claim_ready", late_overlay)

    install(work_graph_module=work_graph_module, orchestrator_module=orchestrator_module)
    repaired = DurableWorkLedger.claim_ready
    assert repaired is not late_overlay
    assert getattr(repaired, "_mmm_parallel_lane_claim", False)

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
    assert second is not None and second["node_id"] == "c-system"
    assert ledger.task("b-content")["state"] == "pending"
