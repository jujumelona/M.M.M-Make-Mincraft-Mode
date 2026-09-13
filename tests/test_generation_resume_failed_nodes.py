from __future__ import annotations

from minecraft_mod_ai.generation_boundary_reconciliation import (
    _resume_failed_generation_nodes,
)
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


def test_resume_failed_generation_nodes_requeues_only_retryable_generation(tmp_path):
    nodes = (
        _node("generate-custom-00000000", "generate:custom"),
        _node("generate-custom-00000001", "generate:custom"),
        _node("generate-custom-00000002", "generate:custom"),
        _node("generate-custom-00000003", "generate:custom"),
        _node("validate-source", "validate:source"),
    )
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:proposal",
        graph_hash="sha256:graph",
        module_count=4,
        nodes=nodes,
    )
    ledger = DurableWorkLedger(
        tmp_path / "work-ledger.sqlite3",
        proposal_hash=plan.proposal_hash,
        graph_hash=plan.graph_hash,
    )
    ledger.sync_plan(plan)

    ledger.begin("generate-custom-00000000")
    ledger.fail("generate-custom-00000000", "RuntimeError: previous generation failure")

    ledger.begin("generate-custom-00000001")
    ledger.fail(
        "generate-custom-00000001",
        "missing required user input",
        input_required=True,
    )

    ledger.cancel("generate-custom-00000002", reason="cancelled by user")

    ledger.begin("generate-custom-00000003")
    ledger.succeed("generate-custom-00000003", {"status": "SUCCEEDED"})

    ledger.begin("validate-source")
    ledger.fail("validate-source", "validation failed")

    resumed = _resume_failed_generation_nodes(ledger, plan)

    assert resumed == ("generate-custom-00000000",)
    assert ledger.task("generate-custom-00000000")["state"] == "pending"
    assert ledger.task("generate-custom-00000000")["error"] is None
    assert ledger.task("generate-custom-00000001")["state"] == "input_required"
    assert ledger.task("generate-custom-00000002")["state"] == "cancelled"
    assert ledger.task("generate-custom-00000003")["state"] == "succeeded"
    assert ledger.task("validate-source")["state"] == "failed"
