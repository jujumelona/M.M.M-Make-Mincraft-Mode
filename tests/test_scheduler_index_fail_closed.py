from __future__ import annotations

import pytest

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_orchestrator_support import CompleteProductionError
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


class _BrokenIndex:
    def update_files(self, _paths):
        raise RuntimeError("index write failed")

    def write_manifest(self):
        raise AssertionError("manifest must not run after update failure")


def test_shared_index_failure_cannot_publish_succeeded_state(tmp_path) -> None:
    node = WorkNode(
        node_id="node",
        stage="generate:content",
        input_hash="sha256:test",
        dependencies=(),
        payload={"resource_class": "cpu_io"},
        resource_class="cpu_io",
    )
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:index-fail",
        graph_hash="sha256:index-fail-graph",
        module_count=0,
        nodes=(node,),
    )
    ledger = DurableWorkLedger(tmp_path / "run.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)

    with pytest.raises(CompleteProductionError, match="Shared ProjectIndex commit failed"):
        CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=lambda: {
                "status": "PASS",
                "touched_paths": ["src/main/java/X.java"],
            },
            validate_cached=lambda _cached: False,
            shared_index=_BrokenIndex(),
        )

    task = ledger.task("node")
    assert task["state"] == "failed"
    assert not ledger.cached_receipt("node", input_hash=node.input_hash)
