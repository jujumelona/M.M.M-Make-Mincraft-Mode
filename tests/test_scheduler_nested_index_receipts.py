from __future__ import annotations

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.scheduler_parallel_safety_contract import _receipt_touched_paths
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


class _Index:
    def __init__(self, ledger: DurableWorkLedger) -> None:
        self.ledger = ledger
        self.paths: tuple[str, ...] = ()
        self.events: list[str] = []

    def update_files(self, paths):
        assert self.ledger.task("nested")["state"] == "running"
        self.paths = tuple(paths)
        self.events.append("index-update")

    def write_manifest(self):
        assert self.ledger.task("nested")["state"] == "running"
        self.events.append("index-manifest")


def _node() -> WorkNode:
    return WorkNode(
        node_id="nested",
        stage="generate:content",
        input_hash="sha256:nested",
        dependencies=(),
        payload={"kind": "module-shard", "resource_class": "cpu_io"},
        resource_class="cpu_io",
    )


def _nested_receipt() -> dict:
    return {
        "schema_version": "mmm/generation-work-node-v1",
        "status": "SUCCEEDED",
        "receipts": [
            {
                "schema_version": "mmm/extended-content-v2",
                "status": "GENERATED",
                "touched_paths": ["src/main/java/A.java"],
            },
            {
                "schema_version": "mmm/system-pack-generation-v5",
                "status": "GENERATED",
                "files": ["src/main/java/B.java"],
                "receipts": {
                    "metadata": {
                        "schema_version": "mmm/source-patch-receipt-v1",
                        "status": "APPLIED",
                        "operations": [
                            {
                                "operation": "replace",
                                "path": "src/main/resources/fabric.mod.json",
                            },
                            {
                                "operation": "delete",
                                "path": "src/main/java/Old.java",
                            },
                        ],
                    }
                },
            },
        ],
    }


def test_nested_generator_receipts_expose_all_touched_source_paths() -> None:
    assert _receipt_touched_paths(_nested_receipt()) == (
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/resources/fabric.mod.json",
        "src/main/java/Old.java",
    )


def test_nested_paths_are_indexed_before_node_success(tmp_path) -> None:
    node = _node()
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:nested",
        graph_hash="sha256:nested-graph",
        module_count=0,
        nodes=(node,),
    )
    ledger = DurableWorkLedger(tmp_path / "run.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)
    index = _Index(ledger)

    receipt = CompleteProductionOrchestrator._run_work_node(
        ledger,
        node,
        action=_nested_receipt,
        validate_cached=lambda _cached: False,
        shared_index=index,
    )

    assert receipt["status"] == "SUCCEEDED"
    assert index.paths == (
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/resources/fabric.mod.json",
        "src/main/java/Old.java",
    )
    # The manifest is persisted once at the generation phase boundary, not
    # once per node. The in-memory index must be updated before ledger success.
    assert index.events == ["index-update"]
    assert ledger.task("nested")["state"] == "succeeded"


def test_index_failure_prevents_dependency_visible_node_success(tmp_path) -> None:
    import pytest

    node = _node()
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:nested",
        graph_hash="sha256:nested-graph",
        module_count=0,
        nodes=(node,),
    )
    ledger = DurableWorkLedger(tmp_path / "run.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)

    class FailingIndex(_Index):
        def update_files(self, paths):
            super().update_files(paths)
            raise OSError("index could not read modified source")

    index = FailingIndex(ledger)
    with pytest.raises(Exception, match="Shared ProjectIndex commit failed"):
        CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=_nested_receipt,
            validate_cached=lambda _cached: False,
            shared_index=index,
        )
    assert ledger.task("nested")["state"] != "succeeded"
