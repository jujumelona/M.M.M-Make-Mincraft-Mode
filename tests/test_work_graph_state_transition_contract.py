from __future__ import annotations

import pytest

from minecraft_mod_ai import work_graph
from minecraft_mod_ai.work_graph_state_transition_contract import install


def _ledger_with_node(tmp_path):
    install(work_graph)
    proposal_hash = "sha256:" + "1" * 64
    node = work_graph.WorkNode(
        node_id="node",
        stage="generate:test",
        input_hash="sha256:" + "2" * 64,
        dependencies=(),
        payload={"kind": "test"},
    )
    plan = work_graph.WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash=proposal_hash,
        graph_hash="sha256:" + "3" * 64,
        module_count=0,
        nodes=(node,),
    )
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash=proposal_hash,
    )
    ledger.sync_plan(plan)
    return ledger


def test_failed_task_cannot_be_failed_again_from_stopped_state(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    ledger.begin("node")
    assert ledger.fail("node", "first")["state"] == "failed"

    with pytest.raises(work_graph.WorkGraphError, match="cannot fail from state failed"):
        ledger.fail("node", "late")


def test_cancelled_task_is_not_overwritten_by_late_failure(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    ledger.begin("node")
    ledger.cancel("node")

    result = ledger.fail("node", "late worker")
    assert result["state"] == "cancelled"


def test_successful_checkpoint_cannot_be_overwritten_by_late_failure(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    input_hash = "sha256:" + "4" * 64
    ledger.begin_checkpoint("checkpoint", stage="test", input_hash=input_hash)
    ledger.succeed_checkpoint(
        "checkpoint",
        input_hash=input_hash,
        receipt={"status": "PASS"},
    )

    with pytest.raises(work_graph.WorkGraphError, match="cannot fail from state succeeded"):
        ledger.fail_checkpoint(
            "checkpoint",
            input_hash=input_hash,
            error="late failure",
        )
    assert ledger.cached_checkpoint(
        "checkpoint",
        input_hash=input_hash,
    ) == {"status": "PASS"}


def test_checkpoint_success_uses_statement_rowcount_not_connection_total(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    ledger.begin("node")
    ledger.fail("node", "unrelated change")
    ledger.begin_checkpoint(
        "checkpoint",
        stage="test",
        input_hash="sha256:" + "5" * 64,
    )

    with pytest.raises(work_graph.WorkGraphError, match="changed while running"):
        ledger.succeed_checkpoint(
            "checkpoint",
            input_hash="sha256:" + "6" * 64,
            receipt={"status": "PASS"},
        )
