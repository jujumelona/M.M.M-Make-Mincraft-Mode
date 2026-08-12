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


def test_pending_task_can_be_marked_input_required_without_fake_execution(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)

    result = ledger.fail(
        "node",
        "external evidence is required",
        input_required=True,
    )

    assert result["state"] == "input_required"
    assert result["attempt"] == 0
    assert result["error"] == "external evidence is required"


def test_cancelled_task_is_not_overwritten_by_late_failure(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    ledger.begin("node")
    ledger.cancel("node")

    result = ledger.fail("node", "late worker")
    assert result["state"] == "cancelled"


def test_successful_task_allows_same_receipt_but_rejects_different_late_receipt(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    ledger.begin("node")
    first = ledger.succeed("node", {"status": "PASS", "winner": "new"})
    assert first["state"] == "succeeded"

    same = ledger.succeed("node", {"status": "PASS", "winner": "new"})
    assert same["receipt"] == {"status": "PASS", "winner": "new"}

    with pytest.raises(work_graph.WorkGraphError, match="different receipt"):
        ledger.succeed("node", {"status": "PASS", "winner": "stale"})
    assert ledger.task("node")["receipt"] == {
        "status": "PASS",
        "winner": "new",
    }


def test_checkpoint_cannot_be_started_twice_while_running(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    input_hash = "sha256:" + "7" * 64
    ledger.begin_checkpoint("checkpoint", stage="test", input_hash=input_hash)

    with pytest.raises(work_graph.WorkGraphError, match="already running"):
        ledger.begin_checkpoint("checkpoint", stage="test", input_hash=input_hash)


def test_successful_checkpoint_cannot_be_restarted_for_same_input(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    input_hash = "sha256:" + "8" * 64
    receipt = {"status": "PASS", "proof": "stable"}
    ledger.begin_checkpoint("checkpoint", stage="test", input_hash=input_hash)
    ledger.succeed_checkpoint(
        "checkpoint",
        input_hash=input_hash,
        receipt=receipt,
    )

    with pytest.raises(work_graph.WorkGraphError, match="already succeeded"):
        ledger.begin_checkpoint("checkpoint", stage="test", input_hash=input_hash)
    assert ledger.cached_checkpoint(
        "checkpoint",
        input_hash=input_hash,
    ) == receipt


def test_named_checkpoint_rebuilds_only_after_cached_output_validation_fails(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    calls = []

    first = work_graph.run_named_checkpoint(
        ledger,
        "artifact",
        stage="test",
        input_value={"source": "same"},
        action=lambda: calls.append("first") or {"path": "missing"},
        encode=lambda value: value,
        decode=lambda value: value,
        validate_cached=lambda value: value.get("path") == "exists",
    )
    assert first == {"path": "missing"}

    second = work_graph.run_named_checkpoint(
        ledger,
        "artifact",
        stage="test",
        input_value={"source": "same"},
        action=lambda: calls.append("second") or {"path": "exists"},
        encode=lambda value: value,
        decode=lambda value: value,
        validate_cached=lambda value: value.get("path") == "exists",
    )

    assert second == {"path": "exists"}
    assert calls == ["first", "second"]
    input_hash = work_graph._hash_json({"source": "same"})
    assert ledger.cached_checkpoint("artifact", input_hash=input_hash) == {
        "path": "exists"
    }


def test_running_checkpoint_rejects_changed_input(tmp_path) -> None:
    ledger = _ledger_with_node(tmp_path)
    ledger.begin_checkpoint(
        "checkpoint",
        stage="test",
        input_hash="sha256:" + "9" * 64,
    )

    with pytest.raises(work_graph.WorkGraphError, match="another attempt is running"):
        ledger.begin_checkpoint(
            "checkpoint",
            stage="test",
            input_hash="sha256:" + "a" * 64,
        )


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
