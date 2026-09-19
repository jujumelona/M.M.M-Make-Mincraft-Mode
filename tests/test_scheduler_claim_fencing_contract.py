from __future__ import annotations

import inspect
import pytest

from minecraft_mod_ai import complete_orchestrator, work_graph


def _ledger_and_node(tmp_path):
    proposal_hash = "sha256:" + "1" * 64
    node = work_graph.WorkNode(
        node_id="node",
        stage="generate:test",
        input_hash="sha256:" + "2" * 64,
        dependencies=(),
        payload={"kind": "test", "resource_class": "cpu_io"},
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
    return ledger, node


def test_claim_fencing_is_source_owned_with_full_lifecycle_signature() -> None:
    owner = complete_orchestrator.CompleteProductionOrchestrator._run_work_node
    signature = inspect.signature(owner, follow_wrapped=False)

    assert "on_commit" in signature.parameters
    assert "on_abort" in signature.parameters
    source = inspect.getsource(owner)
    assert "_snapshot_claim(" in source
    assert "_commit_success(" in source
    assert "_fenced_fail(" in source


def test_run_work_node_preserves_commit_cache_and_abort_callbacks(tmp_path) -> None:
    ledger, node = _ledger_and_node(tmp_path)
    events: list[tuple[str, object, str]] = []

    receipt = complete_orchestrator.CompleteProductionOrchestrator._run_work_node(
        ledger,
        node,
        action=lambda: {"status": "PASS"},
        validate_cached=lambda _: False,
        on_commit=lambda value: events.append(
            ("commit", value["status"], ledger.task(node.node_id)["state"])
        ),
        on_abort=lambda value: events.append(
            ("abort", value, ledger.task(node.node_id)["state"])
        ),
    )
    assert receipt == {"status": "PASS"}
    assert events == [("commit", "PASS", work_graph.WorkState.SUCCEEDED.value)]

    cached = complete_orchestrator.CompleteProductionOrchestrator._run_work_node(
        ledger,
        node,
        action=lambda: pytest.fail("cached work node must not execute its action"),
        validate_cached=lambda _: True,
        on_commit=lambda value: events.append(
            ("cached", value["status"], ledger.task(node.node_id)["state"])
        ),
        on_abort=lambda value: events.append(
            ("abort", value, ledger.task(node.node_id)["state"])
        ),
    )
    assert cached == receipt
    assert events[-1] == ("cached", "PASS", work_graph.WorkState.SUCCEEDED.value)

    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    failed_ledger, failed_node = _ledger_and_node(failure_root)
    failed_events: list[tuple[str, object, str]] = []

    def fail_action() -> dict[str, object]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        complete_orchestrator.CompleteProductionOrchestrator._run_work_node(
            failed_ledger,
            failed_node,
            action=fail_action,
            validate_cached=lambda _: False,
            on_commit=lambda value: failed_events.append(
                ("commit", value, failed_ledger.task(failed_node.node_id)["state"])
            ),
            on_abort=lambda value: failed_events.append(
                ("abort", value, failed_ledger.task(failed_node.node_id)["state"])
            ),
        )

    assert failed_events == [
        ("abort", {}, work_graph.WorkState.RUNNING.value),
    ]
    assert (
        failed_ledger.task(failed_node.node_id)["state"]
        == work_graph.WorkState.FAILED.value
    )


def test_stale_attempt_cannot_complete_reclaimed_node(tmp_path) -> None:
    ledger, node = _ledger_and_node(tmp_path)
    first = ledger.claim_ready(
        "mmm-orchestrator",
        stages=("generate:test",),
        lease_seconds=900,
    )
    assert first is not None
    assert first["attempt"] == 1

    def action():
        # Simulate lease expiry/reclamation while the old action is still alive.
        with ledger._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE tasks
                SET state = ?, lease_owner = NULL, lease_until = NULL
                WHERE node_id = ?
                """,
                (work_graph.WorkState.PENDING.value, node.node_id),
            )
            connection.commit()
        second = ledger.claim_ready(
            "mmm-orchestrator",
            stages=("generate:test",),
            lease_seconds=900,
        )
        assert second is not None
        assert second["attempt"] == 2
        return {"status": "STALE"}

    with pytest.raises(work_graph.WorkGraphError, match="Stale worker claim rejected"):
        complete_orchestrator.CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=action,
            validate_cached=lambda _: False,
        )

    current = ledger.task(node.node_id)
    assert current["state"] == work_graph.WorkState.RUNNING.value
    assert current["attempt"] == 2
    assert current["receipt"] is None


def test_stale_failure_cannot_fail_newer_attempt(tmp_path) -> None:
    ledger, node = _ledger_and_node(tmp_path)
    first = ledger.claim_ready(
        "mmm-orchestrator",
        stages=("generate:test",),
        lease_seconds=900,
    )
    assert first is not None

    def action():
        with ledger._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE tasks
                SET state = ?, lease_owner = NULL, lease_until = NULL
                WHERE node_id = ?
                """,
                (work_graph.WorkState.PENDING.value, node.node_id),
            )
            connection.commit()
        second = ledger.claim_ready(
            "mmm-orchestrator",
            stages=("generate:test",),
            lease_seconds=900,
        )
        assert second is not None and second["attempt"] == 2
        raise RuntimeError("old worker failed late")

    with pytest.raises(RuntimeError, match="old worker failed late"):
        complete_orchestrator.CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=action,
            validate_cached=lambda _: False,
        )

    current = ledger.task(node.node_id)
    assert current["state"] == work_graph.WorkState.RUNNING.value
    assert current["attempt"] == 2
    assert current["error"] is None
