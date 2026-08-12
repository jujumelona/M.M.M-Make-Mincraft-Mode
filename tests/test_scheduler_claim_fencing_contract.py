from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_orchestrator, work_graph
from minecraft_mod_ai.scheduler_claim_fencing_contract import install


def _ledger_and_node(tmp_path):
    install(
        work_graph_module=work_graph,
        orchestrator_module=complete_orchestrator,
    )
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
