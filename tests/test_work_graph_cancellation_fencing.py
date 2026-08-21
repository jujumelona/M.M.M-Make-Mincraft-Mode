from __future__ import annotations

import pytest

from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphError,
    WorkGraphPlan,
    WorkNode,
    run_checkpoint,
)


def _ledger(tmp_path) -> DurableWorkLedger:
    node = WorkNode(
        node_id="work",
        stage="generate:test",
        input_hash="sha256:work",
        dependencies=(),
        payload={"kind": "test"},
    )
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:cancellation-fencing",
        graph_hash="sha256:cancellation-fencing-graph",
        module_count=1,
        nodes=(node,),
    )
    ledger = DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    return ledger


def test_worker_failure_cannot_overwrite_run_cancellation(tmp_path) -> None:
    ledger = _ledger(tmp_path)

    def action() -> str:
        ledger.cancel_run(reason="stop now")
        raise RuntimeError("late worker failure")

    with pytest.raises(RuntimeError, match="late worker failure"):
        run_checkpoint(
            ledger,
            "work",
            action,
            encode=lambda value: {"value": value},
            decode=lambda receipt: str(receipt["value"]),
        )

    task = ledger.task("work")
    assert task["state"] == "cancelled"
    assert task["error"] == "stop now"


def test_worker_completion_cannot_turn_cancellation_into_failure(tmp_path) -> None:
    ledger = _ledger(tmp_path)

    def action() -> str:
        ledger.cancel_run(reason="stop now")
        return "late result"

    with pytest.raises(WorkGraphError, match="cannot succeed from state cancelled"):
        run_checkpoint(
            ledger,
            "work",
            action,
            encode=lambda value: {"value": value},
            decode=lambda receipt: str(receipt["value"]),
        )

    task = ledger.task("work")
    assert task["state"] == "cancelled"
    assert task["error"] == "stop now"


def test_fail_still_rejects_unknown_nodes(tmp_path) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(WorkGraphError, match="Unknown work node"):
        ledger.fail("missing", "boom")
