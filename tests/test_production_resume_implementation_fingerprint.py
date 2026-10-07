from __future__ import annotations

from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphPlan,
    WorkState,
    _node,
)


def _plan(fingerprint: str) -> WorkGraphPlan:
    node = _node(
        "generate-content-00000000",
        "generate:content",
        (),
        {
            "kind": "module-shard",
            "generation_stage": "content",
            "production_implementation": fingerprint,
            "members": [],
        },
    )
    return WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:" + "1" * 64,
        graph_hash="sha256:" + fingerprint[-1] * 64,
        module_count=0,
        nodes=(node,),
    )


def test_resume_invalidates_succeeded_node_when_generator_fingerprint_changes(
    tmp_path,
) -> None:
    first = _plan("sha256:" + "a" * 64)
    ledger = DurableWorkLedger(
        tmp_path / "work-ledger.sqlite3",
        proposal_hash=first.proposal_hash,
        graph_hash=first.graph_hash,
    )
    ledger.sync_plan(first)
    ledger.begin("generate-content-00000000")
    ledger.succeed(
        "generate-content-00000000",
        {"schema_version": "test-receipt-v1", "status": "SUCCEEDED"},
    )
    assert ledger.task("generate-content-00000000")["state"] == WorkState.SUCCEEDED.value

    second = _plan("sha256:" + "b" * 64)
    receipt = ledger.sync_plan(second)

    assert receipt["invalidated_nodes"] == ("generate-content-00000000",)
    task = ledger.task("generate-content-00000000")
    assert task["state"] == WorkState.PENDING.value
    assert task["receipt"] is None
    assert task["output_hash"] is None
