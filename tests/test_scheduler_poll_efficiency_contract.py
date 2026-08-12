from __future__ import annotations

from minecraft_mod_ai import work_graph
from minecraft_mod_ai.scheduler_parallel_safety_contract import install as install_scheduler


def _plan() -> work_graph.WorkGraphPlan:
    nodes = (
        work_graph.WorkNode(
            node_id="generate-a",
            stage="generate:content",
            input_hash="a",
            dependencies=(),
            payload={"kind": "module-shard", "resource_class": "cpu_io"},
        ),
        work_graph.WorkNode(
            node_id="generate-b",
            stage="generate:content",
            input_hash="b",
            dependencies=(),
            payload={"kind": "module-shard", "resource_class": "cpu_io"},
        ),
    )
    return work_graph.WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="proposal",
        graph_hash="graph",
        module_count=2,
        nodes=nodes,
    )


def test_generation_snapshot_expires_after_one_full_scan(tmp_path) -> None:
    install_scheduler(work_graph_module=work_graph, orchestrator_module=__import__(
        "minecraft_mod_ai.complete_orchestrator",
        fromlist=["CompleteProductionOrchestrator"],
    ))
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash="proposal",
    )
    ledger.sync_plan(_plan())

    first = [ledger.task(node_id)["state"] for node_id in ("generate-a", "generate-b")]
    assert first == ["pending", "pending"]
    assert getattr(ledger, "_mmm_generation_poll_snapshot", None) is None

    with ledger._connect() as connection:
        connection.execute(
            "UPDATE tasks SET state = ? WHERE node_id = ?",
            (work_graph.WorkState.SUCCEEDED.value, "generate-a"),
        )
        connection.commit()

    second = [ledger.task(node_id)["state"] for node_id in ("generate-a", "generate-b")]
    assert second == ["succeeded", "pending"]


def test_lane_claim_fences_any_partial_poll_snapshot(tmp_path) -> None:
    orchestrator = __import__(
        "minecraft_mod_ai.complete_orchestrator",
        fromlist=["CompleteProductionOrchestrator"],
    )
    install_scheduler(work_graph_module=work_graph, orchestrator_module=orchestrator)
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash="proposal",
    )
    ledger.sync_plan(_plan())

    # Leave a partially consumed read snapshot on purpose.
    assert ledger.task("generate-a")["state"] == "pending"
    assert getattr(ledger, "_mmm_generation_poll_snapshot", None) is not None

    claimed = ledger.claim_ready(
        "mmm-orchestrator",
        stages=("generate:content",),
        lease_seconds=60,
    )
    assert claimed is not None
    assert claimed["state"] == "running"
    assert claimed["lease_owner"]
    assert getattr(ledger, "_mmm_generation_poll_snapshot", None) is None
