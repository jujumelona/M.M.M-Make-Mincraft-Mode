from __future__ import annotations

from minecraft_mod_ai import work_graph
from minecraft_mod_ai.scheduler_parallel_safety_contract import install as install_scheduler
from minecraft_mod_ai.scheduler_poll_efficiency_contract import install as install_poll


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


def _install_runtime_owners() -> None:
    orchestrator = __import__(
        "minecraft_mod_ai.complete_orchestrator",
        fromlist=["CompleteProductionOrchestrator"],
    )
    install_scheduler(work_graph_module=work_graph, orchestrator_module=orchestrator)
    install_poll(work_graph)


def test_poll_contract_leaves_task_and_claim_hot_paths_unwrapped() -> None:
    _install_runtime_owners()
    assert not getattr(work_graph.DurableWorkLedger.task, "_mmm_batched_generation_poll", False)
    assert not getattr(work_graph.DurableWorkLedger.claim_ready, "_mmm_poll_snapshot_fence", False)
    assert getattr(work_graph.DurableWorkLedger.sync_plan, "_mmm_reusable_connection_sync_plan", False)


def test_generation_task_reads_are_exact_without_snapshot(tmp_path) -> None:
    _install_runtime_owners()
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash="proposal",
    )
    ledger.sync_plan(_plan())

    assert ledger.task("generate-a")["state"] == "pending"
    with ledger._connect() as connection:
        connection.execute(
            "UPDATE tasks SET state = ? WHERE node_id = ?",
            (work_graph.WorkState.SUCCEEDED.value, "generate-a"),
        )
        connection.commit()
    assert ledger.task("generate-a")["state"] == "succeeded"
    assert not hasattr(ledger, "_mmm_generation_poll_snapshot")


def test_sync_plan_temp_tables_are_cleaned_on_reused_connection(tmp_path) -> None:
    _install_runtime_owners()
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash="proposal",
    )
    ledger.sync_plan(_plan())

    with ledger._connect() as connection:
        temp_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_temp_master WHERE type = 'table'"
            )
        }
    assert not temp_tables & {
        "affected_nodes",
        "changed_nodes",
        "desired_edges",
        "desired_tasks",
    }

    # Reusing the same thread-local connection must still allow another exact sync.
    ledger.sync_plan(_plan())
    assert ledger.task("generate-b")["state"] == "pending"
