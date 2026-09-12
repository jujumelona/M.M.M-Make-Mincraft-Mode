from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.scheduler_parallel_safety_contract import install
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


install(
    work_graph_module=work_graph_module,
    orchestrator_module=orchestrator_module,
)


def _node(node_id: str, stage: str, resource_class: str) -> WorkNode:
    return WorkNode(
        node_id=node_id,
        stage=stage,
        input_hash=f"sha256:{node_id}",
        dependencies=(),
        payload={"kind": "test", "resource_class": resource_class},
        resource_class=resource_class,
    )


def _ledger(tmp_path: Path, *nodes: WorkNode) -> DurableWorkLedger:
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:claim-efficiency",
        graph_hash="sha256:claim-efficiency-graph",
        module_count=0,
        nodes=nodes,
    )
    ledger = DurableWorkLedger(
        tmp_path / "claim-efficiency.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    return ledger


def test_successful_claim_scans_ready_queue_once_without_active_stage_prescan(
    tmp_path: Path,
) -> None:
    ledger = _ledger(
        tmp_path,
        _node("a-content", "generate:content", "cpu_io"),
        _node("b-system", "generate:system", "cpu_io"),
    )
    connection = ledger._connect()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        claimed = ledger.claim_ready(
            "mmm-orchestrator",
            stages=("generate:content", "generate:system"),
            lease_seconds=60,
        )
    finally:
        connection.set_trace_callback(None)

    assert claimed is not None
    normalized = [" ".join(statement.split()).upper() for statement in statements]
    ready_scans = [statement for statement in normalized if "WITH READY AS" in statement]
    active_stage_prescans = [
        statement
        for statement in normalized
        if "SELECT DISTINCT STAGE" in statement and "FROM TASKS" in statement
    ]

    assert len(ready_scans) == 1, statements
    assert active_stage_prescans == [], statements
