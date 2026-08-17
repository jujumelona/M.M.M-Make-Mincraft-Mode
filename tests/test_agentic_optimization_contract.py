from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


def _node(node_id: str, resource_class: str) -> WorkNode:
    return WorkNode(
        node_id=node_id,
        stage="generate:test",
        input_hash=f"sha256:{node_id}",
        dependencies=(),
        payload={"kind": "test", "resource_class": resource_class},
        resource_class=resource_class,
    )


def test_verified_repair_memory_retrieves_similar_signature(tmp_path: Path) -> None:
    trace = {
        "signature": "cannot find symbol RegistryKey src/main/java/Test.java",
        "evidence": {"build_status": "FAIL"},
        "repair_pattern": [
            {
                "operation": "edit",
                "path": "src/main/java/Test.java",
                "repair_excerpt": "RegistryKey",
            }
        ],
        "winner_verifier": {"jdt_error_count": 0},
    }
    agentic._write_memory(tmp_path, trace)
    matches = agentic._read_memory(
        tmp_path,
        "cannot find symbol RegistryKey at src/main/java/Test.java",
    )
    assert matches
    assert matches[0]["similarity"] > 0.0
    assert matches[0]["repair_pattern"][0]["path"].endswith("Test.java")


def test_durable_claims_do_not_preclaim_one_scarce_lane(tmp_path: Path) -> None:
    nodes = (
        _node("asset-00", "image_gpu"),
        _node("asset-01", "image_gpu"),
        _node("asset-02", "image_gpu"),
        _node("llm-00", "llm"),
        _node("cpu-00", "cpu_io"),
    )
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:test-proposal",
        graph_hash="sha256:test-graph",
        module_count=0,
        nodes=nodes,
    )
    ledger = DurableWorkLedger(
        tmp_path / "work.sqlite3",
        proposal_hash=plan.proposal_hash,
        graph_hash=plan.graph_hash,
    )
    ledger.sync_plan(plan)

    first = ledger.claim_ready("test", stages=("generate:test",))
    second = ledger.claim_ready("test", stages=("generate:test",))
    third = ledger.claim_ready("test", stages=("generate:test",))

    assert first is not None and second is not None and third is not None
    resources = {
        first["payload"]["resource_class"],
        second["payload"]["resource_class"],
        third["payload"]["resource_class"],
    }
    assert resources == {"llm", "image_gpu", "cpu_io"}
    image_running = [
        task
        for task in (first, second, third)
        if task["payload"]["resource_class"] == "image_gpu"
    ]
    assert len(image_running) == 1
