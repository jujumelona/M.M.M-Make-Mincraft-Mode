from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.scheduler_parallel_safety_contract as safety
import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


safety.install(
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


def _ledger(tmp_path: Path) -> DurableWorkLedger:
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:shared-gpu",
        graph_hash="sha256:shared-gpu-graph",
        module_count=0,
        nodes=(
            _node("generate-assets-00000000", "generate:assets", "image_gpu"),
            _node("generate-custom-00000000", "generate:custom", "llm"),
        ),
    )
    ledger = DurableWorkLedger(
        tmp_path / "shared-gpu.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    return ledger


def test_local_shared_gpu_lane_claims_only_one_gpu_class_at_a_time(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    stages = ("generate:assets", "generate:custom")
    token = safety._SHARED_LOCAL_GPU_LANE.set(True)
    try:
        first = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        assert first is not None
        assert first["node_id"] == "generate-assets-00000000"

        # The LLM lane is independently free, but it shares the same physical GPU.
        # Do not mark it RUNNING just to block later on the router's GPU writer lock.
        assert ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60) is None
        assert ledger.task("generate-custom-00000000")["state"] == "pending"

        ledger.succeed(first["node_id"], {"status": "PASS"})
        second = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        assert second is not None
        assert second["node_id"] == "generate-custom-00000000"
    finally:
        safety._SHARED_LOCAL_GPU_LANE.reset(token)


def test_independent_gpu_classes_remain_parallel_when_profile_does_not_share_device(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    stages = ("generate:assets", "generate:custom")
    token = safety._SHARED_LOCAL_GPU_LANE.set(False)
    try:
        first = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        second = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        assert first is not None and second is not None
        assert {first["node_id"], second["node_id"]} == {
            "generate-assets-00000000",
            "generate-custom-00000000",
        }
    finally:
        safety._SHARED_LOCAL_GPU_LANE.reset(token)


def test_profile_detection_coalesces_only_local_exclusive_gpu_models() -> None:
    assert safety._profile_uses_shared_local_gpu("Qwen3.5-9B_6GB") is True
    assert safety._profile_uses_shared_local_gpu("fast_test") is False
