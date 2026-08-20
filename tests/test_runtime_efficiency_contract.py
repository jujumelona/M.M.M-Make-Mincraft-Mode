from __future__ import annotations

import concurrent.futures
import inspect

import minecraft_mod_ai.custom_generation_search_contract as custom_search
import minecraft_mod_ai.scheduler_parallel_safety_contract as safety
import minecraft_mod_ai.work_graph as work_graph
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


def _node(node_id: str, resource_class: str) -> WorkNode:
    return WorkNode(
        node_id=node_id,
        stage="generate:custom" if resource_class == "llm" else "generate:assets",
        input_hash=f"sha256:{node_id}",
        dependencies=(),
        payload={"kind": "test", "resource_class": resource_class},
        resource_class=resource_class,
    )


def test_stdlib_executor_is_not_globally_patched(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    assert not getattr(
        concurrent.futures.ThreadPoolExecutor,
        "_mmm_exact_llm_executor",
        False,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="llm",
    ) as pool:
        assert pool._max_workers == 1
    assert safety._capacities()["llm"] == 3


def test_generation_scheduler_uses_owner_capacities_and_event_wait() -> None:
    source = inspect.getsource(
        inspect.unwrap(CompleteProductionOrchestrator._execute_generation_work)
    )
    assert "scheduler_safety._capacities()" in source
    assert "FIRST_COMPLETED" in source
    assert "while len(claimed_batch) < 8" not in source
    assert "time.sleep(0.05)" not in source
    assert "node_by_id" in source
    assert "_acquire_path_lock" not in source


def test_runtime_efficiency_preserves_existing_dependency_wave_shards(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    modules = tuple(
        ProductionModule(
            module_id=f"custom_{index}",
            kind="custom_java",
            config={"summary": str(index)},
        )
        for index in range(10)
    )
    policy = type("Policy", (), {"entity_shard_size": 24, "java_shard_size": 48})()
    shards = list(work_graph._module_shards(modules, policy=policy))
    assert [stage for stage, _members in shards] == ["custom", "custom", "custom"]
    assert [len(members) for _stage, members in shards] == [4, 4, 2]


def test_one_slot_keeps_bounded_custom_shards(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    modules = tuple(
        ProductionModule(
            module_id=f"custom_{index}",
            kind="custom_java",
            config={},
        )
        for index in range(5)
    )
    policy = type("Policy", (), {"entity_shard_size": 24, "java_shard_size": 48})()
    shards = list(work_graph._module_shards(modules, policy=policy))
    assert len(shards) == 1
    assert len(shards[0][1]) == 5


def test_only_builtin_sidecar_integration_stays_on_cpu_stage() -> None:
    generic = ProductionModule(
        module_id="bridge",
        kind="integration",
        config={"integration_type": "third_party_bridge"},
    )
    sidecar = ProductionModule(
        module_id="sidecar",
        kind="integration",
        config={"integration_type": "mmm_local_ai_sidecar"},
    )
    assert work_graph._module_stage(generic) == "custom"
    assert work_graph._module_stage(sidecar) == "content"


def test_shared_gpu_allows_llm_read_sharing_but_blocks_image(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:max-efficiency",
        graph_hash="sha256:max-efficiency-graph",
        module_count=2,
        nodes=(
            _node("a-llm", "llm"),
            _node("b-llm", "llm"),
            _node("z-image", "image_gpu"),
        ),
    )
    ledger = DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    stages = ("generate:custom", "generate:assets")

    token = safety._SHARED_LOCAL_GPU_LANE.set(True)
    try:
        first = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        second = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        assert first is not None and second is not None
        assert [first["node_id"], second["node_id"]] == ["a-llm", "b-llm"]
        assert ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60) is None
        assert ledger.task("z-image")["state"] == "pending"

        ledger.succeed("a-llm", {"status": "PASS"})
        ledger.succeed("b-llm", {"status": "PASS"})
        image = ledger.claim_ready("mmm-orchestrator", stages=stages, lease_seconds=60)
        assert image is not None
        assert image["node_id"] == "z-image"
    finally:
        safety._SHARED_LOCAL_GPU_LANE.reset(token)


def test_parallel_custom_search_has_one_runtime_owner() -> None:
    generate = CustomModuleGenerator.generate
    assert getattr(generate, "_mmm_parallel_custom_search", False)
    assert getattr(generate, "_mmm_custom_verifier_search", False)
    assert getattr(generate, "_mmm_research_generation_search", False)
    assert not getattr(custom_search._width, "_mmm_context_single_candidate", False)
    assert custom_search._active_native_slots() >= 1
