from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.scheduler_parallel_safety_contract as safety
import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.scale_policy import ScalePolicy

safety.install(
    work_graph_module=work_graph_module,
    orchestrator_module=orchestrator_module,
)


def _module_node(stage: str, members: list[dict]):
    return work_graph_module._node(
        f"generate-{stage}-test",
        f"generate:{stage}",
        ("prepare-project",),
        {
            "kind": "module-shard",
            "generation_stage": stage,
            "members": members,
        },
    )


def test_deterministic_generation_domains_use_cpu_lane() -> None:
    content = _module_node(
        "content",
        [{"module_id": "ore", "kind": "block", "config": {}}],
    )
    system = _module_node(
        "system",
        [{"module_id": "quests", "kind": "quest", "config": {}}],
    )
    entity = _module_node(
        "entity",
        [{"module_id": "warden", "kind": "entity", "config": {}}],
    )

    assert content.resource_class == "cpu_io"
    assert system.resource_class == "cpu_io"
    assert entity.resource_class == "cpu_io"


def test_unknown_integration_routes_to_custom_llm_stage() -> None:
    module = ProductionModule(
        module_id="third_party_bridge",
        kind="integration",
        config={"integration_type": "unknown_custom_bridge"},
    )
    assert work_graph_module._module_stage(module) == "custom"


def test_builtin_sidecar_integration_is_deterministic_cpu_work() -> None:
    module = ProductionModule(
        module_id="local_ai",
        kind="integration",
        config={"integration_type": "mmm_local_ai_sidecar"},
    )
    assert work_graph_module._module_stage(module) == "content"
    node = _module_node(
        "content",
        [
            {
                "module_id": "local_ai",
                "kind": "integration",
                "config": {"integration_type": "mmm_local_ai_sidecar"},
            }
        ],
    )
    assert node.resource_class == "cpu_io"


def test_stage_write_locks_are_domain_local_not_global() -> None:
    content = _module_node(
        "content",
        [{"module_id": "ore", "kind": "block", "config": {}}],
    )
    system = _module_node(
        "system",
        [{"module_id": "quests", "kind": "quest", "config": {}}],
    )
    entity = _module_node(
        "entity",
        [{"module_id": "warden", "kind": "entity", "config": {}}],
    )

    content_lock = safety._stage_write_lock(content)
    system_lock = safety._stage_write_lock(system)
    entity_lock = safety._stage_write_lock(entity)
    assert content_lock is not None
    assert system_lock is not None
    assert entity_lock is not None
    assert len({id(content_lock), id(system_lock), id(entity_lock)}) == 3


def test_custom_modules_keep_dependency_aware_bounded_shards(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    modules = tuple(
        ProductionModule(
            module_id=f"custom_{index}",
            kind="custom_java",
            config={"summary": f"custom {index}"},
        )
        for index in range(100)
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph_module._module_shards(modules, policy=policy))
    assert [stage for stage, _ in shards] == ["custom", "custom", "custom"]
    assert [len(members) for _, members in shards] == [48, 48, 4]


def test_entities_use_small_pipeline_shards(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", raising=False)
    modules = tuple(
        ProductionModule(
            module_id=f"entity_{index}",
            kind="entity",
            config={},
        )
        for index in range(5)
    )
    shards = list(work_graph_module._module_shards(modules, policy=ScalePolicy()))
    assert [stage for stage, _ in shards] == ["entity", "entity", "entity"]
    assert [len(members) for _, members in shards] == [2, 2, 1]
