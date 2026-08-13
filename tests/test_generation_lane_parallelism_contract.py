from __future__ import annotations

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.scheduler_parallel_safety_contract as safety
import minecraft_mod_ai.work_graph as work_graph_module


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


def test_unknown_integration_cannot_escape_single_llm_safe_lane() -> None:
    node = _module_node(
        "content",
        [
            {
                "module_id": "third_party_bridge",
                "kind": "integration",
                "config": {"integration_type": "unknown_custom_bridge"},
            }
        ],
    )
    assert node.resource_class == "commit"


def test_builtin_sidecar_integration_is_deterministic_cpu_work() -> None:
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
