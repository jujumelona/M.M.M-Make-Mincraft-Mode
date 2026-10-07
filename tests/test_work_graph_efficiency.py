from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import work_graph
from minecraft_mod_ai.complete_spec import ProductionModule


def test_module_shards_use_dependency_ready_waves() -> None:
    independent_content = ProductionModule(module_id="a_content", kind="item")
    dependent_entity = ProductionModule(
        module_id="b_dependent_entity",
        kind="entity",
        depends_on=("a_content",),
    )
    independent_entity = ProductionModule(module_id="z_independent_entity", kind="entity")
    ordered = work_graph._topological_modules(
        (independent_content, dependent_entity, independent_entity)
    )
    assert [value.module_id for value in ordered] == [
        "a_content",
        "b_dependent_entity",
        "z_independent_entity",
    ]
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(ordered, policy=policy))
    shaped = [
        (stage, [module.module_id for module in members])
        for stage, members in shards
    ]
    assert shaped == [
        ("content", ["a_content"]),
        ("entity", ["z_independent_entity"]),
        ("entity", ["b_dependent_entity"]),
    ]


def _typed_host_module(index: int, *, depends_on: tuple[str, ...] = ()) -> ProductionModule:
    return ProductionModule(
        module_id=f"typed_host_{index:03d}",
        kind="typed_host",
        config={"typed_plan_ir": {}},
        depends_on=depends_on,
        required_gates=("target_compile",),
    )


def test_work_graph_exports_typed_host_checkpoint_contract() -> None:
    assert work_graph._module_shards._mmm_typed_host_task_checkpoints is True


def test_typed_host_tasks_are_one_durable_host_node_each() -> None:
    modules = tuple(_typed_host_module(index) for index in range(1, 24))
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)

    shards = list(work_graph._module_shards(modules, policy=policy))

    assert len(shards) == 23
    assert all(stage == "host" for stage, _members in shards)
    assert all(len(members) == 1 for _stage, members in shards)
    assert [
        members[0].module_id
        for _stage, members in shards
    ] == [f"typed_host_{index:03d}" for index in range(1, 24)]


def test_typed_host_dependency_keeps_distinct_durable_nodes() -> None:
    first = _typed_host_module(1)
    dependent = _typed_host_module(2, depends_on=(first.module_id,))
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)

    shards = list(work_graph._module_shards((first, dependent), policy=policy))

    assert len(shards) == 2
    assert [stage for stage, _members in shards] == ["host", "host"]
    assert [members[0].module_id for _stage, members in shards] == [
        first.module_id,
        dependent.module_id,
    ]


def test_long_serial_dependency_chain_is_compressed_into_bounded_shards() -> None:
    modules: list[ProductionModule] = []
    for index in range(120):
        module_id = f"chain_{index:03d}"
        depends_on = () if index == 0 else (f"chain_{index - 1:03d}",)
        modules.append(
            ProductionModule(
                module_id=module_id,
                kind="item",
                depends_on=depends_on,
            )
        )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(tuple(modules), policy=policy))
    assert [stage for stage, _ in shards] == ["content", "content", "content"]
    assert [len(members) for _, members in shards] == [48, 48, 24]
    assert sum(len(members) for _, members in shards) == 120


def test_system_modules_batch_by_shared_pack_writer() -> None:
    modules = (
        ProductionModule(module_id="class_a", kind="class"),
        ProductionModule(module_id="skill_a", kind="skill"),
        ProductionModule(module_id="class_b", kind="class"),
        ProductionModule(module_id="quest_a", kind="quest"),
        ProductionModule(module_id="quest_b", kind="quest"),
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)

    shaped = [
        (stage, [module.module_id for module in members])
        for stage, members in work_graph._module_shards(modules, policy=policy)
    ]

    assert shaped == [
        ("system", ["class_a", "skill_a", "class_b"]),
        ("system", ["quest_a", "quest_b"]),
    ]


def test_system_modules_with_different_pack_writers_never_coalesce() -> None:
    modules = (
        ProductionModule(module_id="economy_a", kind="economy"),
        ProductionModule(module_id="shop_a", kind="shop"),
        ProductionModule(module_id="gui_a", kind="gui"),
        ProductionModule(module_id="network_a", kind="networking"),
        ProductionModule(module_id="party_a", kind="party"),
        ProductionModule(module_id="guild_a", kind="guild"),
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)

    shaped = [
        (stage, [module.kind for module in members])
        for stage, members in work_graph._module_shards(modules, policy=policy)
    ]

    assert shaped == [
        ("system", ["economy", "shop"]),
        ("system", ["gui", "networking"]),
        ("system", ["party", "guild"]),
    ]
