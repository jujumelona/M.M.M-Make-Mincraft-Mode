from __future__ import annotations

import inspect
from types import SimpleNamespace

from minecraft_mod_ai import complete_planner, execution_efficiency_contract, work_graph
from minecraft_mod_ai.complete_spec import ProductionModule


def _raw_module(deliverable: str) -> dict[str, object]:
    return {
        "module_id": deliverable,
        "kind": "item",
        "config": {},
        "depends_on": [],
        "required_gates": [],
        "implements_deliverables": [deliverable],
    }


def test_production_batch_has_no_fixed_four_deliverable_page_width(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    deliverables = tuple(f"d{index}" for index in range(1, 7))

    def page_generator(
        router,
        *,
        system_prompt,
        request,
        media_paths,
        expected_contracts,
        stage,
    ):
        calls.append(
            {
                "request": request,
                "system_prompt": system_prompt,
                "media_paths": media_paths,
                "stage": stage,
            }
        )
        return {
            "modules": [_raw_module(value) for value in deliverables],
            "assets": [],
            "audio": [],
            "acceptance_tests": ["all deliverables are present"],
            "completed_deliverables": list(deliverables),
            "complete": True,
            "next_cursor": "",
        }

    monkeypatch.setattr(
        complete_planner,
        "_generate_json_page_with_repair",
        page_generator,
    )

    planner = object.__new__(complete_planner.CompleteGameDesignPlanner)
    planner.router = object()
    batch = complete_planner._ProductionBatch(
        batch_id="adaptive_page",
        scope="implement six independent completion units",
        depends_on_batches=(),
        deliverables=deliverables,
        exports=(),
    )
    parts = complete_planner._ProductionParts([], [], [], [])

    planner._expand_one_production_batch(
        batch=batch,
        parts=parts,
        module_catalog=complete_planner._ModuleCatalog(),
        asset_catalog=complete_planner._ModuleCatalog(),
        audio_catalog=complete_planner._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={},
        planning_receipt={},
        media_paths=(),
    )

    assert len(calls) == 1
    request = calls[0]["request"]
    assert isinstance(request, dict)
    assert request["current_target_deliverables"] == list(deliverables)
    assert request["remaining_deliverables"] == list(deliverables)
    assert len(parts.modules) == len(deliverables)
    prompt = str(calls[0]["system_prompt"]).lower()
    assert "no fixed deliverable count" in prompt
    assert "no fixed page count" in prompt


def test_module_shards_use_dependency_ready_waves() -> None:
    independent_content = ProductionModule(
        module_id="a_content",
        kind="item",
    )
    dependent_entity = ProductionModule(
        module_id="b_dependent_entity",
        kind="entity",
        depends_on=("a_content",),
    )
    independent_entity = ProductionModule(
        module_id="z_independent_entity",
        kind="entity",
    )

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


def test_custom_llm_modules_are_bounded_without_one_node_per_module(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    modules = tuple(
        ProductionModule(module_id=f"custom_{index:03d}", kind="custom_java")
        for index in range(100)
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))

    assert [stage for stage, _ in shards] == ["custom", "custom", "custom"]
    assert [len(members) for _, members in shards] == [48, 48, 4]
    assert sum(len(members) for _, members in shards) == 100


def test_small_custom_wave_uses_all_selected_llm_slots_without_row_explosion(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    modules = tuple(
        ProductionModule(module_id=f"custom_{index:02d}", kind="custom_java")
        for index in range(10)
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))

    assert [stage for stage, _ in shards] == ["custom", "custom", "custom"]
    assert [len(members) for _, members in shards] == [4, 4, 2]
    assert sum(len(members) for _, members in shards) == 10


def test_large_custom_wave_keeps_java_shard_ceiling(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    modules = tuple(
        ProductionModule(module_id=f"custom_{index:03d}", kind="custom_java")
        for index in range(200)
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))

    sizes = [len(members) for stage, members in shards if stage == "custom"]
    assert sizes == [48, 48, 48, 48, 8]
    assert max(sizes) <= 48
    assert sum(sizes) == 200


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


def test_dependency_sharding_uses_indexed_lookup_not_all_group_reverse_scan() -> None:
    source = inspect.getsource(execution_efficiency_contract._dependency_wave_shards)
    assert "open_by_key" in source
    assert "for index in range(len(groups) - 1" not in source
