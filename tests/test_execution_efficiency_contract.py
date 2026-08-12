from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import complete_planner, work_graph
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
    assert "no fixed host item count" in str(calls[0]["system_prompt"]).lower()


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

    # Heap topological order is intentionally a_content, b_dependent_entity,
    # z_independent_entity. Consecutive-stage sharding would make the independent
    # entity wait inside the dependent entity's shard.
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


def test_custom_llm_modules_are_one_durable_node_each() -> None:
    modules = (
        ProductionModule(module_id="custom_one", kind="custom_java"),
        ProductionModule(module_id="custom_two", kind="custom_java"),
    )
    policy = SimpleNamespace(entity_shard_size=24, java_shard_size=48)
    shards = list(work_graph._module_shards(modules, policy=policy))

    assert [stage for stage, _ in shards] == ["custom", "custom"]
    assert [[item.module_id for item in members] for _, members in shards] == [
        ["custom_one"],
        ["custom_two"],
    ]
