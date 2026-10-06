from __future__ import annotations

from minecraft_mod_ai import design_record_runtime
from minecraft_mod_ai.content_design_contract import (
    relation_type_supported_for_content_pair,
)


def test_relation_capability_contract_reserves_contains_for_tag_sources() -> None:
    assert relation_type_supported_for_content_pair(
        "contains", "registry_tag", "item"
    )
    assert relation_type_supported_for_content_pair(
        "contains", "registry_tag", "block"
    )
    assert relation_type_supported_for_content_pair(
        "contains", "registry_tag", "entity"
    )
    assert not relation_type_supported_for_content_pair(
        "contains", "item", "registry_tag"
    )


def test_relation_capability_contract_matches_special_lowering_shapes() -> None:
    assert relation_type_supported_for_content_pair(
        "consumes", "crafting_recipe", "item"
    )
    assert relation_type_supported_for_content_pair(
        "produces", "smelting_recipe", "item"
    )
    assert relation_type_supported_for_content_pair(
        "key_A", "crafting_recipe", "item"
    )
    assert not relation_type_supported_for_content_pair(
        "key_A", "smelting_recipe", "item"
    )
    assert relation_type_supported_for_content_pair("drops", "block", "item")
    assert not relation_type_supported_for_content_pair("drops", "block", "block")
    assert relation_type_supported_for_content_pair("opens", "item", "gui")
    assert not relation_type_supported_for_content_pair("opens", "item", "block")


def _serial_map(router, jobs, run_one, **kwargs):
    del router, kwargs
    return [run_one(job) for job in jobs]


def _empty_pair_records(
    router,
    identifier,
    *,
    context,
    progress,
    checkpoint,
    **kwargs,
):
    del router, context, progress, checkpoint, kwargs
    assert identifier == "design/content_relation"
    return {"records": []}


def test_distinct_selector_cannot_repeat_model_preferred_identifier(monkeypatch) -> None:
    seen_enums = []

    def fake_generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, messages, kwargs
        props = response_schema["properties"]
        if "count" in props:
            return {"count": 2}
        if "id" in props:
            choices = list(props["id"]["enum"])
            seen_enums.append(choices)
            return {"id": choices[0]}
        raise AssertionError(response_schema)

    monkeypatch.setattr(
        design_record_runtime,
        "generate_fixed_template_value",
        fake_generate,
    )

    selected = design_record_runtime._select_distinct_relation_ids(
        object(),
        {"record_schema": {}},
        {},
        selector_id="regression:duplicate-input",
        selector_context={},
        candidate_ids=["coin", "ingot"],
        max_count=2,
        progress={},
        checkpoint=None,
        noun="ingredient",
    )

    assert selected == ["coin", "ingot"]
    assert seen_enums == [["coin", "ingot"], ["ingot"]]


def test_registry_tag_relations_have_host_owned_nonempty_membership(monkeypatch) -> None:
    seen_schemas = []

    def fake_generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, messages, kwargs
        seen_schemas.append(response_schema)
        props = response_schema["properties"]
        if "member_kind" in props:
            return {"member_kind": "item"}
        if "count" in props:
            return {"count": 1}
        if "id" in props:
            return {"id": "ship_blueprint"}
        raise AssertionError(response_schema)

    monkeypatch.setattr(design_record_runtime, "deterministic_model_map", _serial_map)
    monkeypatch.setattr(
        design_record_runtime,
        "run_bounded_record_template",
        _empty_pair_records,
    )
    monkeypatch.setattr(
        design_record_runtime,
        "generate_fixed_template_value",
        fake_generate,
    )

    records, _ = design_record_runtime._run_relations(
        object(),
        "design/content_relation",
        {
            "entity_ids": ["ship_blueprint", "tag_valid_ships"],
            "entities": [
                {
                    "entity_id": "ship_blueprint",
                    "kind": "item",
                    "implementation_obligations": [],
                },
                {
                    "entity_id": "tag_valid_ships",
                    "kind": "registry_tag",
                    "implementation_obligations": ["groups valid ship blueprints"],
                },
            ],
        },
        {},
        None,
    )

    assert records == [
        {
            "relation_type": "contains",
            "source_id": "tag_valid_ships",
            "target_id": "ship_blueprint",
        }
    ]
    assert all("uniqueItems" not in str(schema) for schema in seen_schemas)


def test_crafting_recipe_relations_have_exact_output_and_bounded_inputs(monkeypatch) -> None:
    seen_id_enums = []
    seen_schemas = []

    def fake_generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, messages, kwargs
        seen_schemas.append(response_schema)
        props = response_schema["properties"]
        if {"mode", "output_id"} <= set(props):
            return {
                "mode": "shaped",
                "output_id": "ship_hull",
            }
        if "count" in props:
            return {"count": 2}
        if "id" in props:
            choices = list(props["id"]["enum"])
            seen_id_enums.append(choices)
            if "steel_plate" in choices:
                return {"id": "steel_plate"}
            return {"id": "engine_core"}
        raise AssertionError(response_schema)

    monkeypatch.setattr(design_record_runtime, "deterministic_model_map", _serial_map)
    monkeypatch.setattr(
        design_record_runtime,
        "run_bounded_record_template",
        _empty_pair_records,
    )
    monkeypatch.setattr(
        design_record_runtime,
        "generate_fixed_template_value",
        fake_generate,
    )

    records, _ = design_record_runtime._run_relations(
        object(),
        "design/content_relation",
        {
            "entity_ids": [
                "ship_recipe",
                "ship_hull",
                "steel_plate",
                "engine_core",
            ],
            "entities": [
                {
                    "entity_id": "ship_recipe",
                    "kind": "crafting_recipe",
                    "implementation_obligations": ["craft the ship hull"],
                },
                {
                    "entity_id": "ship_hull",
                    "kind": "item",
                    "implementation_obligations": [],
                },
                {
                    "entity_id": "steel_plate",
                    "kind": "item",
                    "implementation_obligations": [],
                },
                {
                    "entity_id": "engine_core",
                    "kind": "item",
                    "implementation_obligations": [],
                },
            ],
        },
        {},
        None,
    )

    recipe_edges = [row for row in records if row["source_id"] == "ship_recipe"]
    assert recipe_edges == [
        {
            "relation_type": "produces",
            "source_id": "ship_recipe",
            "target_id": "ship_hull",
        },
        {
            "relation_type": "key_A",
            "source_id": "ship_recipe",
            "target_id": "steel_plate",
        },
        {
            "relation_type": "key_B",
            "source_id": "ship_recipe",
            "target_id": "engine_core",
        },
    ]
    assert sum(row["relation_type"] == "produces" for row in recipe_edges) == 1
    assert "steel_plate" in seen_id_enums[0]
    assert "steel_plate" not in seen_id_enums[1]
    assert all("uniqueItems" not in str(schema) for schema in seen_schemas)


def test_smelting_recipe_relations_have_exact_input_and_output(monkeypatch) -> None:
    def fake_generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, messages, kwargs
        props = response_schema["properties"]
        if {"output_id", "ingredient_id"} <= set(props):
            return {
                "output_id": "refined_alloy",
                "ingredient_id": "raw_alloy",
            }
        raise AssertionError(response_schema)

    monkeypatch.setattr(design_record_runtime, "deterministic_model_map", _serial_map)
    monkeypatch.setattr(
        design_record_runtime,
        "run_bounded_record_template",
        _empty_pair_records,
    )
    monkeypatch.setattr(
        design_record_runtime,
        "generate_fixed_template_value",
        fake_generate,
    )

    records, _ = design_record_runtime._run_relations(
        object(),
        "design/content_relation",
        {
            "entity_ids": ["smelt_alloy", "raw_alloy", "refined_alloy"],
            "entities": [
                {
                    "entity_id": "smelt_alloy",
                    "kind": "smelting_recipe",
                    "implementation_obligations": ["smelt raw alloy"],
                },
                {
                    "entity_id": "raw_alloy",
                    "kind": "item",
                    "implementation_obligations": [],
                },
                {
                    "entity_id": "refined_alloy",
                    "kind": "item",
                    "implementation_obligations": [],
                },
            ],
        },
        {},
        None,
    )

    recipe_edges = [row for row in records if row["source_id"] == "smelt_alloy"]
    assert recipe_edges == [
        {
            "relation_type": "consumes",
            "source_id": "smelt_alloy",
            "target_id": "raw_alloy",
        },
        {
            "relation_type": "produces",
            "source_id": "smelt_alloy",
            "target_id": "refined_alloy",
        },
    ]
