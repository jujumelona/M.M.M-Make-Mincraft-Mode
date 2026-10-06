from __future__ import annotations

from minecraft_mod_ai.authored_content_contract import (
    CONTENT_CONCERN_MINIMUM_ENTITY_COUNT,
)
from minecraft_mod_ai.content_design_contract import (
    CONTENT_CONCERN_KINDS,
    PRIMARY_CONTENT_KINDS,
    RESOURCE_DEFINITION_KINDS,
    resource_definition_kind_for_data_resource,
)
from minecraft_mod_ai.content_design_graph import (
    _materialize_authored_resource_definitions,
    _native_resource_module_config,
)
from minecraft_mod_ai.extended_content_generator import validate_extended_module_contract
from minecraft_mod_ai.implementation_fact import FactType
from minecraft_mod_ai.complete_spec import ProductionModule


def _owned(requirement_id: str) -> dict[str, object]:
    return {
        "requirement_id": requirement_id,
        "requirement": f"Implement {requirement_id}",
    }


def _primary(kind: str, entity_id: str) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "kind": kind,
        "role": entity_id,
        "requirement_refs": ["gameplay"],
        "source_clauses": ["gameplay"],
        "implementation_obligations": [entity_id],
    }


def test_engineering_resource_concerns_do_not_force_content_identity() -> None:
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["registries"] == 0
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["data_resources"] == 0
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["assets"] == 1
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["interactions"] == 1
    assert CONTENT_CONCERN_MINIMUM_ENTITY_COUNT["displayed_state"] == 1


def test_primary_discovery_vocabulary_excludes_resource_definitions() -> None:
    assert set(PRIMARY_CONTENT_KINDS).isdisjoint(RESOURCE_DEFINITION_KINDS)
    for kinds in CONTENT_CONCERN_KINDS.values():
        assert set(kinds).isdisjoint(RESOURCE_DEFINITION_KINDS)


def test_data_resource_kind_mapping_is_explicit_and_fail_closed() -> None:
    assert resource_definition_kind_for_data_resource("tag") == "registry_tag"
    assert resource_definition_kind_for_data_resource("item tag") == "registry_tag"
    assert resource_definition_kind_for_data_resource("crafting recipe") == "crafting_recipe"
    assert resource_definition_kind_for_data_resource("smelting") == "smelting_recipe"
    assert resource_definition_kind_for_data_resource("ship blueprint db") is None
    assert resource_definition_kind_for_data_resource("code registry") is None


def test_resource_definition_without_concrete_target_is_not_created() -> None:
    entities: dict[str, dict] = {}
    created = _materialize_authored_resource_definitions(
        entities,
        {
            "data_resources": [{
                "kind": "tag",
                "purpose": "ship blueprint tag",
                "owner": "space_mode",
            }]
        },
        [_owned("gameplay")],
    )

    assert created == ()
    assert entities == {}


def test_registry_constraint_never_invents_registry_tag() -> None:
    entities = {
        "ship_blueprint": _primary("item", "ship_blueprint"),
    }
    created = _materialize_authored_resource_definitions(
        entities,
        {
            "registries": [{
                "purpose": "ship blueprint db",
                "identifier": "ship_blueprint_registry",
                "binding_requirement": "bind without inventing a tag",
            }]
        },
        [_owned("gameplay")],
    )

    assert created == ()
    assert set(entities) == {"ship_blueprint"}


def test_explicit_tag_is_derived_only_after_concrete_member_exists() -> None:
    entities = {
        "moon_ore": _primary("item", "moon_ore"),
    }
    created = _materialize_authored_resource_definitions(
        entities,
        {
            "data_resources": [{
                "kind": "tag",
                "purpose": "ore resource metadata",
                "owner": "space_mode",
            }]
        },
        [_owned("gameplay")],
    )

    assert len(created) == 1
    node = entities[created[0]]
    assert node["kind"] == "registry_tag"
    assert node["host_derived_resource_definition"] is True
    assert node["requirement_refs"] == ["gameplay"]


def test_recipe_definitions_are_derived_only_with_item_targets() -> None:
    entities = {
        "raw_alloy": _primary("item", "raw_alloy"),
    }
    created = _materialize_authored_resource_definitions(
        entities,
        {
            "data_resources": [
                {
                    "kind": "crafting recipe",
                    "purpose": "assemble alloy",
                    "owner": "alloy",
                },
                {
                    "kind": "smelting recipe",
                    "purpose": "refine alloy",
                    "owner": "alloy",
                },
            ]
        },
        [_owned("gameplay")],
    )

    assert {entities[eid]["kind"] for eid in created} == {
        "crafting_recipe",
        "smelting_recipe",
    }


def test_resource_nodes_never_create_synthetic_requirement_ids() -> None:
    entities = {
        "raw_alloy": _primary("item", "raw_alloy"),
    }
    created = _materialize_authored_resource_definitions(
        entities,
        {
            "data_resources": [{
                "kind": "tag",
                "purpose": "alloy grouping",
                "owner": "alloy",
            }]
        },
        [_owned("content_assets_a"), _owned("content_interactions_b")],
    )

    node = entities[created[0]]
    assert node["requirement_refs"] == [
        "content_assets_a",
        "content_interactions_b",
    ]
    assert all(not ref.startswith("resource_") for ref in node["requirement_refs"])

def test_registry_tag_native_fallback_uses_normalized_resource_ids() -> None:
    node = _primary("registry_tag", "moon_ore_tag")
    config = _native_resource_module_config(
        FactType.REGISTRY_TAG,
        {
            "registry_kind": "item",
            "members": ["space:moon_ore"],
            "resource_references": ["space:moon_ore"],
        },
        node,
    )

    assert config["registry"] == "items"
    assert config["values"] == ["space:moon_ore"]
    assert config["replace"] is False

    validate_extended_module_contract(
        ProductionModule("moon_ore_tag", "tag", config)
    )

