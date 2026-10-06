from __future__ import annotations

from minecraft_mod_ai.content_design_contract import (
    CONTENT_PROPERTY_VALUE_ENUMS,
    REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE,
)
from minecraft_mod_ai.design_generation_schema import context_bound_record_schema
from minecraft_mod_ai.task_template_catalog import load_record_template


def test_content_entity_authors_semantics_before_identity() -> None:
    template = load_record_template("design/content_entity")
    assert template["record_schema"]["required"] == ["role", "kind", "entity_id"]


def test_categorical_property_value_is_bound_before_model_generation() -> None:
    template = load_record_template("design/content_property")
    schema = context_bound_record_schema(
        "design/content_property",
        template["record_schema"],
        {
            "requested_property": "registry_kind",
            "allowed_properties": ["registry_kind"],
        },
    )
    assert schema["properties"]["property"]["enum"] == ["registry_kind"]
    assert schema["properties"]["value"]["enum"] == list(
        CONTENT_PROPERTY_VALUE_ENUMS["registry_kind"]
    )


def test_registry_tag_contract_has_one_shared_supported_vocabulary() -> None:
    assert tuple(REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE) == (
        "item",
        "block",
        "entity_type",
    )
    assert CONTENT_PROPERTY_VALUE_ENUMS["registry_kind"] == tuple(
        REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE
    )


def test_other_late_categorical_validations_are_schema_bound() -> None:
    template = load_record_template("design/content_property")
    for property_name in (
        "category",
        "archetype",
        "behavior",
        "recipe_kind",
        "cooking_type",
    ):
        schema = context_bound_record_schema(
            "design/content_property",
            template["record_schema"],
            {
                "requested_property": property_name,
                "allowed_properties": [property_name],
            },
        )
        assert schema["properties"]["value"]["enum"] == list(
            CONTENT_PROPERTY_VALUE_ENUMS[property_name]
        )
