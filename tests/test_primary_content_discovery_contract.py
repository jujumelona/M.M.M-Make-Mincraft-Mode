from __future__ import annotations

from minecraft_mod_ai.content_design_contract import (
    CONTENT_KINDS,
    PRIMARY_CONTENT_KINDS,
    RESOURCE_DEFINITION_KINDS,
)
from minecraft_mod_ai.design_generation_schema import context_bound_record_schema


def test_primary_content_vocabulary_is_strict_subset_of_all_content() -> None:
    assert set(PRIMARY_CONTENT_KINDS) < set(CONTENT_KINDS)
    assert set(PRIMARY_CONTENT_KINDS).isdisjoint(RESOURCE_DEFINITION_KINDS)
    assert set(PRIMARY_CONTENT_KINDS) | set(RESOURCE_DEFINITION_KINDS) == set(
        CONTENT_KINDS
    )


def test_content_entity_schema_defaults_to_primary_vocabulary() -> None:
    schema = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string"},
            "kind": {"type": "string"},
            "role": {"type": "string"},
        },
        "required": ["entity_id", "kind", "role"],
        "additionalProperties": False,
    }

    bound = context_bound_record_schema(
        "design/content_entity",
        schema,
        {},
    )

    assert bound["properties"]["kind"]["enum"] == list(PRIMARY_CONTENT_KINDS)
    assert not (
        set(bound["properties"]["kind"]["enum"])
        & set(RESOURCE_DEFINITION_KINDS)
    )


def test_explicit_primary_scope_cannot_smuggle_resource_definition_kind() -> None:
    schema = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string"},
            "kind": {"type": "string"},
            "role": {"type": "string"},
        },
        "required": ["entity_id", "kind", "role"],
        "additionalProperties": False,
    }

    bound = context_bound_record_schema(
        "design/content_entity",
        schema,
        {"allowed_content_kinds": list(PRIMARY_CONTENT_KINDS)},
    )

    assert set(bound["properties"]["kind"]["enum"]) == set(PRIMARY_CONTENT_KINDS)
