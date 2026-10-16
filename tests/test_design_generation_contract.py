from __future__ import annotations

import re

import pytest
from jsonschema import Draft202012Validator, ValidationError

from minecraft_mod_ai.content_design_contract import (
    CONTENT_KIND_TO_FACT_TYPE,
    PRIMARY_CONTENT_KINDS,
)
from minecraft_mod_ai.design_generation_schema import context_bound_record_schema
from minecraft_mod_ai.model_output_atomicity_contract import (
    assert_atomic_model_schema,
    effective_model_transport_schema,
)
from minecraft_mod_ai.single_record_template import run_single_record_template
from minecraft_mod_ai.task_template_catalog import load_record_template


def test_content_entity_kind_is_generation_time_closed_vocabulary():
    template = load_record_template("design/content_entity")
    schema = context_bound_record_schema(
        "design/content_entity",
        template["record_schema"],
        {},
    )
    assert schema["properties"]["kind"]["enum"] == list(PRIMARY_CONTENT_KINDS)
    assert "economy_resource_management" not in schema["properties"]["kind"]["enum"]


def test_content_capability_is_bound_to_exact_fact_type_without_model_call():
    captured = []

    def generate(_router, _role, _messages, *, response_schema, **_kwargs):
        captured.append(response_schema)
        return {"fact_type": "ITEM_EXISTS"}

    result = run_single_record_template(
        None,
        "design/content_capability",
        context={"entity": {"kind": "item"}},
        generator=generate,
    )

    assert result == {"fact_type": "ITEM_EXISTS"}
    assert captured == []


def test_content_capability_template_has_no_host_invalid_sentinel():
    template = load_record_template("design/content_capability")
    values = template["record_schema"]["properties"]["fact_type"]["enum"]
    assert "UNSUPPORTED" not in values
    assert set(values) == {
        fact_type.value for fact_type in CONTENT_KIND_TO_FACT_TYPE.values()
    }


def test_broad_system_kind_is_rejected_by_same_schema_sent_to_model():
    template = load_record_template("design/content_entity")
    schema = context_bound_record_schema(
        "design/content_entity",
        template["record_schema"],
        {},
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(
            {
                "entity_id": "economy_resource_management",
                "kind": "economy_resource_management",
                "role": "manage the economy",
            }
        )


def test_host_closed_sets_become_actual_model_schema_enums():
    decision = load_record_template("design/decision")
    schema = context_bound_record_schema(
        "design/decision",
        decision["record_schema"],
        {"allowed_slots": ["economy_source", "economy_sink"]},
    )
    assert schema["properties"]["slot_id"]["enum"] == [
        "economy_source",
        "economy_sink",
    ]

    prop = load_record_template("design/content_property")
    prop_schema = context_bound_record_schema(
        "design/content_property",
        prop["record_schema"],
        {
            "allowed_properties": ["display_name", "main_color"],
            "requested_property": "main_color",
        },
    )
    assert prop_schema["properties"]["property"]["enum"] == ["main_color"]


def test_entity_prefix_excludes_owned_ids_in_both_host_and_native_schema():
    template = load_record_template("design/content_entity")
    accepted = [
        {"entity_id": "red_key", "kind": "item", "role": "Unlock the hangar."},
        {"entity_id": "key_" + "a" * 60, "kind": "item", "role": "Unlock the vault."},
    ]
    schema = context_bound_record_schema(
        "design/content_entity", template["record_schema"], {"accepted_records": accepted}
    )
    assert_atomic_model_schema(schema, surface="entity accepted prefix")
    native = effective_model_transport_schema(schema)
    pattern = native["properties"]["entity_id"]["pattern"]

    for entity_id in (record["entity_id"] for record in accepted):
        assert not re.fullmatch(pattern, entity_id)
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate({**accepted[0], "entity_id": entity_id})

    # Distinct concrete owners may share descriptive roles. Do not force a
    # paraphrase merely to create the second owner.
    for entity_id in ("blue_key", "red_keys", "aa", "a" * 64):
        record = {**accepted[0], "entity_id": entity_id}
        Draft202012Validator(schema).validate(record)
        assert re.fullmatch(pattern, entity_id)
    for entity_id in ("a", "1red_key", "Red_key", "red-key", "a" * 65):
        assert not re.fullmatch(pattern, entity_id)
    assert template["record_schema"]["properties"]["entity_id"]["pattern"] == (
        "^[a-z][a-z0-9_]+$"
    )


def test_existing_entity_can_be_reused_outside_current_prefix():
    template = load_record_template("design/content_entity")
    existing = {"entity_id": "red_key", "kind": "item", "role": "Unlock the hangar."}
    schema = context_bound_record_schema(
        "design/content_entity",
        template["record_schema"],
        {"existing_entities": [existing], "accepted_records": []},
    )
    Draft202012Validator(schema).validate(existing)
