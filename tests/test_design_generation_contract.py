from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, ValidationError

from minecraft_mod_ai.content_design_contract import (
    CONTENT_KIND_TO_FACT_TYPE,
    CONTENT_KINDS,
)
from minecraft_mod_ai.single_record_template import (
    _context_bound_record_schema,
    run_single_record_template,
)
from minecraft_mod_ai.task_template_catalog import load_record_template


def test_content_entity_kind_is_generation_time_closed_vocabulary():
    template = load_record_template("design/content_entity")
    schema = _context_bound_record_schema(
        "design/content_entity",
        template["record_schema"],
        {},
    )
    assert schema["properties"]["kind"]["enum"] == list(CONTENT_KINDS)
    assert "economy_resource_management" not in schema["properties"]["kind"]["enum"]


def test_content_capability_is_bound_to_exact_fact_type_before_model_call():
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
    assert captured[0]["properties"]["fact_type"]["enum"] == ["ITEM_EXISTS"]
    assert "UNSUPPORTED" not in captured[0]["properties"]["fact_type"]["enum"]


def test_content_capability_template_has_no_host_invalid_sentinel():
    template = load_record_template("design/content_capability")
    values = template["record_schema"]["properties"]["fact_type"]["enum"]
    assert "UNSUPPORTED" not in values
    assert set(values) == {
        fact_type.value for fact_type in CONTENT_KIND_TO_FACT_TYPE.values()
    }


def test_broad_system_kind_is_rejected_by_same_schema_sent_to_model():
    template = load_record_template("design/content_entity")
    schema = _context_bound_record_schema(
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
    schema = _context_bound_record_schema(
        "design/decision",
        decision["record_schema"],
        {"allowed_slots": ["economy_source", "economy_sink"]},
    )
    assert schema["properties"]["slot_id"]["enum"] == [
        "economy_source",
        "economy_sink",
    ]

    prop = load_record_template("design/content_property")
    prop_schema = _context_bound_record_schema(
        "design/content_property",
        prop["record_schema"],
        {
            "allowed_properties": ["display_name", "main_color"],
            "requested_property": "main_color",
        },
    )
    assert prop_schema["properties"]["property"]["enum"] == ["main_color"]
