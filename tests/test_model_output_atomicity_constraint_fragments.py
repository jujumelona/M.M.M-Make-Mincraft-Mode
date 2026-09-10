from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


def test_closed_object_allows_same_instance_anyof_property_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "records": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 4},
            "inapplicable": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 4},
        },
        "required": [],
        "anyOf": [
            {
                "required": ["records"],
                "properties": {"records": {"minItems": 1}},
            },
            {
                "required": ["inapplicable"],
                "properties": {"inapplicable": {"minItems": 1}},
            },
        ],
        "additionalProperties": False,
    }

    assert_atomic_model_schema(schema, surface="scoped-anyof-regression")


def test_unscoped_properties_schema_still_must_be_closed() -> None:
    schema = {
        "properties": {"value": {"type": "string"}},
    }

    with pytest.raises(ModelConfigurationError, match="MODEL_JSON_TEMPLATE_REQUIRED"):
        assert_atomic_model_schema(schema, surface="unscoped-open-object")


def test_explicit_object_inside_anyof_still_must_be_closed() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 256}},
        "anyOf": [
            {
                "type": "object",
                "properties": {"value": {"type": "string", "maxLength": 256}},
                "required": ["value"],
            }
        ],
        "additionalProperties": False,
    }

    with pytest.raises(ModelConfigurationError, match="MODEL_JSON_TEMPLATE_REQUIRED"):
        assert_atomic_model_schema(schema, surface="nested-open-object")


def test_all_planning_contracts_accept_scoped_worksheet_signal_fragments() -> None:
    from minecraft_mod_ai.planning_contract_ssot import assert_all_planning_contracts_valid

    assert_all_planning_contracts_valid()
