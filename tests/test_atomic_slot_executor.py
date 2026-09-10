from __future__ import annotations

import pytest

from minecraft_mod_ai.atomic_slot_executor import (
    SlotDefinition,
    SlotFillError,
    fill_one_slot,
)
from minecraft_mod_ai.model_output_atomicity_contract import (
    MAX_MODEL_ARRAY_ITEMS,
    MAX_MODEL_FIELDS,
    MAX_MODEL_STRING_CHARS,
    MAX_SCHEMA_DEPTH,
    assert_strict_atomicity_bounds,
)


def test_slot_definition_rejects_broad_schema():
    # Attempting to declare more than 3 fields in a slot fails
    broad_schema = {
        "type": "object",
        "properties": {
            "f1": {"type": "string"},
            "f2": {"type": "string"},
            "f3": {"type": "string"},
            "f4": {"type": "string"},
        },
    }
    slot = SlotDefinition(slot_id="test_slot", schema=broad_schema)
    with pytest.raises(SlotFillError, match="SLOT_ATOMICITY_VIOLATION"):
        slot.validate_schema()


def test_fill_one_slot_deterministic_default_without_router():
    slot = SlotDefinition(
        slot_id="raw_lunite.max_stack",
        schema={"type": "integer", "minimum": 1, "maximum": 64},
        default=16,
    )
    result = fill_one_slot(None, slot, {})
    assert result == 16


def test_fill_one_slot_with_mock_router():
    class MockRouter:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def generate_text(self, *args, **kwargs):
            self.calls += 1
            import json
            return json.dumps({"max_stack": self.value})

    router = MockRouter(32)
    slot = {
        "id": "max_stack",
        "schema": {
            "type": "object",
            "properties": {"max_stack": {"type": "integer"}},
            "required": ["max_stack"],
            "additionalProperties": False,
        },
    }
    val = fill_one_slot(router, slot, {"item": "ruby_gem"})
    assert val == 32


def test_strict_atomicity_bounds_enforcement():
    # Valid atomic schema
    assert_strict_atomicity_bounds(
        {
            "type": "object",
            "properties": {
                "val": {"type": "string", "maxLength": 100},
            },
        }
    )

    # Rejects too many fields (> 3)
    with pytest.raises(Exception, match="MODEL_ATOMICITY_FIELDS_EXCEEDED"):
        assert_strict_atomicity_bounds(
            {
                "type": "object",
                "properties": {f"f_{i}": {"type": "string"} for i in range(MAX_MODEL_FIELDS + 1)},
            }
        )

    # Rejects schema depth > 2
    with pytest.raises(Exception, match="MODEL_ATOMICITY_DEPTH_EXCEEDED"):
        assert_strict_atomicity_bounds(
            {
                "type": "object",
                "properties": {
                    "nested_1": {
                        "type": "object",
                        "properties": {
                            "nested_2": {
                                "type": "object",
                                "properties": {"val": {"type": "string"}},
                            }
                        },
                    }
                },
            }
        )

    # Rejects string with maxLength > 256
    with pytest.raises(Exception, match="MODEL_ATOMICITY_STRING_EXCEEDED"):
        assert_strict_atomicity_bounds(
            {
                "type": "string",
                "maxLength": MAX_MODEL_STRING_CHARS + 1,
            }
        )

    # Rejects array with maxItems > 4
    with pytest.raises(Exception, match="MODEL_ATOMICITY_ARRAY_EXCEEDED"):
        assert_strict_atomicity_bounds(
            {
                "type": "array",
                "maxItems": MAX_MODEL_ARRAY_ITEMS + 1,
            }
        )
