from __future__ import annotations

import pytest

from minecraft_mod_ai.atomic_slot_executor import (
    MAX_SLOT_CONTEXT_CHARS,
    SlotDefinition,
    SlotFillError,
    fill_one_slot,
)


def test_slot_rejects_broad_schema():
    slot = SlotDefinition(
        slot_id="test_slot",
        schema={
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
                "d": {"type": "string"},
            },
            "additionalProperties": False,
        },
    )
    with pytest.raises(SlotFillError, match="SLOT_ATOMICITY_VIOLATION"):
        slot.validate_schema()


def test_slot_rejects_magic_default():
    slot = SlotDefinition(
        slot_id="raw_lunite.max_stack",
        schema={"type": "integer", "minimum": 1, "maximum": 64},
        default=16,
    )
    with pytest.raises(SlotFillError, match="SLOT_DEFAULT_FORBIDDEN"):
        slot.validate_schema()


def test_missing_router_does_not_fabricate_value():
    slot = SlotDefinition(
        slot_id="raw_lunite.max_stack",
        schema={"type": "integer", "minimum": 1, "maximum": 64},
    )
    with pytest.raises(SlotFillError, match="SLOT_NO_ROUTER"):
        fill_one_slot(None, slot, {"evidence": "explicitly unresolved"})


def test_slot_context_must_be_small_evidence_slice():
    slot = SlotDefinition(
        slot_id="value",
        schema={"type": "integer"},
    )

    class Router:
        def generate_text(self, *args, **kwargs):
            return "1"

    with pytest.raises(SlotFillError, match="SLOT_CONTEXT_TOO_LARGE"):
        fill_one_slot(
            Router(),
            slot,
            {"entire_project": "x" * (MAX_SLOT_CONTEXT_CHARS + 1)},
        )


def test_mock_router_can_fill_one_bounded_object_field():
    class MockRouter:
        def generate_text(self, *args, **kwargs):
            return '{"max_stack":32}'

    slot = SlotDefinition(
        slot_id="max_stack",
        schema={
            "type": "object",
            "properties": {"max_stack": {"type": "integer", "minimum": 1, "maximum": 64}},
            "required": ["max_stack"],
            "additionalProperties": False,
        },
    )
    assert fill_one_slot(MockRouter(), slot, {"evidence": "max stack is 32"}) == 32
