from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.planner_state_authoring import (
    author_state_field_page,
    author_state_semantic_page,
)
from minecraft_mod_ai.structured_state_runtime import (
    StateSymbolTable,
    compile_mutation_ir,
    compile_state_expr_ir,
    state_concern_schema,
)


class StateChoices:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(next(self.responses))


def _symbols():
    return StateSymbolTable([
        {
            "name": "hull",
            "owner": "ship",
            "type": "double",
            "unit": "points",
            "default": "100",
            "domain": "0..100",
        }
    ])


def test_mutation_page_preserves_canonical_typed_ir():
    mutation = [
        {
            "target": "hull",
            "operator": "=",
            "value": {"kind": "number", "value": "100"},
        }
    ]
    router = StateChoices([
        {
            "assignments": [
                {
                    "target": "hull",
                    "operator": "=",
                    "value_kind": "number",
                    "value_state": "hull",
                    "value_context": "",
                    "value_text": "100",
                }
            ]
        }
    ])

    result = author_state_field_page(
        router,
        "Repair the ship hull.",
        concern="updates",
        field="mutation",
        count=1,
        symbols=_symbols(),
        existing_rows=[{"trigger": "repair", "owner": "server"}],
    )

    assert result == {"updates": [{"mutation": mutation}]}
    assert 'setState("hull", Double.valueOf("100"));' == compile_mutation_ir(
        mutation,
        declared={"hull"},
    )
    assert len(router.calls) == 1


def test_expression_page_preserves_canonical_typed_ir():
    guard = {
        "kind": "compare",
        "op": ">",
        "left": {"kind": "state_ref", "name": "hull"},
        "right": {"kind": "number", "value": "0"},
    }
    router = StateChoices([
        {
            "join": "and",
            "terms": [
                {
                    "left_kind": "state",
                    "left_state": "hull",
                    "left_context": "",
                    "left_value": "",
                    "operator": ">",
                    "right_kind": "number",
                    "right_state": "hull",
                    "right_context": "",
                    "right_value": "0",
                }
            ],
        }
    ])

    result = author_state_field_page(
        router,
        "Launch only while hull is positive.",
        concern="transitions",
        field="guard",
        count=1,
        symbols=_symbols(),
        existing_rows=[{"from_state": "docked", "trigger": "launch", "to_state": "space"}],
    )

    assert result == {"transitions": [{"guard": guard}]}
    compiled = compile_state_expr_ir(guard, declared={"hull"})
    assert '$mmmRead("hull", context)' in compiled
    assert "$mmmCompare" in compiled
    assert len(router.calls) == 1
    assert "oneOf" not in json.dumps(router.calls[0]["response_schema"])


def test_empty_mutation_is_explicit_empty_list():
    router = StateChoices([{"assignments": []}])

    result = author_state_field_page(
        router,
        "No state mutation is needed on cleanup.",
        concern="cleanup",
        field="action",
        count=1,
        symbols=_symbols(),
        existing_rows=[{"event": "shutdown", "retained_state": "hull"}],
    )

    assert result == {"cleanup": [{"action": []}]}


def test_mutation_transport_constrains_targets_to_declared_symbols():
    router = StateChoices([
        {
            "assignments": [
                {
                    "target": "hull",
                    "operator": "=",
                    "value_kind": "state",
                    "value_state": "hull",
                    "value_context": "",
                    "value_text": "",
                }
            ]
        }
    ])

    author_state_field_page(
        router,
        "Update hull.",
        concern="updates",
        field="mutation",
        count=1,
        symbols=_symbols(),
    )

    schema = router.calls[0]["response_schema"]
    assignment = schema["properties"]["assignments"]["items"]
    assert assignment["properties"]["target"]["enum"] == ["hull"]
    assert "oneOf" not in json.dumps(schema)


def test_variable_default_schema_is_narrowed_from_fixed_type():
    item_schema = state_concern_schema("variables")
    router = StateChoices([{"default": "0"}])

    result = author_state_semantic_page(
        router,
        "Track hull.",
        concern="variables",
        fields=("default",),
        count=1,
        item_schema=item_schema,
        existing_rows=[{"name": "hull", "type": "double"}],
    )

    assert result == {"variables": [{"default": "0"}]}


def test_invalid_numeric_default_is_rejected_before_merge():
    item_schema = state_concern_schema("variables")
    router = StateChoices([{"default": "not-a-number"}])

    with pytest.raises(ValueError, match="STATE_SEMANTIC_FIELD_INVALID"):
        author_state_semantic_page(
            router,
            "Track hull.",
            concern="variables",
            fields=("default",),
            count=1,
            item_schema=item_schema,
            existing_rows=[{"name": "hull", "type": "double"}],
        )

def test_duplicate_variable_names_are_host_deduplicated_without_retry():
    item_schema = state_concern_schema("variables")
    router = StateChoices([
        {"name": "state_model"},
        {"name": "state_model"},
        {"name": "state_model"},
    ])

    result = author_state_semantic_page(
        router,
        "Track three independent mutable values.",
        concern="variables",
        fields=("name",),
        count=3,
        item_schema=item_schema,
        existing_rows=[
            {"type": "double", "owner": "ship", "unit": "points", "domain": "hull integrity"},
            {"type": "double", "owner": "ship", "unit": "points", "domain": "shield strength"},
            {"type": "double", "owner": "ship", "unit": "units", "domain": "fuel reserve"},
        ],
    )

    assert result == {
        "variables": [
            {"name": "hull_integrity"},
            {"name": "shield_strength"},
            {"name": "fuel_reserve"},
        ]
    }
    assert len(router.calls) == 3

def test_boolean_conjunction_is_host_lowered_from_flat_terms():
    symbols = StateSymbolTable([
        {
            "name": "interstellar_trade_hub",
            "owner": "server",
            "type": "boolean",
            "unit": "flag",
            "default": "false",
            "domain": "trade hub availability",
        },
        {
            "name": "spacecraft_unlocked",
            "owner": "server",
            "type": "boolean",
            "unit": "flag",
            "default": "false",
            "domain": "spacecraft progression unlock",
        },
    ])
    router = StateChoices([
        {
            "join": "and",
            "terms": [
                {
                    "left_kind": "state",
                    "left_state": "interstellar_trade_hub",
                    "left_context": "",
                    "left_value": "",
                    "operator": "truthy",
                    "right_kind": "none",
                    "right_state": "interstellar_trade_hub",
                    "right_context": "",
                    "right_value": "",
                },
                {
                    "left_kind": "state",
                    "left_state": "spacecraft_unlocked",
                    "left_context": "",
                    "left_value": "",
                    "operator": "truthy",
                    "right_kind": "none",
                    "right_state": "spacecraft_unlocked",
                    "right_context": "",
                    "right_value": "",
                },
            ],
        }
    ])

    result = author_state_field_page(
        router,
        "Trading requires the hub and spacecraft unlock.",
        concern="invariants",
        field="condition",
        count=1,
        symbols=symbols,
        existing_rows=[{"enforcement": "deny trading until both flags are active"}],
    )

    assert result == {
        "invariants": [
            {
                "condition": {
                    "kind": "and",
                    "terms": [
                        {"kind": "state_ref", "name": "interstellar_trade_hub"},
                        {"kind": "state_ref", "name": "spacecraft_unlocked"},
                    ],
                }
            }
        ]
    }
    assert "oneOf" not in json.dumps(router.calls[0]["response_schema"])

