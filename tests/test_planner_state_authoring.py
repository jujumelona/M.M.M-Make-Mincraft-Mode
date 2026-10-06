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
    state_variable_default_schema,
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
        {"count": "one"},
        {"target": "hull"},
        {"operator": "="},
        {"source": "number"},
        {"value": "100"},
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
    assert len(router.calls) == 5


def test_expression_page_preserves_canonical_typed_ir():
    guard = {
        "kind": "compare",
        "op": ">",
        "left": {"kind": "state_ref", "name": "hull"},
        "right": {"kind": "number", "value": "0"},
    }
    router = StateChoices([
        {"layout": "single"},
        {
            "source": "state",
            "state": "hull",
            "context": "context",
            "operator": ">",
        },
        {"source": "number"},
        {"value": "0"},
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
    assert len(router.calls) == 4
    assert all(
        "oneOf" not in json.dumps(call["response_schema"])
        for call in router.calls
    )


def test_empty_mutation_is_explicit_empty_list():
    router = StateChoices([{"count": "zero"}])

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
        {"count": "one"},
        {"target": "hull"},
        {"operator": "="},
        {"source": "state"},
        {"value": "hull"},
    ])

    author_state_field_page(
        router,
        "Update hull.",
        concern="updates",
        field="mutation",
        count=1,
        symbols=_symbols(),
    )

    target_schema = router.calls[1]["response_schema"]
    assert target_schema["properties"]["target"]["enum"] == ["hull"]
    assert all(
        "oneOf" not in json.dumps(call["response_schema"])
        for call in router.calls
    )


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
    default_schema = router.calls[0]["response_schema"]["properties"]["default"]
    assert default_schema == state_variable_default_schema("double")


def test_boolean_default_enum_is_enforced_at_model_boundary():
    item_schema = state_concern_schema("variables")
    router = StateChoices([{"default": "false"}])

    result = author_state_semantic_page(
        router,
        "Track whether the ship is docked.",
        concern="variables",
        fields=("default",),
        count=1,
        item_schema=item_schema,
        existing_rows=[{"name": "docked", "type": "boolean"}],
    )

    assert result == {"variables": [{"default": "false"}]}
    default_schema = router.calls[0]["response_schema"]["properties"]["default"]
    assert default_schema["enum"] == ["true", "false"]


def test_invalid_numeric_default_is_rejected_before_merge():
    item_schema = state_concern_schema("variables")
    router = StateChoices([{"default": "not-a-number"}])

    with pytest.raises(ValueError):
        author_state_semantic_page(
            router,
            "Track hull.",
            concern="variables",
            fields=("default",),
            count=1,
            item_schema=item_schema,
            existing_rows=[{"name": "hull", "type": "double"}],
        )


def test_malformed_boolean_default_is_rejected_by_transport_schema():
    item_schema = state_concern_schema("variables")
    router = StateChoices([{"default": "'}> false"}])

    with pytest.raises(ValueError):
        author_state_semantic_page(
            router,
            "Track whether the ship is docked.",
            concern="variables",
            fields=("default",),
            count=1,
            item_schema=item_schema,
            existing_rows=[{"name": "docked", "type": "boolean"}],
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

def test_boolean_conjunction_is_host_lowered_from_atomic_decisions():
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
        {"layout": "and"},
        {
            "source": "state",
            "state": "interstellar_trade_hub",
            "context": "context",
            "operator": "truthy",
        },
        {
            "source": "state",
            "state": "spacecraft_unlocked",
            "context": "context",
            "operator": "truthy",
        },
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
    assert len(router.calls) == 3
    assert all(
        "terms" not in call["response_schema"].get("properties", {})
        for call in router.calls
    )

def test_boolean_condition_is_host_lowered_from_atomic_choice():
    router = StateChoices([
        {"layout": "single"},
        {
            "source": "false",
            "state": "hull",
            "context": "context",
            "operator": "truthy",
        },
    ])

    result = author_state_field_page(
        router,
        "This condition is explicitly false.",
        concern="invariants",
        field="condition",
        count=1,
        symbols=_symbols(),
        existing_rows=[{"enforcement": "disabled"}],
    )

    assert result == {
        "invariants": [{"condition": {"kind": "literal", "value": False}}]
    }
    assert len(router.calls) == 2
    left_schema = router.calls[1]["response_schema"]
    assert left_schema["properties"]["source"]["enum"] == [
        "state",
        "context",
        "true",
        "false",
    ]
    assert "terms" not in left_schema["properties"]

def test_condition_model_schemas_never_request_guard_or_ast_wrappers():
    router = StateChoices([
        {"layout": "single"},
        {
            "source": "state",
            "state": "hull",
            "context": "context",
            "operator": "truthy",
        },
    ])

    author_state_field_page(
        router,
        "Guard on hull state.",
        concern="transitions",
        field="guard",
        count=1,
        symbols=_symbols(),
        existing_rows=[{"from_state": "a", "trigger": "tick", "to_state": "b"}],
    )

    forbidden = {"guard", "condition", "kind", "left", "right", "terms"}
    for call in router.calls:
        properties = set(call["response_schema"].get("properties", {}))
        assert not (properties & forbidden)

