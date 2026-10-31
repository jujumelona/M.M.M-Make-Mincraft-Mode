"""Production regression for the 2026-10-25 callback state-value failure.

A verified, source-owned event must not use arbitrary Java calls as a
substitute for its mandatory typed canonical state writes.
"""
import json

import pytest

from minecraft_mod_ai.host_callback_body import author_callback_body
from minecraft_mod_ai.host_typed_state_values import (
    lower_typed_java_value, typed_value_schema,
)


class Router:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((data_name, parameters))
        return next(self.values)


def _frame():
    return {
        "kind": "method",
        "prefix": "private boolean apply(example.Target target) {\n",
        "suffix": "\n}",
        "host_behavior_contract": {
            "behavior_id": "unlock_world", "intent": "Unlock colonization",
            "state_writes": ["unlock", "world_label"],
        },
        "host_state_access": [{
            "owner": "example.AuthoredStateModel",
            "methods": [
                {"name": "writeState_unlock",
                 "signature": "public static void writeState_unlock(boolean value);"},
                {"name": "writeState_world_label",
                 "signature": "public static void writeState_world_label(String value);"},
            ],
        }],
    }


def _verified_source():
    return [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "event_id": "example.Command.EVENT",
            "parameters": ["example.Target"],
        },
        "callback_input_api": [],
    })}]


def test_verified_callback_never_asks_for_freeform_effect_calls():
    router = Router([{"value": True}, {"value": "colonized"}])
    source = author_callback_body(
        router, _verified_source(), data_name="colonization",
        frame=_frame(), effect_receipt=True,
    )
    assert "example.AuthoredStateModel.writeState_unlock(true);" in source
    assert 'example.AuthoredStateModel.writeState_world_label("colonized");' in source
    assert source.count("__mmmHostEffectApplied = true;") == 2
    assert "return __mmmHostEffectApplied;" in source
    assert len(router.calls) == 2
    assert all("_operation_" not in name for name, _ in router.calls)
    assert router.calls[0][1]["properties"]["value"]["type"] == "boolean"
    assert router.calls[1][1]["properties"]["value"]["type"] == "string"


def test_bad_expression_cannot_become_boolean_state_value():
    router = Router([
        {"value": "source.getPlayer().removeAllEffects();"},
        {"value": "source.getPlayer().removeAllEffects();"},
    ])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(
            router, _verified_source(), data_name="rejected",
            frame=_frame(), effect_receipt=True,
        )
    assert len(router.calls) == 2
    assert all("_operation_" not in name for name, _ in router.calls)


@pytest.mark.parametrize("row,java_type,expected", [
    ({"value": True}, "boolean", "true"),
    ({"value": False}, "boolean", "false"),
    ({"value": 27}, "int", "27"),
    ({"value": 27}, "long", "27L"),
    ({"value": 1.25}, "double", "1.25d"),
    ({"value": "hello\\\"; target.kill(); //"}, "String",
     '"hello\\\\\\\"; target.kill(); //"'),
])
def test_host_renders_typed_java_literals(row, java_type, expected):
    assert lower_typed_java_value(row, java_type) == expected


@pytest.mark.parametrize("row,java_type", [
    ({"value": "false;"}, "boolean"),
    ({"value": True}, "long"),
    ({"value": 2 ** 63}, "long"),
    ({"value": float("inf")}, "double"),
    ({"expression": "target.mutate();"}, "boolean"),
])
def test_host_rejects_invalid_typed_values(row, java_type):
    with pytest.raises(ValueError):
        lower_typed_java_value(row, java_type)


def test_typed_schema_uses_scalar_not_source_expression():
    schema = typed_value_schema("boolean")
    assert schema["properties"] == {"value": {"type": "boolean"}}
    assert "expression" not in schema["properties"]


def test_verified_numeric_state_addition_uses_checked_read_modify_write():
    frame = _frame()
    frame["host_behavior_contract"]["state_writes"] = ["credits"]
    frame["host_state_access"][0]["methods"] = [{
        "name": "writeState_credits",
        "signature": "public static void writeState_credits(long value);",
    }]
    router = Router([{"value": 5, "operator": "add"}])
    source = author_callback_body(
        router, _verified_source(), data_name="balance_delta",
        frame=frame, effect_receipt=True,
    )
    assert ("example.AuthoredStateModel.writeState_credits("
            "java.lang.Math.addExact(example.AuthoredStateModel.readState_credits(), 5L));"
            in source)
    assert "writeState_credits(5L);" not in source
    assert router.calls[0][1]["properties"]["operator"]["enum"] == [
        "set", "add", "subtract",
    ]


def test_checked_typed_state_transition_blocks_noop_delta():
    frame = _frame()
    frame["host_behavior_contract"]["state_writes"] = ["credits"]
    frame["host_state_access"][0]["methods"] = [{
        "name": "writeState_credits",
        "signature": "public static void writeState_credits(long value);",
    }]
    router = Router([{"value": 0, "operator": "add"},
                     {"value": 0, "operator": "add"}])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(
            router, _verified_source(), data_name="zero_delta",
            frame=frame, effect_receipt=True,
        )
