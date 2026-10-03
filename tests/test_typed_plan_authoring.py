from copy import deepcopy

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir


class NativePlanner:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
        assert role == "planner"
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == {"value"}
        self.calls.append((tool_name, messages, parameters))
        return {"value": next(self.answers)}

    def generate_text(self, *args, **kwargs):
        pytest.fail("typed operations must use the native planner tool transport")


def test_native_recursive_operation_builder_preserves_nested_arithmetic():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    router = NativePlanner([
        {"op": "let", "name": "credits", "type": "int"},
        {"op": "binary", "operator": "+"},
        {"op": "literal", "type": "int", "value": 7},
        {"op": "literal", "type": "int", "value": 3},
        {"op": "assert", "message": "credits must total ten"},
        {"op": "binary", "operator": "=="},
        {"op": "ref", "name": "credits"},
        {"op": "literal", "type": "int", "value": 10},
        {"op": "done"},
    ])
    author = TypedOperationAuthor(router, "Credit test", {}, {}, max_calls=40)
    body = author.body("requirement_0001")
    assert body[0] == {
        "op": "let", "name": "credits", "type": "int",
        "value": {"op": "binary", "operator": "+",
                  "left": {"op": "literal", "type": "int", "value": 7},
                  "right": {"op": "literal", "type": "int", "value": 3}},
    }
    assert body[1]["condition"]["left"] == {"op": "ref", "name": "credits"}
    assert len(router.calls) == 9


def test_host_resolves_fully_determined_schema_without_model_call():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    router = NativePlanner([])
    author = TypedOperationAuthor(router, "fixed schema", {}, {}, max_calls=1)
    value = author._ask(
        "deterministic_fixture",
        {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["host"]},
                "count": {"type": "integer", "const": 2},
                "tags": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string", "const": "fixed"},
                },
            },
            "required": ["mode", "count", "tags"],
            "additionalProperties": False,
        },
        scope="fixture",
    )

    assert value == {
        "mode": "host",
        "count": 2,
        "tags": ["fixed", "fixed"],
    }
    assert author.call_count == 0
    assert router.calls == []


def test_native_recursive_authoring_stops_at_hard_call_bound():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    author = TypedOperationAuthor(
        NativePlanner([
            {"op": "unary", "operator": "!"},
            {"op": "unary", "operator": "!"},
            {"op": "unary", "operator": "!"},
        ]),
        "test",
        {},
        {},
        max_calls=3,
    )
    with pytest.raises(ValueError, match="TYPED_PLAN_AUTHORING_LIMIT"):
        author.expression("condition")


def test_legacy_saved_plan_roundtrip_remains_readable():
    legacy = AuthoredPlan.from_dict({
        "schema_version": "mmm/authored-plan-v1", "requested_prompt": "test", "text": "test",
    })
    assert legacy.typed_plan_ir == {}
    restored = AuthoredPlan.from_dict(deepcopy(legacy.to_dict()))
    assert restored.calculate_hash() == legacy.calculate_hash()

def test_event_handler_signature_is_host_owned_during_authoring():
    specification = {
        name: []
        for name in DETAIL_RECORDS["integration"]
    }
    specification["entry_points"] = [
        {
            "boundary": "server",
            "trigger": "server startup",
            "owner": "server",
        }
    ]
    specification["inapplicable_concerns"] = []
    structured = {
        "integration": {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }
    router = NativePlanner([])

    plan = author_typed_plan_ir(
        router,
        "server event design",
        structured,
        {},
        max_calls=16,
    )

    assert plan["event_bindings"] == [
        {
            "event": "server_started",
            "function": "entryPoint1_server_started",
            "entry_point_index": 0,
            "config": {},
        }
    ]
    handler = plan["functions"][0]
    assert handler["id"] == "entryPoint1_server_started"
    assert handler["parameters"] == [
        {"name": "server", "type": "object"}
    ]
    assert handler["return_type"] == "void"
    assert handler["body"] == []
    assert router.calls == []
    requested_fields = [
        call[1][1]["content"]
        for call in router.calls
    ]
    assert all("parameter_name" not in payload for payload in requested_fields)
    assert all("return_type" not in payload for payload in requested_fields)

