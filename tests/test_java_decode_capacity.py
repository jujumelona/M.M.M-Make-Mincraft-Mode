"""Structured Java-action paging and final-wire capacity contracts."""

from minecraft_mod_ai.first_pass_java_declarations import metadata_schema
from minecraft_mod_ai.first_pass_java_source import _one_structured_action, _PLAN_SCHEMA
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


class RecordingRouter:
    def __init__(self):
        self.calls = []

    def generate_structured_data(self, role, messages, **kwargs):
        self.calls.append(kwargs)
        props = kwargs["parameters"]["properties"]
        if "count" in props:
            return {"count": 0}
        return {
            key: (
                [] if value.get("type") == "array"
                else "" if value.get("type") == "string"
                else {}
            )
            for key, value in props.items()
        }


def test_member_metadata_is_bounded_and_does_not_reauthor_host_owned_identity():
    # Implementation bodies are authored separately from the selected signature.
    schema = metadata_schema("method", "execute")
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "name" not in schema["properties"]
    assert "body" not in schema["properties"]
    assert "parameters" in schema["properties"]
    assert_atomic_model_schema(schema, surface="Java method metadata")


def test_single_scalar_action_forwards_closed_schema_to_router():
    router = RecordingRouter()
    schema = {
        "type": "object",
        "properties": {"member": {"type": "string", "maxLength": 2600}},
        "required": ["member"],
        "additionalProperties": False,
    }
    assert _one_structured_action(router, [], name="member", schema=schema) == {
        "member": ""
    }
    assert len(router.calls) == 1
    assert router.calls[0]["parameters"] == schema
    # Runtime output limits are owned by the shared generation budget layer,
    # not invented by each first-pass Java call.
    assert "output_token_ceiling" not in router.calls[0]


def test_large_plan_is_paged_before_inference():
    router = RecordingRouter()
    schema = {
        **_PLAN_SCHEMA,
        "properties": {
            **_PLAN_SCHEMA["properties"],
            "units": {**_PLAN_SCHEMA["properties"]["units"], "minItems": 0},
        },
    }
    result = _one_structured_action(router, [], name="plan", schema=schema)
    assert set(result) == set(schema["properties"])
    assert len(router.calls) > 1
    assert any("count" in call["parameters"]["properties"] for call in router.calls)
    assert all(
        call["parameters"]["additionalProperties"] is False
        and len(call["parameters"]["properties"]) == 1
        for call in router.calls
    )


def test_final_wire_budget_cannot_silently_shrink_complete_action():
    from types import SimpleNamespace

    import pytest

    from minecraft_mod_ai.generation_output_budget import (
        GenerationOutputBudgetError,
        assert_request_output_capacity,
    )

    request = SimpleNamespace(metadata={"mmm_output_token_floor": 15899})
    with pytest.raises(
        GenerationOutputBudgetError,
        match="required=15899 available=2304",
    ):
        assert_request_output_capacity(request, {"max_tokens": 2304})
    assert_request_output_capacity(request, {"max_tokens": 15899})
    assert_request_output_capacity(SimpleNamespace(metadata={}), {"max_tokens": 128})
