from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
)
from minecraft_mod_ai.model_output_atomicity_contract import (
    assert_atomic_model_schema,
    assert_installed,
    install,
    is_atomic_model_schema,
)


def _large_request(name: str, *, field_count: int = 12) -> GenerationRequest:
    properties = {
        f"field_{index}": {"type": "string", "maxLength": 64}
        for index in range(field_count)
    }
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "host-owned container",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
    return GenerationRequest(
        messages=({"role": "user", "content": "perform the already-selected action"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={"type": "function", "function": {"name": name}},
        response_format="text",
    )


def _valid_page_response(request: GenerationRequest) -> GenerationResponse:
    from minecraft_mod_ai.model_adapters.base import ToolCall

    assert request.response_format == "text"
    assert request.response_schema is None
    assert len(request.tools) == 1
    page_tool = request.tools[0]
    action_name = page_tool["function"]["name"]
    page_properties = page_tool["function"]["parameters"]["properties"]
    assert len(page_properties) <= 3
    return GenerationResponse(
        tool_calls=(
            ToolCall(
                id="call_test",
                name=action_name,
                arguments={name: f"value-{name}" for name in page_properties},
            ),
        )
    )


def test_large_closed_model_schema_is_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {
                    f"nested_{inner}": {"type": "string", "maxLength": 64}
                    for inner in range(4)
                },
                "additionalProperties": False,
            }
            for index in range(20)
        },
        "additionalProperties": False,
    }
    with pytest.raises(ModelConfigurationError):
        assert_atomic_model_schema(schema, surface="regression")
    assert not is_atomic_model_schema(schema)


def test_small_atomic_schema_remains_allowed() -> None:
    assert_atomic_model_schema(
        {
            "type": "object",
            "properties": {"value": {"type": "string", "maxLength": 64}},
            "required": ["value"],
            "additionalProperties": False,
        },
        surface="regression",
    )


def test_native_tool_decision_uses_the_same_atomicity_boundary() -> None:
    calls: list[str] = []

    class DummyRouter:
        def generate_text(self, role, messages, **kwargs):
            calls.append("text")
            return "ok"

        def generate_tool_decision(
            self,
            role,
            messages,
            *,
            tool_name,
            parameters,
            description="",
        ):
            calls.append("tool")
            return {"ok": True}

    module = SimpleNamespace(ModelRouter=DummyRouter)
    install(model_router_module=module)
    assert_installed(model_router_module=module)
    oversized = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {
                    f"nested_{inner}": {"type": "string", "maxLength": 64}
                    for inner in range(4)
                },
                "additionalProperties": False,
            }
            for index in range(20)
        },
        "additionalProperties": False,
    }

    with pytest.raises(ModelConfigurationError):
        DummyRouter().generate_tool_decision(
            "planner",
            ({"role": "user", "content": "fill it"},),
            tool_name="oversized_planner_contract",
            parameters=oversized,
        )
    assert calls == []


def test_native_tool_decision_allows_bounded_closed_schema() -> None:
    calls: list[str] = []

    class DummyRouter:
        def generate_text(self, role, messages, **kwargs):
            return "ok"

        def generate_tool_decision(
            self,
            role,
            messages,
            *,
            tool_name,
            parameters,
            description="",
        ):
            calls.append(tool_name)
            return {"value": "ok"}

    module = SimpleNamespace(ModelRouter=DummyRouter)
    install(model_router_module=module)
    result = DummyRouter().generate_tool_decision(
        "planner",
        ({"role": "user", "content": "fill it"},),
        tool_name="bounded_planner_contract",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "maxLength": 64}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    assert result == {"value": "ok"}
    assert calls == ["bounded_planner_contract"]


def test_legacy_raw_json_argument_recovery_helpers_are_removed() -> None:
    assert not hasattr(forced, "_argument_page_request")
    assert not hasattr(forced, "_argument_attempt")
    assert not hasattr(forced, "_argument_failure")


def test_large_host_owned_argument_container_is_decomposed_into_bounded_json_pages() -> None:
    request = _large_request("large_host_action")
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request)

    response = forced.host_selected_argument_turn(
        current,
        object(),
        request,
        "large_host_action",
    )

    assert len(observed) == 4
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "large_host_action"
    assert response.tool_calls[0].arguments == {
        f"field_{index}": f"value-field_{index}" for index in range(12)
    }
    assert all(turn.response_format == "text" for turn in observed)
    assert all(turn.response_schema is None for turn in observed)
    assert all(len(turn.tools) == 1 for turn in observed)


def test_mutation_recovery_uses_the_same_bounded_argument_only_json_pages() -> None:
    request = _large_request("generic_mutation_action", field_count=9)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request)

    response = forced.host_selected_mutation_turn(
        current,
        object(),
        request,
        "generic_mutation_action",
    )

    assert len(observed) == 3
    assert response.tool_calls[0].name == "generic_mutation_action"
    assert response.tool_calls[0].id.startswith("host_mutation_")
    assert all(turn.response_format == "text" for turn in observed)
    assert all(turn.response_schema is None for turn in observed)
    assert all(len(turn.tools) == 1 for turn in observed)


def test_invalid_json_page_repair_stays_argument_only_and_schema_bounded() -> None:
    request = _large_request("repairable_action", field_count=3)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        if len(observed) == 1:
            from minecraft_mod_ai.model_adapters.base import ToolCall
            return GenerationResponse(tool_calls=(ToolCall(id="call_bad", name="repairable_action", arguments={"bad": 1}),))
        return _valid_page_response(page_request)

    response = forced.host_selected_argument_turn(
        current,
        object(),
        request,
        "repairable_action",
    )

    assert len(observed) == 2
    assert response.tool_calls[0].name == "repairable_action"
    assert all(turn.response_format == "text" for turn in observed)
    assert all(turn.response_schema is None for turn in observed)
    assert all(len(turn.tools) == 1 for turn in observed)
    assert "Repair the function arguments only" in observed[1].messages[-1]["content"]


def test_oversized_single_nested_field_fails_closed_before_model_generation() -> None:
    nested_properties = {
        f"nested_{index}": {"type": "string", "maxLength": 64} for index in range(40)
    }
    schema = {
        "type": "function",
        "function": {
            "name": "oversized_nested_action",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": nested_properties,
                        "required": list(nested_properties),
                        "additionalProperties": False,
                    }
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "perform the fixed action"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={"type": "function", "function": {"name": "oversized_nested_action"}},
        response_format="text",
    )
    calls = 0

    def current(_adapter: object, _page_request: GenerationRequest) -> GenerationResponse:
        nonlocal calls
        calls += 1
        raise AssertionError("oversized page must be rejected before generation")

    with pytest.raises(ModelConfigurationError):
        forced.host_selected_argument_turn(
            current,
            object(),
            request,
            "oversized_nested_action",
        )
    assert calls == 0


@pytest.mark.parametrize("kind", ["depth", "properties", "unbounded_string", "unbounded_array"])
def test_schema_violating_bounds_is_rejected(kind):
    if kind == "depth":
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {"level3": {"type": "string", "maxLength": 32}},
                            "additionalProperties": False,
                        }
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }
    elif kind == "properties":
        schema = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string", "maxLength": 32} for i in range(10)},
            "additionalProperties": False,
        }
    elif kind == "unbounded_string":
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 32},
                }
            },
            "additionalProperties": False,
        }

    with pytest.raises(ModelConfigurationError):
        assert_atomic_model_schema(schema, surface=f"bound-violation-{kind}")
    assert not is_atomic_model_schema(schema)


def test_open_object_template_still_rejected():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    with pytest.raises(ModelConfigurationError, match="MODEL_JSON_TEMPLATE_REQUIRED"):
        assert_atomic_model_schema(schema, surface="open template")
    assert not is_atomic_model_schema(schema)
