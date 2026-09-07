from __future__ import annotations

import pytest

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


def _large_request(name: str, *, field_count: int = 12) -> GenerationRequest:
    properties = {f"field_{index}": {"type": "string"} for index in range(field_count)}
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


def _valid_page_response(request: GenerationRequest, *, call_id: str) -> GenerationResponse:
    assert request.response_format == "text"
    assert request.response_schema is None
    assert len(request.tools) == 1
    tool = request.tools[0]
    assert isinstance(tool, dict)
    function = tool["function"]
    assert function["name"] == "mmm_submit_argument_page"
    page_properties = function["parameters"]["properties"]
    assert len(page_properties) <= 4
    return GenerationResponse(
        tool_calls=(
            ToolCall(
                id=call_id,
                name="mmm_submit_argument_page",
                arguments={name: f"value-{name}" for name in page_properties},
            ),
        )
    )


def test_large_model_authored_schema_is_rejected_before_generation() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {f"nested_{inner}": {"type": "string"} for inner in range(4)},
            }
            for index in range(20)
        },
    }
    with pytest.raises(ModelConfigurationError, match="MODEL_STRUCTURE_ATOMICITY"):
        assert_atomic_model_schema(schema, surface="regression")


def test_small_atomic_schema_remains_allowed() -> None:
    assert_atomic_model_schema(
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        surface="regression",
    )


def test_legacy_raw_json_argument_recovery_helpers_are_removed() -> None:
    assert not hasattr(forced, "_argument_page_request")
    assert not hasattr(forced, "_argument_attempt")
    assert not hasattr(forced, "_argument_failure")


def test_large_host_owned_argument_container_is_decomposed_without_raw_json_turns() -> None:
    request = _large_request("large_host_action")
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request, call_id=f"page-{len(observed)}")

    response = forced.host_selected_argument_turn(
        current,
        object(),
        request,
        "large_host_action",
    )

    assert len(observed) == 3
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "large_host_action"
    assert response.tool_calls[0].arguments == {
        f"field_{index}": f"value-field_{index}" for index in range(12)
    }
    assert all(turn.response_format != "json" for turn in observed)
    assert all(turn.tools for turn in observed)


def test_mutation_recovery_uses_the_same_native_atomic_pages() -> None:
    request = _large_request("apply_source_edit", field_count=9)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request, call_id=f"mutation-page-{len(observed)}")

    response = forced.host_selected_mutation_turn(
        current,
        object(),
        request,
        "apply_source_edit",
    )

    assert len(observed) == 3
    assert response.tool_calls[0].name == "apply_source_edit"
    assert response.tool_calls[0].id.startswith("host_mutation_")
    assert all(turn.response_format == "text" for turn in observed)
    assert all(turn.response_schema is None for turn in observed)
    assert all(turn.tools for turn in observed)


def test_invalid_native_page_repair_never_switches_to_raw_json() -> None:
    request = _large_request("repairable_action", field_count=4)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        if len(observed) == 1:
            return GenerationResponse(content='{"field_0":"raw-json-is-not-accepted"}')
        return _valid_page_response(page_request, call_id="repaired-page")

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
    assert all(turn.tools for turn in observed)


def test_oversized_single_nested_field_fails_closed_before_model_generation() -> None:
    nested_properties = {
        f"nested_{index}": {"type": "string"} for index in range(40)
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

    with pytest.raises(ModelConfigurationError, match="MODEL_STRUCTURE_ATOMICITY"):
        forced.host_selected_argument_turn(
            current,
            object(),
            request,
            "oversized_nested_action",
        )

    assert calls == 0
