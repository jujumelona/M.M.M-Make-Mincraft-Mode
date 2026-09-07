from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
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


def _valid_page_response(request: GenerationRequest) -> GenerationResponse:
    assert request.response_format == "json"
    assert isinstance(request.response_schema, dict)
    assert request.tools == ()
    assert request.tool_validation_schemas == ()
    assert request.tool_choice is None
    page_properties = request.response_schema["properties"]
    assert len(page_properties) <= 4
    return GenerationResponse(
        content=json.dumps(
            {name: f"value-{name}" for name in page_properties},
            sort_keys=True,
        )
    )


def test_large_model_authored_schema_is_rejected_before_generation() -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {f"nested_{inner}": {"type": "string"} for inner in range(4)},
                "additionalProperties": False,
            }
            for index in range(20)
        },
        "additionalProperties": False,
    }
    with pytest.raises(ModelConfigurationError, match="MODEL_STRUCTURE_ATOMICITY"):
        assert_atomic_model_schema(schema, surface="regression")


def test_small_atomic_schema_remains_allowed() -> None:
    assert_atomic_model_schema(
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        surface="regression",
    )


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

    assert len(observed) == 3
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "large_host_action"
    assert response.tool_calls[0].arguments == {
        f"field_{index}": f"value-field_{index}" for index in range(12)
    }
    assert all(turn.response_format == "json" for turn in observed)
    assert all(isinstance(turn.response_schema, dict) for turn in observed)
    assert all(turn.tools == () for turn in observed)


def test_mutation_recovery_uses_the_same_bounded_argument_only_json_pages() -> None:
    request = _large_request("apply_source_edit", field_count=9)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        return _valid_page_response(page_request)

    response = forced.host_selected_mutation_turn(
        current,
        object(),
        request,
        "apply_source_edit",
    )

    assert len(observed) == 3
    assert response.tool_calls[0].name == "apply_source_edit"
    assert response.tool_calls[0].id.startswith("host_mutation_")
    assert all(turn.response_format == "json" for turn in observed)
    assert all(isinstance(turn.response_schema, dict) for turn in observed)
    assert all(turn.tools == () for turn in observed)


def test_invalid_json_page_repair_stays_argument_only_and_schema_bounded() -> None:
    request = _large_request("repairable_action", field_count=4)
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        if len(observed) == 1:
            return GenerationResponse(content="not-json")
        return _valid_page_response(page_request)

    response = forced.host_selected_argument_turn(
        current,
        object(),
        request,
        "repairable_action",
    )

    assert len(observed) == 2
    assert response.tool_calls[0].name == "repairable_action"
    assert all(turn.response_format == "json" for turn in observed)
    assert all(isinstance(turn.response_schema, dict) for turn in observed)
    assert all(turn.tools == () for turn in observed)
    assert "Repair the arguments only" in observed[1].messages[-1]["content"]


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
