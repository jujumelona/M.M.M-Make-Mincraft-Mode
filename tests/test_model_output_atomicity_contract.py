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


def test_runtime_installer_routes_forced_recovery_to_native_atomic_pages() -> None:
    assert forced.host_selected_argument_turn.__module__ == "minecraft_mod_ai.native_atomic_argument_recovery"
    assert forced.host_selected_mutation_turn.__module__ == "minecraft_mod_ai.native_atomic_argument_recovery"


def test_large_host_owned_argument_container_is_decomposed_without_raw_json_turns() -> None:
    properties = {f"field_{index}": {"type": "string"} for index in range(12)}
    original_schema = {
        "type": "function",
        "function": {
            "name": "large_host_action",
            "description": "host-owned container",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "perform the already-selected action"},),
        tools=(original_schema,),
        tool_validation_schemas=(original_schema,),
        tool_choice={"type": "function", "function": {"name": "large_host_action"}},
        response_format="text",
    )
    observed: list[GenerationRequest] = []

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        observed.append(page_request)
        assert page_request.response_format == "text"
        assert page_request.response_schema is None
        assert len(page_request.tools) == 1
        tool = page_request.tools[0]
        assert isinstance(tool, dict)
        function = tool["function"]
        assert function["name"] == "mmm_submit_argument_page"
        page_properties = function["parameters"]["properties"]
        assert len(page_properties) <= 4
        arguments = {name: f"value-{name}" for name in page_properties}
        return GenerationResponse(
            tool_calls=(
                ToolCall(
                    id=f"page-{len(observed)}",
                    name="mmm_submit_argument_page",
                    arguments=arguments,
                ),
            )
        )

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
        name: f"value-{name}" for name in properties
    }
    assert all(turn.response_format != "json" for turn in observed)
    assert all(turn.tools for turn in observed)
