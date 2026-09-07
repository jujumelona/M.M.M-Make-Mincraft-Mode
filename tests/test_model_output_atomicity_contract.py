from __future__ import annotations

import pytest

from minecraft_mod_ai import forced_tool_execution_contract as forced
from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_output_atomicity_contract import (
    _same_tool_repair_request,
    assert_atomic_model_schema,
)


def _tool_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "atomic_action",
            "description": "One bounded action",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


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


def test_forced_tool_repair_stays_on_native_tool_wire() -> None:
    schema = _tool_schema()
    request = GenerationRequest(
        messages=({"role": "user", "content": "perform action"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={"type": "function", "function": {"name": "atomic_action"}},
        response_format="text",
    )
    repaired = _same_tool_repair_request(forced, request, "atomic_action", "invalid arguments")

    assert repaired.tools == (schema,)
    assert repaired.tool_validation_schemas == (schema,)
    assert repaired.tool_choice == "required"
    assert repaired.response_format == "text"
    assert repaired.response_schema is None
    assert "JSON document outside the function call" in repaired.messages[-1]["content"]


def test_runtime_installer_replaces_free_form_argument_fallback() -> None:
    assert forced.host_selected_argument_turn.__module__ == "minecraft_mod_ai.model_output_atomicity_contract"
    assert forced.host_selected_mutation_turn.__module__ == "minecraft_mod_ai.model_output_atomicity_contract"
