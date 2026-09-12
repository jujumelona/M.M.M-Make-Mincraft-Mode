from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    GenerationResponse,
    ToolCall,
)
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.native_atomic_argument_recovery import (
    _page_result,
    _source_edit_detail_schema,
    host_selected_argument_turn,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def _source_edit_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one source edit.",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }


def test_source_edit_detail_schema_preserves_unbounded_source_contract() -> None:
    detail = _source_edit_detail_schema(SOURCE_EDIT_SCHEMA, "create_file")

    assert "maxLength" not in detail["properties"]["path"]
    assert "maxLength" not in detail["properties"]["content"]

    content = "x" * 4096
    turn = SimpleNamespace(
        tool_calls=(
            SimpleNamespace(
                name="apply_source_edit",
                arguments={
                    "path": "src/main/resources/mmm/large.txt",
                    "content": content,
                },
            ),
        )
    )

    arguments, error, _ = _page_result(
        turn,
        detail,
        SOURCE_EDIT_SCHEMA,
        "apply_source_edit",
    )

    assert error == ""
    assert arguments == {
        "path": "src/main/resources/mmm/large.txt",
        "content": content,
    }


def test_source_edit_reassembles_unbounded_strings_from_atomic_chunks() -> None:
    long_path = "generated/" + ("nested/" * 40) + "DebugTokenItem.java"
    long_content = "".join(
        f"line_{index:03d}=debug_token_payload_{index:03d};\n"
        for index in range(40)
    )
    expected = {
        "operation": "create_file",
        "path": long_path,
        "content": long_content,
    }
    stream_values = {
        "path": long_path,
        "content": long_content,
    }
    offsets = {name: 0 for name in stream_values}
    observed_schemas: list[Mapping[str, object]] = []

    def current(_adapter: object, request: GenerationRequest) -> GenerationResponse:
        tool = request.tools[0]
        assert isinstance(tool, Mapping)
        function = tool["function"]
        assert isinstance(function, Mapping)
        schema = function["parameters"]
        assert isinstance(schema, Mapping)
        observed_schemas.append(schema)
        assert_atomic_model_schema(
            schema,
            surface="test host-selected forced-function argument page",
        )

        properties = schema.get("properties")
        assert isinstance(properties, Mapping)
        if "operation" in properties:
            arguments = {"operation": "create_file"}
        elif set(properties) == {"chunk", "done"}:
            instruction = str(request.messages[-1]["content"])
            field_name = next(
                name
                for name in stream_values
                if f"field {name!r}" in instruction
            )
            value = stream_values[field_name]
            offset = offsets[field_name]
            chunk = value[offset : offset + 256]
            offsets[field_name] += len(chunk)
            arguments = {
                "chunk": chunk,
                "done": offsets[field_name] >= len(value),
            }
        else:
            raise AssertionError(f"unexpected atomic schema: {schema}")

        return GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="test_call",
                    name="apply_source_edit",
                    arguments=arguments,
                ),
            )
        )

    tool = _source_edit_tool()
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create the debug token source file."},),
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
    )

    response = host_selected_argument_turn(
        current,
        object(),
        request,
        "apply_source_edit",
        prefix="test",
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments == expected
    assert len(long_path) > 256
    assert len(long_content) > 256
    assert offsets == {
        "path": len(long_path),
        "content": len(long_content),
    }

    chunk_schemas = [
        schema
        for schema in observed_schemas
        if isinstance(schema.get("properties"), Mapping)
        and set(schema["properties"]) == {"chunk", "done"}
    ]
    assert chunk_schemas
    assert all(
        schema["properties"]["chunk"]["maxLength"] == 256
        for schema in chunk_schemas
    )


def test_source_edit_stream_rejects_non_progress_without_total_length_cap() -> None:
    tool = _source_edit_tool()
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create the source file."},),
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
    )

    def current(_adapter: object, page_request: GenerationRequest) -> GenerationResponse:
        tool_schema = page_request.tools[0]
        assert isinstance(tool_schema, Mapping)
        function = tool_schema["function"]
        assert isinstance(function, Mapping)
        schema = function["parameters"]
        assert isinstance(schema, Mapping)
        properties = schema["properties"]
        assert isinstance(properties, Mapping)
        if "operation" in properties:
            arguments = {"operation": "create_file"}
        else:
            arguments = {"chunk": "", "done": False}
        return GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="test_call",
                    name="apply_source_edit",
                    arguments=arguments,
                ),
            )
        )

    with pytest.raises(ModelConfigurationError, match="made no progress"):
        host_selected_argument_turn(
            current,
            object(),
            request,
            "apply_source_edit",
        )
