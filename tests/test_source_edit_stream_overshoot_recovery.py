from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.model_adapters.base import GenerationRequest, GenerationResponse
from minecraft_mod_ai.native_atomic_argument_recovery import host_selected_argument_turn
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def test_source_edit_stream_accepts_oversized_one_shot_chunk_without_losing_source() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "test source mutation",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create DebugToken.java"},),
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
    )
    long_content = (
        "package dev.mmm.debugfixture;\n\n"
        "final class DebugToken {\n"
        + "    static final String VALUE = \"oversized-source-edit-chunk\";\n" * 8
        + "}\n"
    )
    assert len(long_content) > 256
    stream_values = {
        "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
        "content": long_content,
    }
    streamed_fields: list[str] = []

    def current(adapter, page_request):
        del adapter
        schema = page_request.tools[0]["function"]["parameters"]
        properties = schema["properties"]
        if tuple(properties) == ("operation",):
            arguments = {"operation": "create"}
        else:
            assert tuple(properties) == ("chunk", "done")
            assert "maxLength" not in properties["chunk"]
            instruction = str(page_request.messages[-1]["content"])
            assert "at most 256 characters" in instruction
            field_name = next(
                name for name in stream_values if f"field {name!r}" in instruction
            )
            streamed_fields.append(field_name)
            arguments = {"chunk": stream_values[field_name], "done": True}
        return GenerationResponse(
            tool_calls=(
                SimpleNamespace(
                    name="apply_source_edit",
                    arguments=arguments,
                ),
            )
        )

    turn = host_selected_argument_turn(
        current,
        object(),
        request,
        "apply_source_edit",
        prefix="test",
    )

    assert streamed_fields == ["path", "content"]
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {
        "operation": "create_file",
        "path": stream_values["path"],
        "content": long_content,
    }
