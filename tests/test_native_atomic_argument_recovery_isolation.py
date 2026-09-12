from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest, GenerationResponse, ToolCall
from minecraft_mod_ai.native_atomic_argument_recovery import host_selected_argument_turn


def _request() -> GenerationRequest:
    parameters = {
        "type": "object",
        "properties": {"actor": {"type": "string"}},
        "required": ["actor"],
        "additionalProperties": False,
    }
    tool = {
        "type": "function",
        "function": {
            "name": "submit_action",
            "description": "test action",
            "parameters": parameters,
        },
    }
    return GenerationRequest(
        messages=(
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "obsolete task"},
            {"role": "assistant", "content": "research prose that must not survive isolation"},
            {"role": "tool", "name": "search", "content": "stale tool residue"},
            {"role": "user", "content": "current task: actor is player"},
        ),
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "submit_action"}},
    )


def test_repeated_invalid_fixed_point_switches_to_isolated_context() -> None:
    calls: list[GenerationRequest] = []

    def current(adapter, page_request: GenerationRequest) -> GenerationResponse:
        del adapter
        calls.append(page_request)
        if len(calls) < 3:
            return GenerationResponse(content="same invalid prose")

        roles = [message["role"] for message in page_request.messages]
        contents = [str(message.get("content", "")) for message in page_request.messages]
        assert roles == ["system", "user", "user"]
        assert "system policy" in contents
        assert "current task: actor is player" in contents
        assert all("obsolete task" not in content for content in contents)
        assert all("research prose" not in content for content in contents)
        assert all("stale tool residue" not in content for content in contents)
        assert "isolated fixed-point recovery" in contents[-1]
        return GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="submit_action",
                    arguments={"actor": "player"},
                ),
            )
        )

    turn = host_selected_argument_turn(
        current,
        object(),
        _request(),
        "submit_action",
        prefix="test",
    )

    assert len(calls) == 3
    assert turn.tool_calls[0].arguments == {"actor": "player"}


def test_isolated_fixed_point_failure_is_terminal() -> None:
    import pytest

    from minecraft_mod_ai.model_adapters.base import ModelConfigurationError

    calls: list[GenerationRequest] = []

    def current(adapter, page_request: GenerationRequest) -> GenerationResponse:
        del adapter
        calls.append(page_request)
        return GenerationResponse(content="same invalid prose")

    with pytest.raises(ModelConfigurationError, match="isolated forced argument-page recovery"):
        host_selected_argument_turn(
            current,
            object(),
            _request(),
            "submit_action",
            prefix="test",
        )

    assert len(calls) == 3
    assert [message["role"] for message in calls[-1].messages] == [
        "system",
        "user",
        "user",
    ]
