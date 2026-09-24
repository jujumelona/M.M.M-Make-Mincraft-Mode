from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationResponse, ToolCall
from tools.qwen_live_integration_capture import (
    SOURCE_EDIT_NEW,
    SOURCE_EDIT_OLD,
    SOURCE_EDIT_TARGET,
    _production_validation,
    _request_for_scenario,
)


def test_source_edit_capture_uses_production_tool_request_shape() -> None:
    request = _request_for_scenario("source-edit", max_tokens=2048)

    assert len(request.tools) == 1
    assert request.tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }
    assert request.parallel_tool_calls is False
    assert request.metadata["tool_stage"] == "generation"
    assert request.metadata["mmm_output_token_ceiling"] == 2048


def test_source_edit_capture_validates_admitted_production_tool_call() -> None:
    arguments = {
        "operation": "replace_exact",
        "path": SOURCE_EDIT_TARGET,
        "old": SOURCE_EDIT_OLD,
        "new": SOURCE_EDIT_NEW,
        "count": 1,
    }
    turn = GenerationResponse(
        tool_calls=(
            ToolCall(
                id="capture",
                name="apply_source_edit",
                arguments=arguments,
                raw_arguments="",
            ),
        )
    )

    accepted, error, _content = _production_validation(turn, scenario="source-edit")
    assert accepted is True
    assert error is None


def test_json_capture_still_uses_structured_production_validation() -> None:
    request = _request_for_scenario("json", max_tokens=None)
    assert request.response_format == "json"
    accepted, error, content = _production_validation(
        GenerationResponse(content='{"answer":"ok"}'),
        scenario="json",
    )
    assert accepted is True
    assert error is None
    assert content == '{"answer":"ok"}'
