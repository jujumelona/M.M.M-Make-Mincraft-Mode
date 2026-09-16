from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolCall
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _admit_model_tool_calls,
    _native_tool_generation_response,
)
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup
from minecraft_mod_ai.progress_aware_tool_loop import _model_tool_rejection_feedback

TOOL = {
    "type": "function",
    "function": {
        "name": "apply_source_edit",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "enum": ["src/main/java/dev/mmm/DebugToken.java"]},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}
SCHEMAS = {"apply_source_edit": TOOL["function"]["parameters"]}


def _call(path: str) -> ToolCall:
    return ToolCall(
        id="call_x",
        name="apply_source_edit",
        arguments={"path": path, "content": "class X {}"},
        raw_arguments=f'{{"path":"{path}","content":"class X {{}}"}}',
    )


def test_qwen_parser_is_transport_only_and_does_not_raise_on_schema_invalid_value():
    text = (
        '<tool_call><function=apply_source_edit>'
        '<parameter=path>wrong.java</parameter>'
        '<parameter=content>class X {}</parameter>'
        '</function></tool_call>'
    )
    visible, calls = parse_qwen_tool_markup(text, SCHEMAS)
    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "apply_source_edit"
    assert calls[0].arguments["path"] == "wrong.java"


def test_single_admission_rejects_schema_invalid_qwen_without_backend_exception():
    text = (
        '<tool_call><function=apply_source_edit>'
        '<parameter=path>wrong.java</parameter>'
        '<parameter=content>class X {}</parameter>'
        '</function></tool_call>'
    )
    request = GenerationRequest(
        tools=(TOOL,),
        tool_choice="required",
        parallel_tool_calls=False,
    )
    response = _native_tool_generation_response({"content": text}, request)
    assert len(response.tool_calls) == 1
    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_SCHEMA_INVALID"


def test_required_missing_and_parallel_violations_are_rejections_not_exceptions():
    missing = _admit_model_tool_calls((), SCHEMAS, tool_choice="required", parallel_tool_calls=False)
    assert missing[0].arguments["failure_code"] == "REQUIRED_TOOL_MISSING"

    parallel = _admit_model_tool_calls(
        (_call("src/main/java/dev/mmm/DebugToken.java"), _call("src/main/java/dev/mmm/DebugToken.java")),
        SCHEMAS,
        tool_choice="required",
        parallel_tool_calls=False,
    )
    assert len(parallel) == 1
    assert parallel[0].arguments["failure_code"] == "PARALLEL_TOOL_CALLS_DISABLED"


def test_admission_is_transactional_no_valid_sibling_executes_with_invalid_sibling():
    admitted = _admit_model_tool_calls(
        (_call("src/main/java/dev/mmm/DebugToken.java"), _call("wrong.java")),
        SCHEMAS,
        tool_choice="required",
        parallel_tool_calls=True,
    )
    assert admitted
    assert all(call.name == "__mmm_rejected_tool_call__" for call in admitted)


def test_progress_loop_consumes_rejection_as_feedback_not_as_runtime_tool():
    rejection = _admit_model_tool_calls(
        (_call("wrong.java"),),
        SCHEMAS,
        tool_choice="required",
        parallel_tool_calls=False,
    )
    feedback = _model_tool_rejection_feedback(rejection)
    assert feedback is not None
    assert "not executed" in feedback
    assert "TOOL_SCHEMA_INVALID" in feedback


def test_phase_handoff_closes_old_tool_protocol_and_preserves_observation_data():
    from minecraft_mod_ai.progress_aware_tool_loop import (
        LoopPhase,
        _compact_phase_tool_transcript,
    )

    messages = [
        {"role": "user", "content": "implement the target"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search_code_rag",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "name": "search_code_rag",
            "content": "public final class DebugToken {}",
        },
    ]

    removed = _compact_phase_tool_transcript(
        messages,
        previous_phase=LoopPhase.OBSERVE,
        next_phase=LoopPhase.ACT,
    )

    assert removed == 2
    assert all(message.get("role") != "tool" for message in messages)
    assert all(not message.get("tool_calls") for message in messages)
    assert messages[-1]["role"] == "system"
    assert "OBSERVE->ACT" in messages[-1]["content"]
    assert "public final class DebugToken {}" in messages[-1]["content"]
    assert "search_code_rag" not in messages[-1]["content"]


def test_phase_handoff_is_generic_through_act_to_verify():
    from minecraft_mod_ai.progress_aware_tool_loop import (
        LoopPhase,
        _compact_phase_tool_transcript,
    )

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_edit",
                    "type": "function",
                    "function": {
                        "name": "apply_source_edit",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_edit",
            "name": "apply_source_edit",
            "content": "workspace_changed=true; sha256=abc123",
        },
    ]

    removed = _compact_phase_tool_transcript(
        messages,
        previous_phase=LoopPhase.ACT,
        next_phase=LoopPhase.VERIFY,
    )

    assert removed == 2
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "ACT->VERIFY" in messages[0]["content"]
    assert "workspace_changed=true" in messages[0]["content"]
    assert "apply_source_edit" not in messages[0]["content"]


def test_phase_handoff_is_wired_before_next_model_turn():
    import inspect

    from minecraft_mod_ai.progress_aware_tool_loop import _generate_with_tools_impl

    source = inspect.getsource(_generate_with_tools_impl)
    assert "state.phase != last_prompt_phase" in source
    assert "_compact_phase_tool_transcript(" in source
    assert "phase_tool_transcript_handoff" in source
