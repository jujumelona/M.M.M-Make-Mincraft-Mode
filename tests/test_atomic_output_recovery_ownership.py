from __future__ import annotations

from minecraft_mod_ai.llama_finish_reason_contract import (
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
    completion_boundary_error,
    completion_boundary_kind,
)
from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    ModelBackendError,
    ModelConfigurationError,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _assistant_prefill_payload
from minecraft_mod_ai.progress_aware_tool_loop import _atomic_output_recovery_instruction


def _source_edit_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one source edit",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _forced_source_edit_choice() -> dict:
    return {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }


def test_assistant_prefill_preserves_forced_tool_contract() -> None:
    tools = [_source_edit_schema()]
    choice = _forced_source_edit_choice()
    original = {
        "model": "local",
        "messages": [{"role": "user", "content": "edit one target"}],
        "tools": tools,
        "tool_choice": choice,
        "parallel_tool_calls": False,
        "max_tokens": 1886,
    }

    continued = _assistant_prefill_payload(
        original,
        {"role": "assistant", "content": "partial bounded action"},
    )

    assert continued["tools"] == tools
    assert continued["tool_choice"] == choice
    assert continued["parallel_tool_calls"] is False
    assert continued["max_tokens"] == 1886


def test_atomic_recovery_instruction_forces_one_structural_edit() -> None:
    request = GenerationRequest(
        tools=(_source_edit_schema(),),
        tool_choice=_forced_source_edit_choice(),
        parallel_tool_calls=False,
    )

    instruction = _atomic_output_recovery_instruction(request)

    assert "exactly one" in instruction
    assert "one small semantic edit" in instruction
    assert "create_java_type" in instruction
    assert "add_java_import" in instruction
    assert "insert_java_member" in instruction
    assert "never create a complete Java file" in instruction


def test_atomic_stall_preserves_typed_output_boundary() -> None:
    boundary = LlamaCompletionBoundaryError(
        "bounded output exhausted",
        kind=OUTPUT_EXHAUSTED,
        partial_message={"role": "assistant", "content": "partial"},
        prompt_tokens=12211,
        completion_tokens=6935,
        max_tokens=6935,
    )
    stalled = ModelConfigurationError(
        "ATOMIC_ACTION_OUTPUT_STALLED: bounded atomic retry also exhausted"
    )
    stalled.__cause__ = boundary
    backend = ModelBackendError(role="coder", model_id="qwen", cause=stalled)
    backend.__cause__ = stalled

    # The terminal owner can still classify and report the original boundary without
    # starting a second generation loop or resetting HostRunState.
    assert completion_boundary_error(backend) is boundary
    assert completion_boundary_kind(backend) == OUTPUT_EXHAUSTED


def test_unrecovered_output_boundary_remains_classifiable() -> None:
    boundary = LlamaCompletionBoundaryError(
        "bounded output exhausted",
        kind=OUTPUT_EXHAUSTED,
        max_tokens=100,
        completion_tokens=100,
    )

    assert completion_boundary_kind(boundary) == OUTPUT_EXHAUSTED
