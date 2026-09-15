from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.generation_verifier_fallback_installation import (
    _fallback_status,
    _failure_fields,
)
from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolDefinition
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


def test_all_rejected_non_visible_tool_call_is_recoverable_observation() -> None:
    request = GenerationRequest(
        tools=(_tool("recover_context"),),
        tool_choice="required",
        parallel_tool_calls=False,
    )
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "call_stale_edit",
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "arguments": "{}",
                },
            }
        ],
    }

    response = _native_tool_generation_response(message, request)

    assert len(response.tool_calls) == 1
    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["original_tool"] == "apply_source_edit"
    assert rejection.arguments["failure_code"] == "TOOL_NOT_VISIBLE"
    assert "non-visible tool" in str(rejection.arguments["error"])


def test_validation_input_change_is_unavailable_not_source_failure() -> None:
    report = SimpleNamespace(
        passed=False,
        status="FAIL",
        error="Project inputs changed during validation; result is not certifiable",
    )

    status, failure_code = _fallback_status(report, "")

    assert status == "UNAVAILABLE"
    assert failure_code == "VALIDATION_INPUTS_CHANGED"
    assert _failure_fields(failure_code) == {
        "failure_class": "validation_state",
        "repairable": False,
        "code": "VALIDATION_INPUTS_CHANGED",
    }


def test_real_gradle_failure_remains_source_failure() -> None:
    report = SimpleNamespace(
        passed=False,
        status="FAIL",
        error="Compilation failed with Java compiler errors",
    )

    status, failure_code = _fallback_status(report, "")

    assert status == "FAIL"
    assert failure_code == "GRADLE_BUILD_FAILED"


def test_java_toolchain_failure_remains_environment_unavailable() -> None:
    report = SimpleNamespace(
        passed=False,
        status="FAIL",
        error="No matching toolchains found for requested specification",
    )

    status, failure_code = _fallback_status(report, "")

    assert status == "UNAVAILABLE"
    assert failure_code == "JAVA_TOOLCHAIN_UNAVAILABLE"
