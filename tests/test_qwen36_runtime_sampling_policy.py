from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.qwen_agent_family_contract import _apply_family_payload_policy


_TOOL = {
    "type": "function",
    "function": {
        "name": "read_project_file",
        "description": "Read one project file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

_PROFILES = {
    "general_thinking": {
        "temperature": 0.31,
        "top_p": 0.71,
        "top_k": 17,
        "presence_penalty": 0.21,
        "repeat_penalty": 0.91,
    },
    "precise_coding": {
        "temperature": 0.23,
        "top_p": 0.67,
        "top_k": 13,
        "presence_penalty": 0.12,
        "repeat_penalty": 0.89,
    },
    "non_thinking": {
        "temperature": 0.11,
        "top_p": 0.61,
        "top_k": 7,
        "presence_penalty": 0.04,
        "repeat_penalty": 0.83,
        "reasoning_effort": "none",
    },
}


def _config(*, effort: str = "") -> SimpleNamespace:
    extra = {
        "runtime_contract": "qwen",
        "agent_thinking": True,
        "sampling_profiles": _PROFILES,
    }
    if effort:
        extra["thinking_reasoning_effort"] = effort
    return SimpleNamespace(
        model_id="vendor/arbitrary-runtime-model",
        role="researcher",
        extra=extra,
    )


def _request(**kwargs) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "Solve the task."},),
        **kwargs,
    )


def _payload(*, role: str, request: GenerationRequest, effort: str = "") -> dict:
    return _apply_family_payload_policy(
        {"temperature": 0.0, "repetition_penalty": 1.05},
        config=_config(effort=effort),
        role=role,
        request=request,
    )


def test_general_thinking_sampling_comes_from_registry_metadata() -> None:
    payload = _payload(role="researcher", request=_request())

    assert payload["temperature"] == 0.31
    assert payload["top_p"] == 0.71
    assert payload["top_k"] == 17
    assert payload["presence_penalty"] == 0.21
    assert payload["repeat_penalty"] == 0.91
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "repetition_penalty" not in payload


def test_coder_sampling_comes_from_precise_registry_profile() -> None:
    payload = _payload(role="coder_safe", request=_request())

    assert payload["temperature"] == 0.23
    assert payload["top_p"] == 0.67
    assert payload["top_k"] == 13
    assert payload["presence_penalty"] == 0.12
    assert payload["repeat_penalty"] == 0.89


def test_auto_tool_agent_preserves_thinking_without_model_identity() -> None:
    payload = _payload(
        role="researcher",
        request=_request(tools=(_TOOL,), tool_choice="auto"),
        effort="high",
    )

    assert payload["temperature"] == 0.31
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "preserve_thinking": True,
    }
    assert "reasoning_effort" not in payload


def test_json_fill_uses_registry_non_thinking_profile() -> None:
    payload = _payload(role="planner", request=_request(response_format="json"))

    assert payload["temperature"] == 0.11
    assert payload["top_p"] == 0.61
    assert payload["top_k"] == 7
    assert payload["presence_penalty"] == 0.04
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_forced_tool_does_not_apply_registry_sampling_profile() -> None:
    payload = _payload(
        role="researcher",
        request=_request(
            tools=(_TOOL,),
            tool_choice={
                "type": "function",
                "function": {"name": "read_project_file"},
            },
        ),
    )

    assert payload["temperature"] == 0.0
    assert payload["repetition_penalty"] == 1.05
    assert "chat_template_kwargs" not in payload
