from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_hardware_policy
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_context_budget import effective_context_tokens
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.qwen_agent_family_contract import (
    _apply_family_payload_policy,
    _strip_reasoning_history,
)
from minecraft_mod_ai.qwen_family_capabilities import qwen_family_capabilities


def _extra(family: str) -> dict[str, object]:
    preservation = family in {"qwen3.6", "qwen3.8"}
    effort = family == "qwen3.8"
    return {
        "runtime_contract": "qwen",
        "qwen_family": family,
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": preservation,
        "qwen_reasoning_effort": effort,
        "qwen_assistant_prefill": True,
        "agent_thinking": True,
        "thinking_reasoning_effort": "xhigh" if effort else "",
        "sampling_profiles": {
            "general_thinking": {"temperature": 1.0},
            "precise_coding": {"temperature": 0.6},
            "non_thinking": {"temperature": 0.7},
        },
    }


def _config(family: str, *, role: str = "coder") -> SimpleNamespace:
    return SimpleNamespace(role=role, extra=_extra(family))


def _tool_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            {"role": "user", "content": "act"},
            {
                "role": "assistant",
                "content": "prior",
                "reasoning_content": "private trace",
                "reasoning": "private trace alias",
            },
            {"role": "user", "content": "continue"},
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        tool_choice="auto",
    )


@pytest.mark.parametrize(
    ("profile", "family"),
    [
        ("Qwen3.5-9B_6GB", "qwen3.5"),
        ("Qwen3.8-27B_18GB", "qwen3.8"),
    ],
)
def test_both_registry_copies_declare_exact_family_capabilities(
    profile: str,
    family: str,
) -> None:
    for path in ("config/model_registry.yaml", "minecraft_mod_ai/config/model_registry.yaml"):
        config = ModelRegistry(path).role(profile, "coder")
        capabilities = qwen_family_capabilities(config, required=True)
        assert capabilities is not None
        assert capabilities.family == family
        assert capabilities.tool_markup == "qwen3_coder_xml"
        assert capabilities.assistant_prefill is True


def test_qwen_runtime_without_family_contract_fails_closed() -> None:
    config = SimpleNamespace(extra={"runtime_contract": "qwen"})

    with pytest.raises(ValueError, match="requires qwen_family"):
        qwen_family_capabilities(config)


@pytest.mark.parametrize(
    ("family", "expected_kwargs"),
    [
        ("qwen3.5", {"enable_thinking": False}),
        (
            "qwen3.6",
            {"enable_thinking": False, "preserve_thinking": False},
        ),
        (
            "qwen3.8",
            {"enable_thinking": False, "preserve_thinking": False},
        ),
    ],
)
def test_family_action_pages_use_explicit_official_non_thinking_controls(
    family: str,
    expected_kwargs: dict[str, bool],
) -> None:
    payload = _apply_family_payload_policy(
        {
            "temperature": 0.0,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": True},
        },
        config=_config(family),
        role="coder",
        request=_tool_request(),
    )

    assert payload["temperature"] == 0.7
    assert payload["chat_template_kwargs"] == expected_kwargs
    assert "reasoning_effort" not in payload


def test_qwen38_planning_uses_its_model_native_effort_and_preservation() -> None:
    payload = _apply_family_payload_policy(
        {"temperature": 0.0},
        config=_config("qwen3.8", role="researcher"),
        role="researcher",
        request=GenerationRequest(
            messages=({"role": "user", "content": "plan"},),
        ),
    )

    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize("family", ["qwen3.5", "qwen3.6", "qwen3.8"])
def test_action_request_strips_historical_private_reasoning(family: str) -> None:
    stripped = _strip_reasoning_history(_tool_request())

    assistant = next(
        message for message in stripped.messages if message.get("role") == "assistant"
    )
    assert assistant["content"] == "prior"
    assert "reasoning_content" not in assistant
    assert "reasoning" not in assistant


def test_qwen38_rejects_unsupported_reasoning_effort() -> None:
    config = _config("qwen3.8", role="researcher")
    config.extra["thinking_reasoning_effort"] = "none"

    with pytest.raises(ValueError, match="must be xhigh, medium, or low"):
        _apply_family_payload_policy(
            {"temperature": 0.0},
            config=config,
            role="researcher",
            request=GenerationRequest(
                messages=({"role": "user", "content": "plan"},),
            ),
        )


def test_fully_composed_qwen38_tool_payload_never_leaks_reasoning_none() -> None:
    config = ModelRegistry("config/model_registry.yaml").role(
        "Qwen3.8-27B_18GB", "coder"
    )
    request = _tool_request()

    payload = llama_server_hardware_policy._server_payload(
        SimpleNamespace(config=config), request
    )

    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "reasoning_effort" not in payload
    assert payload["tools"] == list(request.tools)
    assert payload["tool_choice"] == "auto"
    assert 8192 < payload["max_tokens"] < effective_context_tokens(config)


@pytest.mark.parametrize(
    "profile",
    ("Qwen3.5-9B_6GB", "Qwen3.8-27B_18GB"),
)
@pytest.mark.parametrize("request_kind", ("plain", "json", "tool"))
def test_all_local_qwen_families_use_finite_dynamic_completion(
    monkeypatch, profile: str, request_kind: str
) -> None:
    monkeypatch.delenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("MMM_GENERATION_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TEXT_MAX_TOKENS", raising=False)
    config = ModelRegistry("config/model_registry.yaml").role(profile, "coder")
    if request_kind == "plain":
        request = GenerationRequest(
            messages=({"role": "user", "content": "implement one task"},),
        )
    elif request_kind == "json":
        request = GenerationRequest(
            messages=({"role": "user", "content": "emit one section"},),
            response_format="json",
            response_schema={
                "type": "object",
                "properties": {"section": {"type": "string"}},
                "required": ["section"],
                "additionalProperties": False,
            },
        )
    else:
        request = _tool_request()

    payload = llama_server_hardware_policy._server_payload(
        SimpleNamespace(config=config), request
    )

    assert config.max_new_tokens == 8192
    assert config.extra["dynamic_output_budget"] is True
    assert 8192 < payload["max_tokens"] < effective_context_tokens(config)


def test_fully_composed_qwen38_required_and_json_pages_remove_generic_none() -> None:
    config = ModelRegistry("config/model_registry.yaml").role(
        "Qwen3.8-27B_18GB", "coder"
    )
    adapter = SimpleNamespace(config=config)
    tool = _tool_request().tools[0]
    required = GenerationRequest(
        messages=({"role": "user", "content": "lookup"},),
        tools=(tool,),
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        parallel_tool_calls=False,
    )
    structured = GenerationRequest(
        messages=({"role": "user", "content": "fill"},),
        response_format="json",
        response_schema={"type": "object"},
    )

    required_payload = llama_server_hardware_policy._server_payload(adapter, required)
    structured_payload = llama_server_hardware_policy._server_payload(
        adapter, structured
    )

    assert required_payload["temperature"] == 0.0
    assert required_payload["tool_choice"] == "required"
    for payload in (required_payload, structured_payload):
        assert payload["chat_template_kwargs"] == {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
        assert "reasoning_effort" not in payload
