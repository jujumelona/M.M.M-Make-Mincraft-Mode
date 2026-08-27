from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_hardware_policy
from minecraft_mod_ai.llama_structured_decode_policy import (
    _bind_structured_generation_retry,
    bind_structured_decode_policy,
)
from minecraft_mod_ai.structured_output import StructuredOutputValidationError


def _adapter():
    return SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192))


def _request(*, response_format: str, tools=(), response_schema=None):
    return SimpleNamespace(
        messages=(
            {"role": "system", "content": "Return the requested output."},
            {"role": "user", "content": "test"},
        ),
        response_format=response_format,
        response_schema=response_schema,
        tools=tools,
        tool_choice=None,
    )


def test_installed_payload_keeps_structured_validation_host_side() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    payload = llama_server_hardware_policy._server_payload(
        _adapter(),
        _request(response_format="json", response_schema=schema),
    )

    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload


def test_schema_less_json_keeps_validation_host_side() -> None:
    module = SimpleNamespace(_server_payload=llama_server_hardware_policy._server_payload)
    bind_structured_decode_policy(module)

    payload = module._server_payload(
        _adapter(),
        _request(response_format="json", response_schema=None),
    )

    assert "response_format" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_explicit_schema_stays_out_of_native_llama_payload() -> None:
    module = SimpleNamespace(_server_payload=llama_server_hardware_policy._server_payload)
    bind_structured_decode_policy(module)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    payload = module._server_payload(
        _adapter(),
        _request(response_format="json", response_schema=schema),
    )

    assert "response_format" not in payload


def test_native_structured_invalid_then_valid_regenerates_exactly_once() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.outputs = ['{"broken"', '{"ok":true}']
            self.calls = 0

        def generate(self, request):
            del request
            self.calls += 1
            return self.outputs.pop(0)

    module = SimpleNamespace(LlamaCppAdapter=FakeAdapter)
    _bind_structured_generation_retry(module)
    adapter = FakeAdapter()

    result = adapter.generate(_request(response_format="json"))

    assert result == '{"ok":true}'
    assert adapter.calls == 2


def test_native_structured_second_invalid_response_escapes_after_two_calls() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            del request
            self.calls += 1
            return '{"broken"'

    module = SimpleNamespace(LlamaCppAdapter=FakeAdapter)
    _bind_structured_generation_retry(module)
    adapter = FakeAdapter()

    with pytest.raises(StructuredOutputValidationError):
        adapter.generate(_request(response_format="json"))

    assert adapter.calls == 2


def test_non_structured_native_generation_is_not_retried_or_validated() -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            del request
            self.calls += 1
            return "not-json"

    module = SimpleNamespace(LlamaCppAdapter=FakeAdapter)
    _bind_structured_generation_retry(module)
    adapter = FakeAdapter()

    assert adapter.generate(_request(response_format="text")) == "not-json"
    assert adapter.calls == 1


def test_freeform_text_keeps_model_default_reasoning() -> None:
    module = SimpleNamespace(_server_payload=llama_server_hardware_policy._server_payload)
    bind_structured_decode_policy(module)
    payload = module._server_payload(
        _adapter(), _request(response_format="text")
    )

    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_native_tool_transport_keeps_tools_visible_for_pure_content_parser() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    request = _request(response_format="json", tools=(tool,))
    request.tool_choice = "auto"
    request.parallel_tool_calls = True
    module = SimpleNamespace(_server_payload=llama_server_hardware_policy._server_payload)
    bind_structured_decode_policy(module)

    payload = module._server_payload(_adapter(), request)

    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
