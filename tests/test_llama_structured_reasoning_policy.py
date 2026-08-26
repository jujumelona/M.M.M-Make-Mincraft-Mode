from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy
from minecraft_mod_ai.llama_structured_decode_policy import bind_structured_decode_policy


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


def test_base_payload_leaves_structured_constraint_to_runtime_policy() -> None:
    payload = llama_server_hardware_policy._server_payload(
        _adapter(),
        _request(
            response_format="json",
            response_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ),
    )

    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_schema_less_json_keeps_native_json_object_constraint() -> None:
    module = SimpleNamespace(_server_payload=llama_server_hardware_policy._server_payload)
    bind_structured_decode_policy(module)

    payload = module._server_payload(
        _adapter(),
        _request(response_format="json", response_schema=None),
    )

    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_explicit_schema_reaches_native_llama_payload() -> None:
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

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_structured_response",
            "strict": True,
            "schema": schema,
        },
    }


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
