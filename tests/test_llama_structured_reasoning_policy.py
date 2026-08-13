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


def test_structured_json_reserves_budget_for_visible_contract() -> None:
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

    assert payload["response_format"]["type"] == "json_object"
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_schema_less_json_uses_host_validated_fast_decode(monkeypatch) -> None:
    # Bind against a tiny isolated module-shaped object so this test proves the
    # policy itself and does not depend on package bootstrap import order.
    module = SimpleNamespace(_server_payload=llama_server_hardware_policy._server_payload)
    bind_structured_decode_policy(module)

    payload = module._server_payload(
        _adapter(),
        _request(response_format="json", response_schema=None),
    )

    assert "response_format" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_freeform_text_keeps_model_default_reasoning() -> None:
    payload = llama_server_hardware_policy._server_payload(
        _adapter(), _request(response_format="text")
    )

    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_tool_turn_keeps_minimal_function_calling_transport() -> None:
    payload = llama_server_hardware_policy._server_payload(
        _adapter(),
        _request(
            response_format="json",
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ),
        ),
    )

    assert "tools" in payload
    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload
