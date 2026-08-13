from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload


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
    payload = _server_payload(
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


def test_freeform_text_keeps_model_default_reasoning() -> None:
    payload = _server_payload(_adapter(), _request(response_format="text"))

    assert "response_format" not in payload
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_tool_turn_keeps_minimal_function_calling_transport() -> None:
    payload = _server_payload(
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
