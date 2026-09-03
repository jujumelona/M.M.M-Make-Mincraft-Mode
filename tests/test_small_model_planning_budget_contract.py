from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import generation_output_budget as budget
from minecraft_mod_ai import llama_generation_budget as llama_budget
from minecraft_mod_ai import semantic_requirement_authority as semantic


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"requirements": {"type": "array"}},
                "required": ["requirements"],
            },
        },
    }


def _fallback_request(schema: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        tools=(),
        response_format="json",
        response_schema=schema,
    )


def _installed_fake_hardware(monkeypatch):
    hardware = SimpleNamespace()

    def raw_server_payload(_adapter, _request):
        return {"max_tokens": 30000}

    hardware._server_payload = raw_server_payload
    monkeypatch.setattr(
        llama_budget,
        "apply_generation_budget",
        lambda payload, *, config: {**payload, "max_tokens": 30000},
    )
    monkeypatch.setattr(
        llama_budget,
        "tool_action_token_budget",
        lambda _config: 8192,
    )
    llama_budget.install(hardware)
    return hardware


def test_semantic_batch_uses_existing_bounded_tool_page_budget(monkeypatch) -> None:
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_new_tokens=8192,
        extra={"dynamic_output_budget": True},
    )
    monkeypatch.setattr(budget, "effective_context_tokens", lambda _config: 32768)
    monkeypatch.setattr(budget, "tool_action_token_budget", lambda _config: 8192)

    result = budget.generation_output_token_budget(
        config,
        input_tokens=400,
        tools=(_tool("compile_semantic_requirements"),),
    )

    assert result == 8192
    assert result < 32768 - 400 - 2048


def test_unknown_side_effect_tool_does_not_get_planner_exception(monkeypatch) -> None:
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_new_tokens=8192,
        extra={"dynamic_output_budget": True},
    )
    monkeypatch.setattr(budget, "effective_context_tokens", lambda _config: 32768)
    monkeypatch.setattr(budget, "tool_action_token_budget", lambda _config: 8192)

    result = budget.generation_output_token_budget(
        config,
        input_tokens=400,
        tools=(_tool("unreviewed_unknown_action"),),
    )

    assert result == 32768 - 400 - 2048


def test_semantic_json_fallback_cannot_regress_to_30k(monkeypatch) -> None:
    hardware = _installed_fake_hardware(monkeypatch)
    adapter = SimpleNamespace(config=SimpleNamespace())
    request = _fallback_request(semantic._semantic_schema(3))

    payload = hardware._server_payload(adapter, request)

    assert payload["max_tokens"] == 8192


def test_unrelated_json_schema_keeps_general_dynamic_budget(monkeypatch) -> None:
    hardware = _installed_fake_hardware(monkeypatch)
    adapter = SimpleNamespace(config=SimpleNamespace())
    request = _fallback_request(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    payload = hardware._server_payload(adapter, request)

    assert payload["max_tokens"] == 30000
