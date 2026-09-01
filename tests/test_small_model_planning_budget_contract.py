from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import generation_output_budget as budget


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


def test_retrieval_batch_uses_existing_bounded_tool_page_budget(monkeypatch) -> None:
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_new_tokens=8192,
        extra={"dynamic_output_budget": True},
    )
    monkeypatch.setattr(budget, "effective_context_tokens", lambda _config: 32768)
    monkeypatch.setattr(budget, "tool_action_token_budget", lambda _config: 8192)

    result = budget.generation_output_token_budget(
        config,
        input_tokens=600,
        tools=(_tool("plan_requirement_retrieval"),),
    )

    assert result == 8192
    assert result < 32768 - 600 - 2048


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
