from __future__ import annotations

from minecraft_mod_ai.model_router import _agent_tool_round_limit


def test_agent_tool_round_limit_is_unbounded_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)

    assert _agent_tool_round_limit() is None


def test_agent_tool_round_limit_honors_only_positive_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "17")
    assert _agent_tool_round_limit() == 17

    for raw in ("", "0", "-1", "not-an-integer"):
        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)
        assert _agent_tool_round_limit() is None
