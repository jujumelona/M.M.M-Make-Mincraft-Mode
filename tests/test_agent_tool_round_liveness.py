from __future__ import annotations

from minecraft_mod_ai.model_router import _agent_tool_round_limit


def test_agent_tool_round_limit_uses_finite_default_without_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)

    assert _agent_tool_round_limit() == 128


def test_agent_tool_round_limit_honors_positive_override_and_fails_safe_to_default(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "17")
    assert _agent_tool_round_limit() == 17

    for raw in ("", "0", "-1", "not-an-integer"):
        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)
        assert _agent_tool_round_limit() == 128
