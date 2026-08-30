from __future__ import annotations

from minecraft_mod_ai import model_router


def test_default_tool_round_limit_is_finite(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)
    assert model_router._agent_tool_round_limit() == 128


def test_default_tool_round_limit_is_tunable_with_safe_bounds(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")
    assert model_router._agent_tool_round_limit() == 64
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "1")
    assert model_router._agent_tool_round_limit() == 16
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "9999")
    assert model_router._agent_tool_round_limit() == 512


def test_explicit_positive_tool_round_limit_wins(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "37")
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")
    assert model_router._agent_tool_round_limit() == 37


def test_invalid_or_nonpositive_explicit_limit_stays_finite(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")
    for raw in ("garbage", "0", "-7"):
        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)
        assert model_router._agent_tool_round_limit() == 64
