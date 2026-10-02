from __future__ import annotations

from minecraft_mod_ai import model_router


def test_default_tool_round_limit_is_finite(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)
    assert model_router._agent_tool_round_limit() == 64


def test_retired_default_round_environment_does_not_override_engine_cap(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "999")
    assert model_router._agent_tool_round_limit() == 64


def test_explicit_positive_tool_round_limit_is_operator_safety_cap(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "37")
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")
    assert model_router._agent_tool_round_limit() == 37


def test_invalid_or_nonpositive_explicit_limit_falls_back_to_engine_cap(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "999")
    for raw in ("garbage", "0", "-7"):
        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)
        assert model_router._agent_tool_round_limit() == 64
