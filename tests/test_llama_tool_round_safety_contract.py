from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_tool_round_safety_contract as contract
from minecraft_mod_ai import model_router


def test_default_tool_round_limit_is_finite_and_well_above_legacy_eight(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)
    limit = model_router._agent_tool_round_limit()
    assert limit >= 16
    assert limit > 8
    assert limit <= 512


def test_explicit_operator_tool_round_limit_remains_authoritative(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "17")
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "128")
    assert model_router._agent_tool_round_limit() == 17


def test_default_safety_limit_can_be_tuned_within_guardrails(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "40")
    assert model_router._agent_tool_round_limit() == 40

    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "1")
    assert model_router._agent_tool_round_limit() == 16

    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "9999")
    assert model_router._agent_tool_round_limit() == 512


def test_contract_preserves_explicit_limit_on_isolated_router() -> None:
    fake = SimpleNamespace(_agent_tool_round_limit=lambda: 23)
    contract.install(fake)
    assert fake._agent_tool_round_limit() == 23


def test_runtime_installs_finite_default_tool_round_policy() -> None:
    assert getattr(
        model_router._agent_tool_round_limit,
        "_mmm_finite_default_tool_rounds_v1",
        False,
    )
