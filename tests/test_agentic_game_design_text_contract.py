from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.agentic_research_game_design import generate_sectioned_game_design


class _NoModelRouter:
    calls = 0

    def generate_text(self, *_args, **_kwargs):
        self.calls += 1
        pytest.fail("Deterministic game design must not call generate_text")

    def generate_tool_decision(self, *_args, **_kwargs):
        self.calls += 1
        pytest.fail("Deterministic game design must not call generate_tool_decision")


class _GameDesignModule:
    @staticmethod
    def _validate_design(value):
        assert isinstance(value["title"], str)
        assert isinstance(value["pitch"], str)
        assert isinstance(value["core_loop"], list)
        assert isinstance(value["progression"], list)
        assert isinstance(value["combat"], dict)
        assert isinstance(value["mod_context"], dict)
        assert isinstance(value["modules"], list)
        assert isinstance(value["assets"], list)
        assert isinstance(value["acceptance_tests"], list)


def _ledger():
    return (
        {
            "requirement_id": "req_trade",
            "capability": "economy.trade",
            "authored_text": "거래한다",
            "semantic_statement": "Trade resources for upgrades.",
            "observable_behavior": {
                "given": "The player has currency.",
                "when": "The player trades.",
                "then": "Items and currency are exchanged.",
            },
            "acceptance": ["Trade updates items and currency atomically."],
        },
        {
            "requirement_id": "req_colony",
            "capability": "colony.colonization",
            "authored_text": "식민지화한다",
            "semantic_statement": "Establish a colony on a planet.",
            "observable_behavior": {
                "given": "The player reaches a planet.",
                "when": "The player establishes a colony.",
                "then": "The colony exists.",
            },
            "acceptance": ["A colony can be established on the reached planet."],
        },
    )


def test_game_design_is_host_projected_with_zero_model_calls(monkeypatch):
    ledger = _ledger()
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: ledger)
    router = _NoModelRouter()

    result = generate_sectioned_game_design(
        _GameDesignModule,
        router,
        "거래하고 식민지화하는 우주 모드",
        research={"claims": ["must not rewrite authored design"]},
    )

    assert router.calls == 0
    assert result["core_loop"] == [
        "Trade resources for upgrades.",
        "Establish a colony on a planet.",
    ]
    assert result["progression"] == result["core_loop"]
    assert result["assets"] == []
    assert [module["requirement_refs"] for module in result["modules"]] == [
        ["req_trade"],
        ["req_colony"],
    ]
    assert result["acceptance_tests"] == [
        "Trade updates items and currency atomically.",
        "A colony can be established on the reached planet.",
    ]


def test_missing_optional_semantic_detail_uses_host_default_without_retry(monkeypatch):
    ledger = (
        {
            "requirement_id": "req_explore",
            "capability": "custom.semantic_explore",
            "authored_text": "행성을 탐사한다",
            "semantic_statement": "Explore a planet.",
            "observable_behavior": {},
            "acceptance": [],
        },
    )
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: ledger)
    router = _NoModelRouter()

    result = generate_sectioned_game_design(
        _GameDesignModule,
        router,
        "행성을 탐사한다",
        research={},
    )

    assert router.calls == 0
    assert result["core_loop"] == ["Explore a planet."]
    assert result["acceptance_tests"] == ["Explore a planet."]
    assert result["modules"][0]["implementation_obligations"] == ["Explore a planet."]
