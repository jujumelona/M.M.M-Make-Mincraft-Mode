from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.design_requirement_contract import _validate_section_types
from minecraft_mod_ai.spec import SpecValidationError


def _ledger():
    return (
        {
            "requirement_id": "req_space_mode_trading_b1d7cc479a",
            "capability": "space_mode_trading",
            "authored_text": "거래로 우주선 업그레이드를 구매한다.",
            "semantic_statement": "Trade resources for spaceship upgrades.",
            "observable_behavior": {
                "given": "A player has currency.",
                "when": "The player buys an upgrade.",
                "then": "Currency is exchanged for the upgrade.",
            },
            "acceptance": ["Trading changes inventory and currency atomically."],
        },
    )


class _GameDesignModule:
    @staticmethod
    def _validate_design(value):
        assert value["modules"]
        assert value["acceptance_tests"]


class _NoModelRouter:
    def generate_text(self, *_args, **_kwargs):
        pytest.fail("Host-owned game design must not call the model")

    def generate_tool_decision(self, *_args, **_kwargs):
        pytest.fail("Host-owned game design must not call the model")


def test_unknown_requirement_id_is_rejected_in_nested_map():
    with pytest.raises(SpecValidationError, match="unknown requirement ids"):
        _validate_section_types(
            {"combat": {"encounter": ["Uses req_not_approved_123."]}},
            ("combat",),
            requirement_ids=("req_space_mode_trading_b1d7cc479a",),
        )


def test_exact_approved_requirement_id_is_accepted():
    _validate_section_types(
        {
            "progression": [
                "Use `req_space_mode_trading_b1d7cc479a` for the trade loop."
            ]
        },
        ("progression",),
        requirement_ids=("req_space_mode_trading_b1d7cc479a",),
    )


def test_host_design_binds_exact_requirement_without_model_prompt(monkeypatch):
    ledger = _ledger()
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: ledger)

    result = design.generate_sectioned_game_design(
        _GameDesignModule,
        _NoModelRouter(),
        "space trading",
        research={},
    )

    assert len(result["modules"]) == 1
    module = result["modules"][0]
    assert module["requirement_refs"] == [ledger[0]["requirement_id"]]
    assert module["capability"] == ledger[0]["capability"]
    assert ledger[0]["semantic_statement"] in module["implementation_obligations"]
    assert result["acceptance_tests"] == ledger[0]["acceptance"]
