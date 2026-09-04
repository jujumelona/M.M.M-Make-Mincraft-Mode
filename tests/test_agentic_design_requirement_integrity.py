from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.spec import SpecValidationError


class _Router:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate_text(self, role, messages, **kwargs):
        del role, messages, kwargs
        return self.output


def _ledger():
    return (
        {
            "requirement_id": "req_space_mode_trading_b1d7cc479a",
            "capability": "space_mode_trading",
            "authored_text": "거래로 우주선 업그레이드를 구매한다.",
            "semantic_statement": "Trade resources for spaceship upgrades.",
            "observable_behavior": {},
            "acceptance": ["Trading changes inventory and currency atomically."],
        },
    )


def test_unknown_requirement_id_in_progression_falls_back_field_locally(monkeypatch):
    monkeypatch.setenv("MMM_PLANNER_TRACE", "0")
    monkeypatch.setenv("MMM_PLANNER_TRACE_CONSOLE", "0")
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: _ledger())
    output = """## progression
- Currency supports `req_space_trading_protocol_001` (Inferred for context).
## combat
### encounters
- Combat remains independent of trading.
## mod_context
### authority
- Mutable economy state is server-authoritative.
"""
    section = design._generate_section(
        _Router(output),
        prompt="space trading",
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert section["progression"] == ["Trade resources for spaceship upgrades."]
    assert section["combat"] == {"encounters": ["Combat remains independent of trading."]}
    assert section["mod_context"] == {"authority": ["Mutable economy state is server-authoritative."]}


def test_unknown_requirement_id_is_rejected_in_nested_map():
    with pytest.raises(SpecValidationError, match="unknown requirement ids"):
        design._validate_section_types(
            {"combat": {"encounter": ["Uses req_not_approved_123."]}},
            ("combat",),
            requirement_ids=("req_space_mode_trading_b1d7cc479a",),
        )


def test_exact_approved_requirement_id_is_accepted():
    design._validate_section_types(
        {"progression": ["Use `req_space_mode_trading_b1d7cc479a` for the trade loop."]},
        ("progression",),
        requirement_ids=("req_space_mode_trading_b1d7cc479a",),
    )


def test_non_module_stage_prompt_does_not_receive_modules_section_contract(monkeypatch):
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: _ledger())
    messages = design._section_messages(
        prompt="space trading",
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        research={},
    )
    system = messages[0]["content"]
    assert "The modules section is the implementation-leaf index" not in system
    assert "MODULE LEAF INDEX" not in system
    assert "requirement_refs" not in system
    assert "Never invent requirement IDs" in system


def test_modules_stage_receives_leaf_index_contract(monkeypatch):
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: _ledger())
    messages = design._section_messages(
        prompt="space trading",
        section_id="modules_and_assets",
        fields=("modules", "assets"),
        research={},
    )
    system = messages[0]["content"]
    assert "MODULE LEAF INDEX" in system
    assert "requirement_refs" in system
