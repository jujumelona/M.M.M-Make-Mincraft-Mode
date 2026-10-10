"""Regression: a successful GUI compile is not a playable authored mod."""

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_production import _assert_executable_gameplay_floor
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai import authored_structured_design


def _plan(*, mutations: bool) -> SimpleNamespace:
    # The test is for the production binding contract, not worksheet syntax.
    return SimpleNamespace(
        structured_sections={"_test_mutations": mutations},
        typed_plan_ir={
            "event_bindings": [{"event": "player_join", "function": "join"}],
            "functions": [{
                "id": "join",
                "body": [{"op": "state_set", "key": {"op": "literal", "value": "started"}}],
            }],
        },
    )


def _concerns(sections: dict, section: str) -> dict:
    if section == "algorithm":
        return {
            "steps": [{"operation": "fabricate_ship_component"}],
            "atomic_mutations": [{"mutations": "deduct_resources"}]
            if sections["_test_mutations"] else [],
        }
    return {}


def test_gui_only_space_gameplay_rejected_before_build(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    modules = (
        ProductionModule("ship_construction", "gui", {"name": "Ship Construction"}),
        ProductionModule("planet_travel", "gui", {"name": "Planet Travel"}),
    )
    with pytest.raises(ValueError, match="GAMEPLAY_IMPLEMENTATION_ABSENT.*GUI"):
        _assert_executable_gameplay_floor(plan, modules, {"capabilities": []})


def test_actionable_non_gui_module_is_not_treated_as_gui_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    modules = (
        ProductionModule("ship_construction", "gui", {"name": "Ship Construction"}),
        ProductionModule("ore_item", "item", {"action": "consume_ore_to_fabricate"}),
    )
    _assert_executable_gameplay_floor(plan, modules, {"capabilities": []})


def test_passive_gui_can_still_be_authored_without_gameplay_mutations(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=False)
    _assert_executable_gameplay_floor(
        plan, (ProductionModule("about_screen", "gui", {}),),
        {"capabilities": []},
    )


def test_player_join_flag_does_not_count_as_a_gameplay_loop(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_structured_design, "active_concern_records", _concerns,
    )
    plan = _plan(mutations=True)
    with pytest.raises(ValueError, match="GAMEPLAY_IMPLEMENTATION_ABSENT.*non-bootstrap"):
        _assert_executable_gameplay_floor(plan, (), {"capabilities": []})
