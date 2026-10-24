"""Bounded feature repair must never borrow an effect or change approved UI."""
from copy import deepcopy
import pytest

from minecraft_mod_ai import authored_structured_design as design
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.request_structured_design import repair_request_action_effects
from minecraft_mod_ai.typed_action_authoring import preflight_ui_action_effects
from test_typed_ui_action_authoring import sections


def fixture():
    src = sections("Purchase one ship component")
    ui = src["resources_and_ui"]["specification"]["interactions"][0]
    ui.update(trigger="buy_ship_component", source_requirement_id="ship")
    state = src["state_model"]["specification"]
    old = state["updates"][0]
    old.update(trigger="mine_ore", source_requirement_id="ore")
    src["behavior_contract"] = {
        "constraint_evidence_refs": [],
        "specification": {
            **{key: [] for key in DETAIL_RECORDS["behavior_contract"]},
            "inapplicable_concerns": [],
        },
    }
    return src


def scope():
    return {"requested_prompt": "Mine ore and purchase a ship component", "requirements": [
        {"requirement_id": "ore", "statement": "Mine ore", "source_span": {"text": "Mine ore"}},
        {"requirement_id": "ship", "statement": "Purchase ship component",
         "source_span": {"text": "purchase a ship component"}},
    ]}


def candidate(trigger="buy_ship_component"):
    spec = {key: [] for key in DETAIL_RECORDS["state_model"]}
    spec["variables"] = []
    spec["updates"] = [{"owner": "player", "trigger": trigger, "mutation": [{
        "target": "amount", "operator": "+=", "value": {
            "kind": "number", "value": "1",
        },
    }]}]
    spec["inapplicable_concerns"] = []
    return {"state_model": {"constraint_evidence_refs": [], "specification": spec}}


def test_repair_only_regenerates_state_and_keeps_unrelated_actions(monkeypatch):
    source = fixture()
    original = deepcopy(source)
    calls = []

    def replace_state(router, prompt, **kwargs):
        calls.append((prompt, kwargs))
        assert kwargs["required_action_count"] == 1
        assert kwargs["shared_state_variables"][0]["name"] == "amount"
        assert "buy_ship_component" in prompt
        return candidate()

    monkeypatch.setattr(design, "author_state_repair_section", replace_state)
    result = repair_request_action_effects(None, source, scope())
    assert len(calls) == 1
    assert result["resources_and_ui"] == original["resources_and_ui"]
    updates = result["state_model"]["specification"]["updates"]
    assert [row["trigger"] for row in updates] == [
        "mine_ore", "buy_ship_component",
    ]
    assert updates[0] == original["state_model"]["specification"]["updates"][0]
    assert updates[1]["source_requirement_id"] == "ship"
    preflight_ui_action_effects(result, strict_source=True)
    assert source == original


def test_repair_rejects_different_trigger_instead_of_forging_success(monkeypatch):
    monkeypatch.setattr(
        design, "author_state_repair_section",
        lambda *args, **kwargs: candidate("launch_to_space"),
    )
    with pytest.raises(ValueError, match="ACTION_SOURCE_REPAIR_EXHAUSTED"):
        repair_request_action_effects(None, fixture(), scope())


def test_unowned_action_cannot_be_repaired_using_adjacent_requirements(monkeypatch):
    source = fixture()
    source["resources_and_ui"]["specification"]["interactions"][0].pop(
        "source_requirement_id"
    )
    with pytest.raises(ValueError, match="ACTION_REPAIR_SOURCE_UNKNOWN"):
        repair_request_action_effects(None, source, scope())


def test_valid_purchase_requires_no_additional_model_generation(monkeypatch):
    source = fixture()
    source["state_model"]["specification"]["updates"][0].update(
        trigger="buy_ship_component", source_requirement_id="ship",
    )
    def forbidden(*args, **kwargs):
        raise AssertionError("already grounded actions must not be regenerated")
    monkeypatch.setattr(design, "author_state_repair_section", forbidden)
    result = repair_request_action_effects(None, source, scope())
    assert result == source


@pytest.mark.parametrize("operator,value", [
    ("+=", "0"), ("-=", "0"), ("*=", "1"), ("/=", "1"),
])
def test_noop_cannot_count_as_purchase_effect(operator, value):
    source = fixture()
    state = source["state_model"]["specification"]
    state["updates"][0].update(
        trigger="buy_ship_component", source_requirement_id="ship",
    )
    state["updates"][0]["mutation"][0].update(
        operator=operator,
        value={"kind": "number", "value": value},
    )
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(source, strict_source=True)


def test_unrelated_meaningful_write_cannot_hide_same_target_noop():
    source = fixture()
    state = source["state_model"]["specification"]
    state["updates"].append({
        "owner": "player", "trigger": "buy_ship_component",
        "source_requirement_id": "ship",
        "mutation": [{"target": "amount", "operator": "+=",
                      "value": {"kind": "number", "value": "0"}}],
    })
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(source, strict_source=True)
