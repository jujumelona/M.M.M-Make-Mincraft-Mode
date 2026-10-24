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
    with pytest.raises(ValueError, match="(ACTION_UI_EFFECT_UNBOUND|ACTION_DESIGN_NO_EXECUTABLE_EFFECT)"):
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


def test_multiple_actions_in_one_feature_repair_individually(monkeypatch):
    source = fixture()
    ui = source["resources_and_ui"]["specification"]["interactions"]
    for index in range(5):
        extra = deepcopy(ui[0])
        extra.update(trigger=f"upgrade_component_{index}",
                     server_action=f"Upgrade component {index}")
        ui.append(extra)
    before = deepcopy(source)
    calls = []

    def one_action_only(router, prompt, **kwargs):
        assert kwargs["required_action_count"] == 1
        import re
        action = re.search(r'"trigger": "(buy_ship_component|upgrade_component_[0-9]+)"', prompt)
        assert action is not None
        calls.append(action.group(1))
        return candidate(action.group(1))

    monkeypatch.setattr(design, "author_state_repair_section", one_action_only)
    repaired = repair_request_action_effects(None, source, scope())
    assert len(calls) == 6
    assert repaired["resources_and_ui"] == before["resources_and_ui"]
    assert repaired["state_model"]["specification"]["updates"][0] == (
        before["state_model"]["specification"]["updates"][0]
    )
    preflight_ui_action_effects(repaired, strict_source=True)


def test_repair_preserves_valid_same_feature_action(monkeypatch):
    source = fixture()
    src = source["resources_and_ui"]["specification"]["interactions"]
    already_valid = deepcopy(src[0])
    already_valid["trigger"] = "install_weapon"
    src.append(already_valid)
    source["state_model"]["specification"]["updates"].append({
        "owner": "player", "trigger": "install_weapon",
        "source_requirement_id": "ship",
        "mutation": [{"target": "amount", "operator": "+=",
                      "value": {"kind": "number", "value": "1"}}],
    })
    original_valid = deepcopy(source["state_model"]["specification"]["updates"][-1])
    monkeypatch.setattr(
        design, "author_state_repair_section",
        lambda *args, **kwargs: candidate(),
    )
    repaired = repair_request_action_effects(None, source, scope())
    assert original_valid in repaired["state_model"]["specification"]["updates"]
    preflight_ui_action_effects(repaired, strict_source=True)


def test_long_ui_description_uses_bounded_host_event_id(monkeypatch):
    from minecraft_mod_ai.request_record_ownership import (
        same_action_source, source_trigger_identity,
    )
    source = fixture()
    ui = source["resources_and_ui"]["specification"]["interactions"][0]
    ui["trigger"] = (
        "player opens the spacecraft construction screen with at least one "
        "unlocked module slot and sufficient credits, and purchases a modular "
        "engine component from the shipyard for the currently owned spacecraft"
    )
    assert len(ui["trigger"]) > 128
    canonical = source_trigger_identity(ui)
    assert canonical.startswith("ui_evt_") and len(canonical) <= 128
    assert source_trigger_identity(ui) == canonical
    assert not same_action_source(
        {"trigger": canonical, "source_requirement_id": "other"}, ui
    )
    calls = []

    def bounded(router, prompt, **kwargs):
        assert kwargs["fixed_state_trigger"] == canonical
        assert kwargs["required_action_count"] == 1
        calls.append(prompt)
        return candidate(canonical)

    monkeypatch.setattr(design, "author_state_repair_section", bounded)
    result = repair_request_action_effects(None, source, scope())
    update = result["state_model"]["specification"]["updates"][-1]
    assert update["trigger"] == canonical
    assert same_action_source(update, ui)
    assert result["resources_and_ui"]["specification"]["interactions"][0]["trigger"] == ui["trigger"]
    assert len(calls) == 1
    preflight_ui_action_effects(result, strict_source=True)


def test_repair_schema_binds_host_event_before_model_decoding(monkeypatch):
    from minecraft_mod_ai import authored_structured_design as impl

    fixed = "ui_evt_f1ea5d9ef6a7b8c9d0123456"
    calls = []

    def fake_template(section, concerns, *, record_counts, state_symbols):
        assert section == "state_model" and concerns == ("updates",)
        return {"properties": {"updates": {
            "items": {"properties": {"trigger": {"type": "string", "maxLength": 128}}}
        }}}

    def fake_generator(router, prompt, messages, page, schema, state_symbols,
                       *, section_context, on_additional_call):
        calls.append(schema["properties"]["updates"]["items"]["properties"]["trigger"])
        return {"updates": [{"trigger": fixed}]}

    monkeypatch.setattr(
        "minecraft_mod_ai.worksheet_atomic_chunker.worksheet_chunk_schema",
        fake_template,
    )
    monkeypatch.setattr(impl, "_generate_authored_page_value", fake_generator)
    page = impl._PlannerPageRequest(
        section="state_model", chunk_index=0, chunk_count=1,
        concerns=("updates",), completed={}, media_paths=(),
        fixed_state_trigger=fixed,
    )
    result = impl._generate_authored_chunk(
        None, "repair", page=page, record_counts={"updates": 1},
    )
    assert result["updates"][0]["trigger"] == fixed
    assert calls == [{"type": "string", "enum": [fixed]}]
