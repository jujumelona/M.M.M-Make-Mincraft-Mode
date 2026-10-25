"""An eligibility condition must not erase the player's requested action."""
from copy import deepcopy
import json

import pytest

from minecraft_mod_ai.source_owned_hook import SourceHookUnavailable
from minecraft_mod_ai.source_owned_ui_event import (
    author_typed_ui_occurrence,
    materialize_typed_ui_occurrence,
    plan_typed_ui_occurrence,
    read_verified_typed_ui_events,
)


# Exact behavior from the uploaded MiMo completion immediately before failure.
TRADE = {
    "behavior_id": "behavior_7e44c67dc12bf6885bb9",
    "source_quote": "거래 등으로",
    "intent": "Trade and purchase goods to fund spacecraft construction and upgrades",
    "trigger": "Player has accumulated money and wants to build or upgrade a spacecraft",
    "effects": "Player trades mined resources for goods, then purchases spacecraft construction materials and upgrades",
    "runtime_state_reads": ["available goods", "spacecraft build requirements", "spacecraft upgrade requirements"],
    "runtime_state_writes": ["player inventory"],
}
SYMBOLS = {name: {"kind": "CLASS", "owner": name} for name in (
    "net.minecraft.commands.CommandSourceStack", "net.fabricmc.fabric.api.event.Event",
    "net.fabricmc.fabric.api.event.EventFactory",
    "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback",
)}


class ActionPlanner:
    """Supply only external model data; rendering and installation stay real."""

    def __init__(self, field="effects", quote=None, *, activation="typed_ui_action"):
        self.field = field
        self.quote = TRADE[field] if quote is None else quote
        self.activation = activation

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        if data_name == "source_owned_player_action_evidence":
            return {"activation": self.activation, "request_field": self.field,
                    "request_quote": self.quote}
        return {
            "ui_action_label": {"label": "Requested action"},
            "ui_action_input_count": {"count": 0},
        }[data_name]


def _author(behavior, router):
    return author_typed_ui_occurrence(
        router, behavior, symbols=SYMBOLS, package="example.generated",
        mod_id="occurrence_test", minecraft_version="26.2",
    )


def test_production_precondition_keeps_trade_action_and_materializes_replayable_server_event(tmp_path):
    original = deepcopy(TRADE)
    planned = _author(TRADE, ActionPlanner())
    assert TRADE == original
    assert planned["recipe"]["behavior"]["trigger"] == original["trigger"]
    assert planned["recipe"]["activation_evidence"]["request_field"] == "effects"
    assert planned["callback"]["parameters"] == ["net.minecraft.commands.CommandSourceStack"]
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"occurrence_test"}')
    materialize_typed_ui_occurrence(planned, project_root=tmp_path, symbols=SYMBOLS)
    actual = read_verified_typed_ui_events(project_root=tmp_path, symbols=SYMBOLS,
                                         behaviors=[TRADE], package="example.generated")
    assert actual == [planned["callback"]]
    metadata = json.loads(manifest.read_text())
    assert metadata["entrypoints"]["main"] == [planned["entrypoints"]["main"]]
    assert metadata["entrypoints"]["client"] == [planned["entrypoints"]["client"]]


@pytest.mark.parametrize("trigger", [
    "Player purchases supplies", "플레이어가 필요한 물품을 구매한다",
])
def test_action_planning_does_not_depend_on_an_english_activation_verb_list(trigger):
    behavior = {**TRADE, "trigger": trigger}
    planned = _author(behavior, ActionPlanner("trigger", trigger))
    assert planned["recipe"]["behavior"] == behavior
    assert planned["recipe"]["activation_evidence"]["request_quote"] == trigger


def test_action_evidence_is_bound_to_the_entire_original_behavior():
    planned = _author(TRADE, ActionPlanner())
    changed = {**TRADE, "trigger": "Player receives supplies automatically"}
    with pytest.raises(ValueError, match="ACTION_EVIDENCE_BEHAVIOR_MISMATCH"):
        plan_typed_ui_occurrence(
            behavior=changed, action_metadata=planned["recipe"]["action_metadata"],
            activation_evidence=planned["recipe"]["activation_evidence"],
            package="example.generated", mod_id="occurrence_test",
            minecraft_version="26.2", symbols=SYMBOLS,
        )


@pytest.mark.parametrize("field,quote", [
    ("effects", "Player presses a new shortcut"),
    ("trigger", "Player trades mined resources for goods"),
])
def test_planner_cannot_invent_or_misattribute_an_action_quote(field, quote):
    with pytest.raises(SourceHookUnavailable, match="action_evidence_quote_not_in_original_field"):
        _author(TRADE, ActionPlanner(field, quote))


@pytest.mark.parametrize("trigger", [
    "Player enters a region and the server selects a target automatically",
    "Player reaches the destination without having to click or confirm anything",
    "Player does not select any target",
    "Player selects a target automatically",
])
def test_exact_quote_does_not_override_explicit_automatic_or_negated_activation(trigger):
    action = "Player confirms arrival" if "click or confirm" in trigger else "Player selects a target"
    with pytest.raises(SourceHookUnavailable):
        _author({**TRADE, "trigger": trigger, "effects": action}, ActionPlanner("effects", action))


def test_planner_can_leave_a_passive_occurrence_for_its_runtime_source_owner():
    behavior = {**TRADE, "trigger": "Player reaches a new world",
                "effects": "The destination is recorded automatically"}
    with pytest.raises(SourceHookUnavailable, match="original_trigger_has_no_owned_ui_activation"):
        _author(behavior, ActionPlanner("trigger", "Player reaches a new world", activation="unavailable"))


def test_modifying_saved_action_evidence_does_not_admit_a_different_producer(tmp_path):
    planned = _author(TRADE, ActionPlanner())
    planned["recipe"]["activation_evidence"]["request_quote"] = "Player presses a shortcut"
    with pytest.raises((SourceHookUnavailable, ValueError)):
        materialize_typed_ui_occurrence(planned, project_root=tmp_path, symbols=SYMBOLS)
    assert not (tmp_path / "src").exists()


@pytest.mark.parametrize("changes", [
    {"effects": "Player purchases supplies and the server automatically subtracts the price"},
    {"trigger": "Player has enough credits automatically accumulated over time"},
    {"effects": "Player purchases supplies and the purchase price is automatically deducted"},
])
def test_automatic_prerequisites_and_consequences_do_not_erase_voluntary_activation(changes):
    behavior = {**TRADE, "trigger": "Player has enough credits", "effects": "Player purchases supplies", **changes}
    planned = _author(behavior, ActionPlanner("effects", "Player purchases supplies"))
    assert planned["recipe"]["behavior"] == behavior
    assert planned["recipe"]["activation_evidence"]["activation_scope"]["clauses"]


@pytest.mark.parametrize("field,restriction", [
    ("source_quote", "The player purchases supplies automatically without player input"),
    ("trigger", "Player must not purchase supplies"),
    ("intent", "The player must never purchase supplies"),
    ("source_quote", "No player input is required; supplies are delivered automatically"),
])
def test_another_original_field_cannot_hide_automatic_or_prohibited_activation(field, restriction):
    behavior = {**TRADE, "trigger": "Player has enough credits", "effects": "Player purchases supplies",
                field: restriction}
    with pytest.raises(SourceHookUnavailable, match="original_occurrence_is_not_manual_activation"):
        _author(behavior, ActionPlanner("effects", "Player purchases supplies"))


def test_negated_eligibility_does_not_forbid_the_requested_action():
    behavior = {**TRADE, "trigger": "Player does not have a supply kit",
                "effects": "Player purchases supplies"}
    planned = _author(behavior, ActionPlanner("effects", "Player purchases supplies"))
    assert planned["recipe"]["behavior"]["trigger"] == "Player does not have a supply kit"


def test_replay_rejects_changed_activation_scope_before_writing_any_files(tmp_path):
    behavior = {**TRADE, "effects": "Player purchases supplies and the server automatically subtracts the price"}
    planned = _author(behavior, ActionPlanner("effects", "Player purchases supplies"))
    planned["recipe"]["activation_evidence"]["activation_scope"]["clauses"] = []
    with pytest.raises(ValueError, match="ACTION_EVIDENCE_SCOPE_MISMATCH"):
        materialize_typed_ui_occurrence(planned, project_root=tmp_path, symbols=SYMBOLS)
    assert not (tmp_path / "src").exists()


@pytest.mark.parametrize("action,restricted", [
    ("Player applies the selected policy", "Player must not apply the selected policy"),
    ("Player supplies materials", "Player must not supply materials"),
    ("Player replies to the request", "Player must not reply to the request"),
])
def test_action_verbs_ending_in_ly_are_not_mistaken_for_modifiers(action, restricted):
    behavior = {**TRADE, "trigger": restricted, "effects": action}
    with pytest.raises(SourceHookUnavailable, match="original_occurrence_is_not_manual_activation"):
        _author(behavior, ActionPlanner("effects", action))


@pytest.mark.parametrize("restriction", ["Player cannot purchase supplies", "Player doesn't purchase supplies"])
def test_a_quoted_prohibition_cannot_claim_to_be_a_voluntary_action(restriction):
    behavior = {**TRADE, "effects": restriction}
    with pytest.raises(SourceHookUnavailable, match="original_occurrence_is_not_manual_activation"):
        _author(behavior, ActionPlanner("effects", restriction))


def test_model_abstention_cannot_veto_an_explicit_player_trade():
    # Regression: 2026-10-25 production trace; the local 9B model returned
    # {"activation":"unavailable"} even though effects describe a trade.
    planned = _author(TRADE, ActionPlanner(activation="unavailable"))
    assert planned["recipe"]["activation_evidence"]["request_field"] == "effects"
    assert planned["recipe"]["activation_evidence"]["request_quote"] == TRADE["effects"]
    assert planned["recipe"]["behavior"] == TRADE
    assert planned["callback"]["provenance"] == "host_typed_ui_action"


@pytest.mark.parametrize("intent,effects", [
    ("Commission a settlement building", "Player commissions a settlement building"),
    ("Apply the selected policy", "Player applies the selected policy"),
    ("Supply the settlement", "Player supplies the settlement"),
])
def test_host_recovers_other_original_actions_without_an_action_verb_allowlist(intent, effects):
    behavior = {**TRADE, "intent": intent, "effects": effects}
    planned = _author(behavior, ActionPlanner(activation="unavailable"))
    evidence = planned["recipe"]["activation_evidence"]
    assert evidence["request_field"] == "effects"
    assert evidence["request_quote"] == effects


@pytest.mark.parametrize("intent,effects", [
    ("Enter another planet", "Player enters another planet"),
    ("Reach a new dimension", "Player reaches a new dimension"),
    ("Receive supplies", "Player receives supplies automatically"),
    ("Purchase supplies", "Player purchases supplies automatically"),
    ("Purchase supplies", "Player does not purchase supplies"),
])
def test_host_never_fabricates_manual_action_for_passive_or_prohibited_events(intent, effects):
    behavior = {**TRADE, "intent": intent, "effects": effects}
    with pytest.raises(SourceHookUnavailable):
        _author(behavior, ActionPlanner(activation="unavailable"))
