"""A passive prerequisite cannot hide the player's subsequent input control."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from minecraft_mod_ai.host_action_occurrence import (
    author_host_owned_player_occurrence,
    select_original_player_action,
)
from minecraft_mod_ai.player_action_modality import assess_player_action_modality
from minecraft_mod_ai.source_occurrence_error import SourceHookUnavailable
from minecraft_mod_ai.source_owned_ui_event import (
    materialize_typed_ui_occurrence,
    read_verified_typed_ui_events,
)

CAPTURED = {
    "behavior_id": "behavior_51abfccc946fe0413520",
    "source_quote": "식민지화등 여러가지가 가능한 모드",
    "intent": "Colonize worlds and unlock multiple gameplay possibilities.",
    "trigger": "Player reaches a planet or space location and initiates colonization.",
    "effects": "Colonization becomes available, enabling multiple gameplay possibilities.",
    "runtime_state_reads": ["Player location", "Available planets", "Colonization prerequisites"],
}
SYMBOLS = {name: {"kind": "CLASS", "owner": name} for name in (
    "net.minecraft.commands.CommandSourceStack", "net.fabricmc.fabric.api.event.Event",
    "net.fabricmc.fabric.api.event.EventFactory",
    "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback",
)}


@pytest.mark.parametrize("trigger,anchor", [
    (CAPTURED["trigger"], "initiates"),
    ("Player has the required supplies and confirms the selected building", "confirms"),
    ("Player reaches a station and submits a repair request", "submits"),
    ("Player enters a workshop, then requests a custom assembly", "requests"),
    ("Player arrives at the counter and explicitly chooses an offer", "chooses"),
])
def test_host_owns_the_stated_control_after_a_passive_prerequisite(trigger, anchor):
    behavior = {**CAPTURED, "trigger": trigger}
    original = deepcopy(behavior)
    chosen = select_original_player_action(behavior)
    assert chosen is not None
    assert chosen["request_field"] == "trigger"
    assert chosen["request_quote"] in trigger
    assert chosen["activation_scope"]["action_anchor"] == anchor
    assert behavior == original


@pytest.mark.parametrize("trigger", [
    "Player reaches a station and the server initiates processing",
    "Player has materials and an operator confirms the request",
    "Player reaches a planet and it automatically initiates colonization",
    "Player reaches a planet and initiates colonization automatically",
    "Player reaches a planet and does not initiate colonization",
    "Player reaches a planet without having to initiate colonization",
    "Player has enough resources and wants to initiate colonization",
    "Player reaches a planet and the system initiates colonization and confirms it",
    "Player reaches a station and Alice initiates service and confirms it",
    "Player reaches a station and we initiate service and confirm it",
    "Player reaches a station and a caretaker initiates service and confirms it",
])
def test_passive_automatic_or_differently_owned_occurrences_do_not_create_an_input(trigger):
    assert select_original_player_action({**CAPTURED, "trigger": trigger}) is None


@pytest.mark.parametrize("field,restriction", [
    ("intent", "Player must not initiate colonization"),
    ("effects", "Player initiates colonization automatically"),
    ("source_quote", "The player initiates colonization without player input"),
])
def test_compound_control_keeps_the_selected_verbs_modality_in_all_original_fields(field, restriction):
    assert select_original_player_action({**CAPTURED, field: restriction}) is None


def test_automatic_arrival_does_not_erase_the_independent_confirm_control():
    behavior = {**CAPTURED,
                "trigger": "Player reaches the destination automatically and confirms the allocation"}
    chosen = select_original_player_action(behavior)
    assert chosen is not None
    assert chosen["activation_scope"]["action_anchor"] == "confirms"


def test_direct_action_keeps_its_own_anchor_when_a_later_control_is_mentioned():
    behavior = {**CAPTURED, "trigger": "Player purchases supplies and confirms the receipt"}
    assert assess_player_action_modality(behavior, behavior["trigger"])["action_anchor"] == "purchases"


def test_without_input_on_an_independent_server_consequence_keeps_the_manual_control():
    behavior = {**CAPTURED,
                "effects": "The server records the result without player input"}
    chosen = select_original_player_action(behavior)
    assert chosen is not None
    assert chosen["activation_scope"]["action_anchor"] == "initiates"


def test_without_player_input_only_restricts_the_preceding_arrival_predicate():
    behavior = {**CAPTURED,
                "trigger": "Player reaches the destination without player input and explicitly confirms the allocation"}
    chosen = select_original_player_action(behavior)
    assert chosen is not None
    assert chosen["activation_scope"]["action_anchor"] == "confirms"


def test_original_player_ownership_survives_multiple_eligibility_predicates():
    behavior = {**CAPTURED,
                "trigger": "Player reaches the destination and has enough supplies and confirms the allocation"}
    chosen = select_original_player_action(behavior)
    assert chosen is not None
    assert chosen["activation_scope"]["action_anchor"] == "confirms"


def test_valid_producer_from_previous_implementation_replays_with_identical_identity_and_files(tmp_path):
    from minecraft_mod_ai.source_owned_ui_event import _render_recipe

    # Captured with the actual 4d6f66e assessor/host author, not the current
    # implementation. Both the digest and every source byte hash are pinned.
    fixture = json.loads((Path(__file__).parent / "fixtures/compound_player_legacy_v1.json").read_text())
    recipe = fixture["recipe"]
    planned = _render_recipe(recipe, SYMBOLS)
    assert planned["digest"] == fixture["digest"]
    assert {name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in planned["files"].items()} == fixture["files_sha256"]
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"legacy_test"}')
    materialize_typed_ui_occurrence(planned, project_root=tmp_path, symbols=SYMBOLS)
    assert read_verified_typed_ui_events(
        project_root=tmp_path, symbols=SYMBOLS, behaviors=[recipe["behavior"]],
        package="example.generated") == [planned["callback"]]


def test_legacy_scope_compatibility_cannot_bypass_current_restrictions(tmp_path):
    from minecraft_mod_ai.source_owned_ui_event import _render_recipe

    fixture = json.loads((Path(__file__).parent / "fixtures/compound_player_legacy_v1.json").read_text())
    with pytest.raises(SourceHookUnavailable, match="original_occurrence_is_not_manual_activation"):
        _render_recipe(fixture["rejected_recipe"], SYMBOLS)
    assert not (tmp_path / "src").exists()


def test_captured_missing_hook_becomes_a_replayable_producer_without_a_model_ownership_verdict(tmp_path):
    class DecliningMiMo:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            # Deliberately cannot plan events, source sites, activation or Java.
            return {"ui_action_label": {"label": "Start requested action"},
                    "ui_action_input_count": {"count": 0}}[data_name]

    planned = author_host_owned_player_occurrence(
        DecliningMiMo(), CAPTURED, symbols=SYMBOLS, package="example.generated",
        mod_id="compound_test", minecraft_version="26.2")
    assert planned is not None
    assert planned["callback"]["parameters"] == ["net.minecraft.commands.CommandSourceStack"]
    assert planned["recipe"]["behavior"] == CAPTURED
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"compound_test"}')
    materialize_typed_ui_occurrence(planned, project_root=tmp_path, symbols=SYMBOLS)
    installed = read_verified_typed_ui_events(
        project_root=tmp_path, symbols=SYMBOLS, behaviors=[CAPTURED], package="example.generated")
    assert installed == [planned["callback"]]
    assert planned["runtime_verified"] is False
