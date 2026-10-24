"""Regression for real shipyard screen navigation vs server transactions.

Derived from the 2026-10-24 space-mod production failure. This must not
silently turn a menu-open operation into an unrelated credits/launch update.
"""
from copy import deepcopy

import pytest

from minecraft_mod_ai.authored_ui_action_bindings import (
    _mutating_ui_actions,
    is_presentation_only_interaction,
)
from minecraft_mod_ai.typed_action_authoring import (
    author_action_functions,
    preflight_ui_action_effects,
    promote_action_sections,
)
from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
from test_typed_ui_action_authoring import sections


SHIPYARD_OPEN = (
    "player opens the spacecraft construction screen with at least one "
    "unlocked module slot and sufficient credits or salvage, entering "
    "the shipyard in build-ready,"
)


class NoModelCalls:
    def generate_structured_data(self, *_args, **_kwargs):
        raise AssertionError("Read-only navigation must not require LLM inference")


def test_logged_shipyard_navigation_cannot_borrow_unrelated_space_mining_state():
    design = sections(SHIPYARD_OPEN)
    ui = design["resources_and_ui"]["specification"]["interactions"][0]
    ui["trigger"] = SHIPYARD_OPEN
    ui["client_feedback"] = "Show section slots and a build confirmation button"
    state = design["state_model"]["specification"]
    state["updates"][0]["trigger"] = "launch_to_space=true"
    assert is_presentation_only_interaction(ui, state)
    assert _mutating_ui_actions(design) == ()
    preflight_ui_action_effects(design, strict_source=True)
    assert promote_action_sections(NoModelCalls(), design) is design
    functions, bindings = author_action_functions(
        NoModelCalls(), "Shipyard", design, typed_host_capability_contracts(),
        first_index=0, prefer_grounded=True,
    )
    assert functions == bindings == []
    assert ui["server_action"] == SHIPYARD_OPEN


def test_screen_open_does_not_suppress_separately_owned_purchase_action():
    design = sections("Update the selected amount on the server")
    interactions = design["resources_and_ui"]["specification"]["interactions"]
    interaction = deepcopy(interactions[0])
    interaction.update(trigger=SHIPYARD_OPEN, server_action=SHIPYARD_OPEN,
                       client_feedback="Show the build confirmation button")
    interactions.append(interaction)
    preflight_ui_action_effects(design, strict_source=True)
    promoted = promote_action_sections(NoModelCalls(), design)
    rows = promoted["resources_and_ui"]["specification"]["interactions"]
    assert isinstance(rows[0]["server_action"], dict)
    assert rows[1]["server_action"] == SHIPYARD_OPEN
    assert len(_mutating_ui_actions(promoted)) == 1
    preflight_ui_action_effects(promoted, strict_source=True)
    functions, bindings = author_action_functions(
        NoModelCalls(), "Server amount update", promoted,
        typed_host_capability_contracts(), first_index=0, prefer_grounded=True,
    )
    assert len(functions) == len(bindings) == 1
    assert functions[0]["body"][0]["op"] == "state_set"


def test_purchase_hidden_in_screen_open_is_not_dropped():
    design = sections("player opens the screen and purchases a spacecraft upgrade")
    interaction = design["resources_and_ui"]["specification"]["interactions"][0]
    interaction["trigger"] = interaction["server_action"]
    design["state_model"]["specification"]["updates"][0]["trigger"] = "launch_to_space"
    assert not is_presentation_only_interaction(
        interaction, design["state_model"]["specification"]
    )
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(design, strict_source=True)


def test_navigation_with_source_owned_mutation_remains_executable():
    # Canonical state event IDs are limited to 128 characters. The UI's
    # display description may be longer but state triggers must be short.
    trigger = "player opens spacecraft construction screen"
    design = sections(trigger)
    interaction = design["resources_and_ui"]["specification"]["interactions"][0]
    interaction["trigger"] = trigger
    state = design["state_model"]["specification"]
    state["updates"][0]["trigger"] = trigger
    assert not is_presentation_only_interaction(interaction, state)
    assert _mutating_ui_actions(design)


def test_presentation_screen_does_not_steal_runtime_owner_for_empty_transition():
    from minecraft_mod_ai.request_runtime_ownership import _partition_source_owners

    design = sections(SHIPYARD_OPEN)
    interaction = design["resources_and_ui"]["specification"]["interactions"][0]
    interaction["trigger"] = SHIPYARD_OPEN
    interaction["source_requirement_id"] = "shipyard"
    state = design["state_model"]["specification"]
    state["updates"][0]["trigger"] = "mine_resource"
    empty_transition = {
        "trigger": SHIPYARD_OPEN,
        "mutation": [],
        "source_requirement_id": "shipyard",
    }
    state["transitions"] = [empty_transition]
    assigned, remaining = _partition_source_owners(
        [("state_model.transitions[0]", empty_transition)],
        [{"requirement_id": "shipyard"}], design,
    )
    assert assigned == [("state_model.transitions[0]", "shipyard")]
    assert remaining == []


def test_logged_completed_section_list_is_read_only_not_a_second_build():
    design = sections("player opens the completed-section list and inspects the build result")
    ui = design["resources_and_ui"]["specification"]["interactions"][0]
    ui["trigger"] = ui["server_action"]
    design["state_model"]["specification"]["updates"][0]["trigger"] = "craft_section"
    assert is_presentation_only_interaction(
        ui, design["state_model"]["specification"]
    )
    preflight_ui_action_effects(design, strict_source=True)
    assert _mutating_ui_actions(design) == ()
