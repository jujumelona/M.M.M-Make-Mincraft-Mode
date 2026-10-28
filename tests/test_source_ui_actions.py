"""Source-owned UI operations defer without inventing a state transaction."""
from copy import deepcopy

import pytest
from test_typed_ui_action_authoring import sections

from minecraft_mod_ai.authored_ui_action_bindings import action_record_identity
from minecraft_mod_ai.source_ui_actions import (
    SOURCE_UI_CAPABILITY,
    deferred_ui_interactions,
    source_ui_body,
    source_ui_callback,
    source_ui_capabilities,
    source_ui_preflight_sections,
)


def source_sections(*, effect=False, prose=False):
    result = sections("Apply the requested operation" if prose else None)
    row = result["resources_and_ui"]["specification"]["interactions"][0]
    row["source_requirement_id"] = "feature_a"
    update = result["state_model"]["specification"]["updates"][0]
    update["source_requirement_id"] = "feature_a" if effect else "feature_b"
    return result


@pytest.mark.parametrize("prose", [False, True])
def test_deferred_action_cannot_borrow_another_source_mutation(prose):
    source = source_sections(prose=prose)
    row = source["resources_and_ui"]["specification"]["interactions"][0]
    assert deferred_ui_interactions(source) == [row]
    assert deferred_ui_interactions(source_sections(effect=True, prose=prose)) == []


@pytest.mark.parametrize("operator,value", [("+=", "0"), ("-=", "0"), ("*=", "1"), ("/=", "1")])
def test_zero_effect_is_still_a_source_implementation_obligation(operator, value):
    source = source_sections(effect=True)
    mutation = source["state_model"]["specification"]["updates"][0]["mutation"][0]
    mutation.update(operator=operator, value={"kind": "number", "value": value})
    assert len(deferred_ui_interactions(source)) == 1


def test_absent_and_empty_mutations_defer_and_presentation_remains_presentation():
    source = source_sections(effect=True)
    state = source["state_model"]["specification"]
    state["updates"][0]["mutation"] = []
    assert len(deferred_ui_interactions(source)) == 1
    state["updates"] = []
    assert len(deferred_ui_interactions(source)) == 1
    row = source["resources_and_ui"]["specification"]["interactions"][0]
    row.update(trigger="Player opens the screen", server_action="Open the screen")
    assert deferred_ui_interactions(source) == []


def test_unknown_and_missing_sources_fail_when_original_requirement_ids_are_given():
    source = source_sections()
    with pytest.raises(ValueError, match="SOURCE.*UNKNOWN"):
        deferred_ui_interactions(source, requirement_ids={"other"})
    source["resources_and_ui"]["specification"]["interactions"][0].pop("source_requirement_id")
    with pytest.raises(ValueError, match="SOURCE.*UNKNOWN"):
        deferred_ui_interactions(source, requirement_ids={"feature_a"})
    assert deferred_ui_interactions(source) == []


def test_source_delegate_contract_preserves_identity_actor_and_exact_input_refs():
    source = source_sections()
    row = source["resources_and_ui"]["specification"]["interactions"][0]
    caps = source_ui_capabilities({"base": {"value": True}}, source)
    assert caps == {"base": {"value": True}, SOURCE_UI_CAPABILITY: {
        "owner": "AuthoredSourceActions", "method": "invokeSourceAction",
        "parameters": ["string", "object", "map"], "return_type": "int",
        "gameplay_mutation": False,
    }}
    assert source_ui_body(row, source, caps) == [{"op": "return", "value": {
        "op": "capability", "id": SOURCE_UI_CAPABILITY, "args": [
            {"op": "literal", "type": "string", "value": action_record_identity(row)},
            {"op": "ref", "name": "source"},
            {"op": "map", "entries": [{
                "key": {"op": "literal", "type": "string", "value": "amount"},
                "value": {"op": "ref", "name": "amount"},
            }]},
        ],
    }}]
    assert source_ui_body(row, source, {}) is None
    grounded = source_sections(effect=True)
    assert source_ui_body(grounded["resources_and_ui"]["specification"]["interactions"][0], grounded, caps) is None
    assert source_ui_capabilities({}, grounded) == {}
    prose = source_sections(prose=True)
    assert source_ui_body(prose["resources_and_ui"]["specification"]["interactions"][0], prose, caps) is None


def test_preflight_shadow_keeps_original_source_obligations():
    source = source_sections()
    before = deepcopy(source)
    shadow = source_ui_preflight_sections(source)
    assert shadow["resources_and_ui"]["specification"]["interactions"] == []
    assert shadow["state_model"] == source["state_model"]
    assert source == before
    assert source_ui_preflight_sections(source, enabled=False) == source


def test_callback_abi_uses_safe_names_and_source_scoped_identity():
    source = source_sections()
    row = source["resources_and_ui"]["specification"]["interactions"][0]
    row["server_action"]["inputs"][0]["name"] = "class"
    callback = source_ui_callback(row, package="test")
    suffix = action_record_identity(row).removeprefix("action_")
    assert callback == {
        "action_id": action_record_identity(row), "source_requirement_id": "feature_a",
        "trigger": row["trigger"], "inputs": row["server_action"]["inputs"],
        "event_id": "test.AuthoredSourceActions#EVENT_" + suffix,
        "event_owner": "test.AuthoredSourceActions", "event_field": "EVENT_" + suffix,
        "event_interface": "test.AuthoredSourceActions.Listener_" + suffix,
        "method": "onAction", "parameters": ["net.minecraft.commands.CommandSourceStack", "int"],
        "parameter_names": ["source", "input_0"], "return_type": "int",
        "throws": [], "type_parameters": [],
    }
    other = deepcopy(row)
    other["source_requirement_id"] = "feature_b"
    assert source_ui_callback(other, package="test")["event_id"] != callback["event_id"]
