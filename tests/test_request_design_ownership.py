"""Identical English trigger labels never transfer effects across requirements."""
from copy import deepcopy

import pytest

from minecraft_mod_ai import authored_structured_design as design
from minecraft_mod_ai.request_structured_design import author_request_structured_sections
from minecraft_mod_ai.typed_action_authoring import preflight_ui_action_effects
from test_typed_ui_action_authoring import sections


def merged_actions(monkeypatch, *, second_has_effect=False):
    first = sections({"intent": "Confirm", "inputs": []})
    update = first["state_model"]["specification"]["updates"][0]
    update["trigger"] = "confirm"
    update["mutation"][0].update(operator="+=", value={"kind": "number", "value": "1"})
    first["resources_and_ui"]["specification"]["interactions"][0]["trigger"] = "confirm"
    second = deepcopy(first)
    if second_has_effect:
        second["state_model"]["specification"]["updates"][0]["mutation"][0]["value"]["value"] = "2"
    else:
        first["resources_and_ui"]["specification"]["interactions"] = []
        second["state_model"]["specification"]["updates"] = []
        second["resources_and_ui"]["specification"]["interactions"][0]["server_action"]["intent"] = "Buy a ship"
    responses = iter((first, second))
    monkeypatch.setattr(design, "author_structured_sections", lambda *a, **k: next(responses))
    scope = {"requested_prompt": "Reward and purchase", "requirements": [
        {"requirement_id": key, "statement": key, "source_span": {"text": key}}
        for key in ("reward", "purchase")
    ]}
    return author_request_structured_sections(None, scope["requested_prompt"], scope)


def test_purchase_cannot_inherit_reward_with_the_same_trigger_label(monkeypatch):
    merged = merged_actions(monkeypatch)
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(merged, strict_source=True)


def test_distinct_requirements_keep_distinct_ui_identity_for_identical_labels(monkeypatch):
    from minecraft_mod_ai.authored_ui_action_bindings import action_record_identity

    merged = merged_actions(monkeypatch, second_has_effect=True)
    rows = merged["resources_and_ui"]["specification"]["interactions"]
    assert len(rows) == 2
    assert action_record_identity(rows[0]) != action_record_identity(rows[1])


def test_explicit_ui_owner_cannot_select_a_different_requirement_effect(monkeypatch):
    from minecraft_mod_ai.ui_action_source_lowering import _host_lower_ui_action_body

    merged = merged_actions(monkeypatch, second_has_effect=True)
    rows = merged["resources_and_ui"]["specification"]["interactions"]
    with pytest.raises(ValueError, match="ACTION_HOST_OWNER_TRIGGER_MISMATCH"):
        _host_lower_ui_action_body(
            rows[0], rows[0]["server_action"], ["state_model.updates[1]"],
            merged, {"amount": "int"},
        )


def test_production_ownership_lowers_same_named_transactions_separately(monkeypatch):
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts

    merged = merged_actions(monkeypatch, second_has_effect=True)
    owners = author_runtime_ownership(None, {"requirements": [
        {"requirement_id": "reward"}, {"requirement_id": "purchase"},
    ]}, merged)
    functions, bindings = author_action_functions(
        None, "Reward and purchase", merged, typed_host_capability_contracts(),
        first_index=0, runtime_owners=owners, prefer_grounded=True,
    )
    assert len(bindings) == 2
    assert [fn["body"][0]["value"]["right"]["value"] for fn in functions] == [1, 2]


def test_same_named_transactions_keep_separate_source_input_fields(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import promote_action_sections

    merged = merged_actions(monkeypatch, second_has_effect=True)
    interactions = merged["resources_and_ui"]["specification"]["interactions"]
    updates = merged["state_model"]["specification"]["updates"]
    for index, (interaction, update) in enumerate(zip(interactions, updates), 1):
        interaction["server_action"] = "Set the selected amount"
        update["mutation"][0].update(
            operator="=", value={"kind": "context_ref", "name": f"choice_{index}"},
        )
    promoted = promote_action_sections(None, merged)
    contracts = promoted["resources_and_ui"]["specification"]["interactions"]
    assert [[field["name"] for field in row["server_action"]["inputs"]]
            for row in contracts] == [["choice_1"], ["choice_2"]]
