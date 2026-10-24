"""Regression: the 2026-10-24 Colab run failed on action.value.name 'true'.

State DSL has a literal boolean branch. A keyword-shaped context reference
can never be a valid client input or Java symbol, regardless of model intent.
"""
from copy import deepcopy

import pytest

from minecraft_mod_ai.structured_state_runtime import (
    canonicalize_boolean_context_keywords,
)
from minecraft_mod_ai.typed_action_authoring import (
    preflight_ui_action_source_abi, author_action_functions,
)
from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
from test_typed_ui_action_authoring import sections


def _design(value):
    design = sections({"intent": "Set space launch flag", "inputs": []})
    state = design["state_model"]["specification"]
    state["variables"] = [{
        "name": "launch_to_space", "type": "boolean", "default": "false",
        "owner": "player", "domain": "flag", "unit": "flag",
    }]
    state["updates"][0]["mutation"] = [{
        "target": "launch_to_space", "operator": "=", "value": value,
    }]
    return design


@pytest.mark.parametrize("keyword,expected", [("true", True), ("false", False)])
def test_keyword_context_ref_is_canonical_boolean_literal(keyword, expected):
    original = _design({"kind": "context_ref", "name": keyword})
    immutable = deepcopy(original)
    fixed = canonicalize_boolean_context_keywords(original)
    assert original == immutable
    node = fixed["state_model"]["specification"]["updates"][0]["mutation"][0]["value"]
    assert node == {"kind": "literal", "value": expected}
    assert canonicalize_boolean_context_keywords(fixed) == fixed
    capabilities = typed_host_capability_contracts()
    preflight_ui_action_source_abi(fixed, capabilities)
    functions, bindings = author_action_functions(
        None, "Set launch flag", fixed, capabilities, first_index=0,
        prefer_grounded=True,
    )
    assert len(functions) == len(bindings) == 1
    assert functions[0]["body"][0]["op"] == "state_set"
    assert functions[0]["body"][0]["value"] == {
        "op": "literal", "type": "boolean", "value": expected,
    }


@pytest.mark.parametrize("missing", ["true", "false", "nonexistent", "other_input"])
def test_source_abi_cannot_silently_accept_invalid_client_input(missing):
    design = _design({"kind": "context_ref", "name": missing})
    with pytest.raises(ValueError, match="ACTION_UI_VALUE_ABI_UNBOUND"):
        preflight_ui_action_source_abi(design, typed_host_capability_contracts())


def test_declared_boolean_player_field_stays_a_reference():
    design = sections({
        "intent": "Set launch flag",
        "inputs": [{"name": "approved", "type": "boolean", "label": "Approved"}],
    })
    state = design["state_model"]["specification"]
    state["variables"][0].update(type="boolean", default="false")
    state["updates"][0]["mutation"][0]["value"] = {
        "kind": "context_ref", "name": "approved",
    }
    fixed = canonicalize_boolean_context_keywords(design)
    assert fixed == design
    preflight_ui_action_source_abi(fixed, typed_host_capability_contracts())


def test_non_boolean_constant_is_never_implicitly_coerced():
    design = sections({"intent": "Set credits", "inputs": []})
    design["state_model"]["specification"]["updates"][0]["mutation"][0]["value"] = {
        "kind": "context_ref", "name": "true",
    }
    fixed = canonicalize_boolean_context_keywords(design)
    with pytest.raises(ValueError, match="(ACTION_UI_VALUE_ABI_UNBOUND|STRUCTURED_STATE_MUTATION)"):
        preflight_ui_action_source_abi(fixed, typed_host_capability_contracts())


def test_nested_guards_migrate_only_reserved_boolean_identifiers():
    design = _design({"kind": "literal", "value": False})
    state = design["state_model"]["specification"]
    state["updates"][0]["guard"] = {
        "kind": "compare", "op": "==",
        "left": {"kind": "state_ref", "name": "launch_to_space"},
        "right": {"kind": "context_ref", "name": "true"},
    }
    fixed = canonicalize_boolean_context_keywords(design)
    assert fixed["state_model"]["specification"]["updates"][0]["guard"]["right"] == {
        "kind": "literal", "value": True,
    }
    preflight_ui_action_source_abi(fixed, typed_host_capability_contracts())


@pytest.mark.parametrize("method,tag", [
    ("_decode_int_literal", "INT"),
    ("_decode_long_literal", "LONG"),
    ("_decode_double_literal", "DOUBLE"),
])
@pytest.mark.parametrize("value", [True, False])
def test_bool_never_coerces_to_numeric_host_literal(method, tag, value):
    from minecraft_mod_ai import typed_plan_authoring as typed
    with pytest.raises(ValueError, match=f"TYPED_PLAN_LITERAL_{tag}_INVALID"):
        getattr(typed, method)(value, scope="boolean-context-regression")


def test_source_abi_rejects_wrong_mutation_operator_before_content_graph():
    design = _design({"kind": "literal", "value": True})
    design["state_model"]["specification"]["updates"][0]["mutation"][0]["operator"] = "+="
    with pytest.raises(ValueError, match="(ACTION_UI_VALUE_ABI_UNBOUND|STRUCTURED_STATE_MUTATION)"):
        preflight_ui_action_source_abi(design, typed_host_capability_contracts())
