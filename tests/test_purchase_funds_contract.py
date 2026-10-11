"""Regression tests for purchase balance authoring before content generation."""
from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.purchase_funds_contract import (
    currency_state_keys,
    is_purchase_interaction,
    repair_missing_purchase_funds,
)


def _section(name, updates=None):
    spec = {concern: [] for concern in DETAIL_RECORDS[name]}
    spec["inapplicable_concerns"] = []
    spec.update(updates or {})
    return {"specification": spec, "constraint_evidence_refs": []}


def _variable(name, kind="int", unit="points", domain="nonnegative game counter"):
    return {
        "name": name,
        "owner": "server",
        "type": kind,
        "unit": unit,
        "default": "0",
        "domain": domain,
    }


def _sections(*, purchase=True, variables=None):
    return {
        "state_model": _section("state_model", {
            "variables": list(variables if variables is not None else [
                _variable("player_health", unit="health"),
            ]),
        }),
        "resources_and_ui": _section("resources_and_ui", {
            "interactions": [{
                "trigger": "command:buy_parts",
                "server_action": "purchase ship part" if purchase else "inspect ship part",
                "client_feedback": "show server outcome",
            }],
        }),
    }


def test_purchase_detection_shared_across_preflight_and_dispatch():
    assert is_purchase_interaction({"server_action": "buy ship upgrade"})
    assert is_purchase_interaction({"server_action": "함선 부품 구매"})
    assert is_purchase_interaction({"server_action": "결제 후 지급"})
    assert not is_purchase_interaction({"server_action": "inspect ship part"})


def test_gold_and_credits_numeric_states_are_detected_without_model(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    def forbidden(*args, **kwargs):
        raise AssertionError("should not invoke model when currency already exists")

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", forbidden,
    )
    for name, unit in (("gold", "gold"), ("my_credits", "credits")):
        s = _sections(variables=[_variable(name, unit=unit)])
        assert currency_state_keys(s) == {name}
        assert repair_missing_purchase_funds(object(), "buy part", s) == s


def test_unrelated_numeric_state_does_not_qualify_as_money():
    s = _sections()
    assert currency_state_keys(s) == set()


def test_nonpurchase_does_not_create_currency(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    def forbidden(*args, **kwargs):
        raise AssertionError("should not call model")

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", forbidden,
    )
    s = _sections(purchase=False)
    assert repair_missing_purchase_funds(object(), "inspect ship", s) == s


def test_missing_funds_authored_as_one_canonical_variable(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation
    from minecraft_mod_ai.model_output_atomicity_contract import (
        structured_output_token_ceiling,
    )

    original = _sections()
    calls = []
    answers = iter([
        {"name": "ship_credits", "type": "int", "default": "0"},
        {"owner": "server", "unit": "credits",
         "domain": "nonnegative currency balance for ship parts"},
    ])

    def model(_router, _role, messages, *, response_schema, **kwargs):
        assert kwargs["enable_tools"] is False
        assert len(response_schema["properties"]) == 3
        assert kwargs["output_token_ceiling"] == structured_output_token_ceiling(
            response_schema
        )
        calls.append(messages)
        return next(answers)

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", model,
    )
    repaired = repair_missing_purchase_funds(
        object(), "buy ship parts with credits", original,
    )
    assert len(calls) == 2
    assert currency_state_keys(repaired) == {"ship_credits"}
    assert len(repaired["state_model"]["specification"]["variables"]) == 2
    assert repaired["state_model"]["specification"]["variables"][0] == (
        original["state_model"]["specification"]["variables"][0]
    )
    assert original == _sections()
    # The regular design validator must accept the new typed record.
    from minecraft_mod_ai.authored_structured_design import normalize_structured_sections
    assert normalize_structured_sections(repaired) == repaired


def test_full_state_slots_fails_early_without_overwriting_anything(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    def forbidden(*args, **kwargs):
        raise AssertionError("no model inference when there are no slots")

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", forbidden,
    )
    s = _sections(variables=[_variable("health_" + str(i), unit="health")
                             for i in range(4)])
    before = deepcopy(s)
    with pytest.raises(ValueError, match="PURCHASE_FUNDS_STATE_REAUTHOR_REQUIRED"):
        repair_missing_purchase_funds(object(), "buy item", s)
    assert s == before


def test_duplicate_funds_name_rejected_before_semantics_call(monkeypatch):
    from minecraft_mod_ai import fixed_template_generation

    def duplicate(*args, **kwargs):
        return {"name": "player_health", "type": "int", "default": "0"}

    monkeypatch.setattr(
        fixed_template_generation, "generate_fixed_template_value", duplicate,
    )
    with pytest.raises(ValueError, match="PURCHASE_FUNDS_STATE_REPAIR_DUPLICATE"):
        repair_missing_purchase_funds(object(), "buy part", _sections())
