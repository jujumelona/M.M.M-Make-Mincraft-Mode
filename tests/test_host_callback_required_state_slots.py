"""Regression: required canonical state transitions are host-owned compiler work."""

import pytest

from minecraft_mod_ai.host_callback_body import author_callback_body


class Router:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append(data_name)
        return next(self.rows)


def effect_frame(*, declared=("credits",), required=("credits",)):
    methods = [
        {"name": "writeState_" + name,
         "signature": "public static void writeState_" + name + "(long value) { }"}
        for name in declared
    ]
    return {
        "kind": "method",
        "prefix": "private boolean apply(Target target) {\n",
        "suffix": "\n}",
        "host_state_access": [{"owner": "example.AuthoredStateModel", "methods": methods}],
        "host_behavior_contract": {
            "behavior_id": "trade", "intent": "purchase a ship upgrade",
            "state_reads": ["credits"], "state_writes": list(required),
        },
    }


def test_required_state_write_is_lowered_with_exact_owner_and_before_receipt_return():
    router = Router([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "19L"},
    ])
    source = author_callback_body(
        router, [], data_name="ship_purchase", frame=effect_frame(),
        effect_receipt=True,
    )
    writer = "example.AuthoredStateModel.writeState_credits(19L);"
    assert writer in source
    assert source.index(writer) < source.index("return __mmmHostEffectApplied;")
    assert router.calls[-1] == "ship_purchase_required_state_credits_try_0"


def test_missing_declared_state_is_rejected_before_any_model_work():
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED: credits"):
        author_callback_body(router, [], data_name="unowned", frame=effect_frame(declared=()),
                             effect_receipt=True)
    assert router.calls == []


def test_write_into_different_state_cannot_satisfy_original_obligation():
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED: credits"):
        author_callback_body(router, [], data_name="wrong_state",
                             frame=effect_frame(declared=("ship_level",)),
                             effect_receipt=True)
    assert router.calls == []


def test_model_does_not_choose_writable_state_identifier():
    router = Router([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "21L"},
        {"expression": "22L"},
    ])
    source = author_callback_body(
        router, [], data_name="two_state", frame=effect_frame(
            declared=("credits", "ship_level"), required=("credits", "ship_level")),
        effect_receipt=True,
    )
    assert "example.AuthoredStateModel.writeState_credits(21L);" in source
    assert "example.AuthoredStateModel.writeState_ship_level(22L);" in source
    assert router.calls[-2:] == [
        "two_state_required_state_credits_try_0", "two_state_required_state_ship_level_try_0",
    ]
