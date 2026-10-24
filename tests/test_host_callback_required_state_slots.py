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



def test_late_required_state_writer_never_bypasses_an_effect_condition():
    # Earlier host lowering appended the required mutation after the closing
    # brace, so an effect hidden behind if(false) still changed canonical
    # state. Reject the semantic ownership mismatch rather than silently
    # converting a conditional effect into an unconditional one.
    router = Router([{"action": "if"}, {"action": "if"}])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(
            router, [], data_name="conditional_purchase",
            frame=effect_frame(), effect_receipt=True,
        )
    assert not any("required_state_credits" in name for name in router.calls)
    assert "if" not in router.calls[0].split("_operation_")[0]



def test_required_writer_ignores_context_overload_and_exposes_exact_value_type():
    from minecraft_mod_ai.host_callback_body import _required_state_write_slots
    frame = effect_frame()
    frame["host_state_access"][0]["methods"].insert(0, {
        "name": "writeState_credits",
        "signature": "public static void writeState_credits(long value, java.util.Map<String, Object> context) { }",
    })
    slot = _required_state_write_slots(frame)["credits"]
    assert slot["value_type"] == "long"
    assert "(long value)" in slot["signature"]
    router = Router([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "25L"},
    ])
    author_callback_body(router, [], data_name="typed_write", frame=frame,
                         effect_receipt=True)
    request = None
    # Router records actions by name. A separate capture router can verify
    # exact contextual Java type in the mandatory writer request.
    class CaptureRouter(Router):
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if "required_state_credits" in data_name:
                nonlocal request
                request = json.loads(messages[-1]["content"])
            return super().generate_structured_data(
                role, messages, data_name=data_name, parameters=parameters)
    import json
    capture = CaptureRouter([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "25L"},
    ])
    author_callback_body(capture, [], data_name="typed_write", frame=frame,
                         effect_receipt=True)
    assert request["required_java_value_type"] == "long"


def test_required_state_write_rejects_context_only_overload():
    from minecraft_mod_ai.host_callback_body import _required_state_write_slots
    frame = effect_frame()
    frame["host_state_access"][0]["methods"] = [{
        "name": "writeState_credits",
        "signature": "public static void writeState_credits(long value, java.util.Map<String, Object> context) { }",
    }]
    with pytest.raises(ValueError, match="CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED"):
        _required_state_write_slots(frame)
