"""The small coder supplies values; the host writes callback syntax."""
import json

import pytest

from minecraft_mod_ai.host_callback_body import author_callback_body, parse_expression


class OperationRouter:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((data_name, parameters, messages))
        return next(self.decisions)


def frame(return_type="void"):
    return {"kind": "method", "prefix": f"private {return_type} apply(Target target) {{\n",
            "suffix": "\n}"}


@pytest.mark.parametrize("bad", [
    "return true;", "true; target.mutate();", "true; } void rogue() {",
    "() -> { return true; }", "new Object() { void rogue() {} }",
])
def test_expression_slot_cannot_contain_statements_or_hidden_members(bad):
    with pytest.raises(ValueError):
        parse_expression(bad)


def test_predicate_return_and_effect_statement_delimiters_are_host_owned():
    predicate = OperationRouter([{"expression": "target.active"}])
    assert "return target.active;" in author_callback_body(
        predicate, [], data_name="predicate", frame=frame("boolean"), predicate=True)
    router = OperationRouter([
        {"action": "call"},
        {"receiver": "target", "method": "mutate", "arguments": ["1", '"a"']},
        {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="effect", frame=frame())
    assert result == frame()["prefix"] + 'target.mutate(1, "a");' + frame()["suffix"]
    first_schema = router.calls[0][1]["properties"]["action"]["enum"]
    assert "return" not in first_schema
    assert "complete" not in first_schema
    assert "complete" in router.calls[2][1]["properties"]["action"]["enum"]


def test_early_return_from_captured_production_cannot_be_an_effect_operation():
    router = OperationRouter([{"action": "return"}, {"action": "return"}])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(router, [], data_name="replay", frame=frame())


def test_host_closes_control_flow_and_names_local_values():
    router = OperationRouter([
        {"action": "declare"}, {"expression": "target.value"},
        {"action": "if"}, {"expression": "mmmValue0 > 0"},
        {"action": "assign"}, {"target": "target.value", "operator": "+=", "value": "mmmValue0"},
        {"action": "else"},
        {"action": "assign"}, {"target": "target.value", "operator": "=", "value": "1"},
        {"action": "end"}, {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="flow", frame=frame())
    assert "var mmmValue0 = target.value;" in result
    assert "if (mmmValue0 > 0) {" in result
    assert "} else {" in result
    assert result.count("{") == result.count("}")


def test_invalid_call_argument_is_corrected_with_rejected_payload_present():
    rejected = {"receiver": "target", "method": "mutate", "arguments": ["1); return;"]}
    router = OperationRouter([
        {"action": "call"}, rejected,
        {"receiver": "target", "method": "mutate", "arguments": ["1"]},
        {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="correction", frame=frame())
    assert "target.mutate(1);" in result
    assert json.loads(router.calls[2][2][-1]["content"])["rejected_payload"] == rejected


def test_predicate_cannot_mutate_canonical_state():
    router = OperationRouter([
        {"expression": "State.writeState_value(1) == null"},
        {"expression": "State.writeState_value(2) == null"},
    ])
    with pytest.raises(ValueError, match="PREDICATE_STATE_WRITE"):
        author_callback_body(router, [], data_name="predicate", frame=frame("boolean"), predicate=True)
