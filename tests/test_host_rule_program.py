"""Typed rule values are executable data; no model-authored Java enters them."""
import importlib

import pytest


def symbols():
    return {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "example.State": {"kind": "CLASS", "owner": "example.State", "access": 1},
        "java.lang.String": {"kind": "CLASS", "owner": "java.lang.String", "access": 17},
        "clock": {"kind": "METHOD", "owner": "example.World", "name": "getTime",
                  "return_type": "long", "parameters": [], "access": 1},
        "danger": {"kind": "METHOD", "owner": "example.World", "name": "explode",
                   "return_type": "boolean", "parameters": [], "access": 1},
        "effects": {"kind": "FIELD", "owner": "example.World", "name": "effects",
                    "return_type": "int", "access": 1},
        "flags": {"kind": "FIELD", "owner": "example.World", "name": "flags",
                  "return_type": "int[]", "access": 1},
        "write": {"kind": "METHOD", "owner": "example.State", "name": "writeState_phase",
                  "return_type": "void", "parameters": ["long"], "static": True, "access": 9},
        "write_other": {"kind": "METHOD", "owner": "example.State", "name": "writeState_money",
                        "return_type": "void", "parameters": ["long"], "static": True, "access": 9},
        "take_int": {"kind": "METHOD", "owner": "example.World", "name": "take",
                     "return_type": "void", "parameters": ["int"], "access": 1},
        "take_long": {"kind": "METHOD", "owner": "example.World", "name": "take",
                      "return_type": "void", "parameters": ["long"], "access": 1},
    }


def program():
    module = importlib.import_module("minecraft_mod_ai.host_rule_program")
    return module.RuleProgram(symbols(), [{"type": "example.World", "name": "arg0"}],
        {"behavior_id": "boundary", "state_reads": [], "state_writes": ["phase"]},
        owner="example.Bridge", target_path="src/main/java/example/Bridge.java",
        state_owners={"example.State"})


def test_clock_arithmetic_and_state_write_are_typed_and_overloads_are_exact():
    p = program()
    clock = p.value("call", [p.inputs[0]], symbol_ref="clock")
    divisor = p.value("literal", type="long", value=24000)
    phase = p.value("binary", [clock, divisor], operator="/")
    assert p.kind(phase) == "long"
    p.effect("call", [phase], symbol_ref="write")
    p.effect("call", [p.inputs[0], phase], symbol_ref="take_long")
    code = p.render()
    assert "24000L" in code and "((long)" in code
    assert "example.State.writeState_phase" in code
    assert p.written_states == {"phase"}
    with pytest.raises(ValueError, match="PARAMETER_TYPE"):
        p.effect("call", [p.inputs[0], phase], symbol_ref="take_int")


def test_value_cannot_call_world_mutation_or_write_other_behavior_state():
    p = program()
    with pytest.raises(ValueError, match="OBSERVATION_CALL"):
        p.value("call", [p.inputs[0]], symbol_ref="danger")
    value = p.value("literal", type="long", value=3)
    with pytest.raises(ValueError, match="STATE_AUTHORITY"):
        p.effect("call", [value], symbol_ref="write_other")
    with pytest.raises(ValueError, match="SYMBOL_MISSING"):
        p.value("call", [], symbol_ref="madeUp(); explode()")


def test_literals_are_escaped_and_cannot_add_source_statements():
    p = program()
    raw = '"; System.exit(0); //'
    literal = p.value("literal", type="java.lang.String", value=raw)
    assert p.source(literal) == '"\\\"; System.exit(0); //"'
    with pytest.raises(ValueError, match="LITERAL"):
        p.value("literal", type="int", value="0; System.exit(0)")


def test_nested_scope_values_do_not_escape_and_empty_effects_are_rejected():
    p = program()
    flags = p.value("field", [p.inputs[0]], symbol_ref="flags")
    element = p.begin_each(flags)
    target = p.value("field", [p.inputs[0]], symbol_ref="effects")
    p.effect("assign", [target, element], operator="+=")
    p.end()
    assert "for (int " in p.render()
    with pytest.raises(ValueError, match="VALUE_SCOPE"):
        p.source(element)
    with pytest.raises(ValueError, match="EFFECT_REQUIRED"):
        program().render()


def test_boolean_composition_preserves_short_circuit_as_expression_tree():
    p = program()
    no = p.value("literal", type="boolean", value=False)
    yes = p.value("literal", type="boolean", value=True)
    condition = p.value("binary", [no, yes], operator="&&")
    assert p.source(condition) == "(false && true)"
    with pytest.raises(ValueError, match="BOOLEAN_REQUIRED"):
        number = p.value("literal", type="int", value=1)
        p.value("binary", [number, yes], operator="&&")


def test_model_authors_clock_formula_by_ids_without_java_or_type_claims():
    import json
    from collections import deque
    p = program()
    replies = deque([
        ("rule_value_operation", {"operation": "call"}),
        ("rule_api_query", {"query": "getTime"}),
        ("rule_api_page", {"symbol": "s0"}),
        ("rule_operand", {"value_ref": "v0"}),
        ("rule_value_operation", {"operation": "literal"}),
        ("rule_literal_type", {"type": "long"}),
        ("rule_literal", {"value": 24000}),
        ("rule_value_operation", {"operation": "binary"}),
        ("rule_binary_operator", {"operator": "/"}),
        ("rule_operand", {"value_ref": "v1"}),
        ("rule_operand", {"value_ref": "v2"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v3"}),
    ])

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            expected, reply = replies.popleft()
            assert data_name == expected
            task = json.loads(messages[-1]["content"])
            assert len(json.dumps(task)) < 7000
            assert "expression" not in parameters["properties"]
            return reply

    author = importlib.import_module("minecraft_mod_ai.host_rule_author").RuleAuthor(Router(), p)
    value = author.value("Compute a phase index from the actual clock", expected="long")
    assert p.kind(value) == "long" and "24000L" in p.source(value)
    assert not replies


def test_string_equality_uses_value_semantics():
    p = program()
    a = p.value("literal", type="java.lang.String", value="spring")
    b = p.value("literal", type="java.lang.String", value="spring")
    result = p.value("binary", [a, b], operator="==")
    assert "java.util.Objects.equals" in p.source(result)
