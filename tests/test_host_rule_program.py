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
        "read": {"kind": "METHOD", "owner": "example.State", "name": "readState_phase",
                 "return_type": "long", "parameters": [], "static": True, "access": 9},
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
    assert p.kind(condition) == "boolean" and "(false && true)" in p.declarations()
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
    assert p.kind(value) == "long" and "24000L" in p.declarations()
    assert not replies


def test_string_equality_uses_value_semantics():
    p = program()
    a = p.value("literal", type="java.lang.String", value="spring")
    b = p.value("literal", type="java.lang.String", value="spring")
    result = p.value("binary", [a, b], operator="==")
    assert p.kind(result) == "boolean" and "java.util.Objects.equals" in p.declarations()


def test_canonical_binding_values_are_host_copied_and_keep_provenance():
    from collections import deque
    p = program()
    replies = deque([
        ("rule_value_operation", {"operation": "data_value"}),
        ("rule_data_page", {"data_id": "approved-calendar"}),
        ("rule_data_value", {"field_id": "d0"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v1"}),
    ])

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            expected, reply = replies.popleft()
            assert data_name == expected
            return reply

    author = importlib.import_module("minecraft_mod_ai.host_rule_author").RuleAuthor(
        Router(), p, data_pages={"approved-calendar": [{"period_ticks": 17}]})
    value = author.value("Use the approved period without inventing a new number", expected="long")
    assert p.source(value) == "17"
    assert any(row.get("data_origin", {}).get("pointer") == "/0/period_ticks" for row in p.ir)


def test_state_write_authority_also_permits_read_modify_write():
    p = program()
    old = p.value("call", [], symbol_ref="read")
    one = p.value("literal", type="long", value=1)
    updated = p.value("binary", [old, one], operator="+")
    p.effect("call", [updated], symbol_ref="write")
    assert "writeState_phase" in p.render()


def test_shared_object_and_observation_values_are_evaluated_once_lazily(tmp_path):
    import shutil
    import subprocess
    assert shutil.which("javac")
    module = importlib.import_module("minecraft_mod_ai.host_rule_program")
    box = "example.Box"
    api = {
        box: {"kind": "CLASS", "owner": box, "access": 1},
        "new": {"kind": "CONSTRUCTOR", "owner": box, "name": "<init>",
                "access": 1, "parameters": [], "return_type": box},
        "set": {"kind": "METHOD", "owner": box, "name": "setValue", "access": 1,
                "parameters": ["int"], "return_type": "void"},
        "get": {"kind": "METHOD", "owner": box, "name": "getValue", "access": 1,
                "parameters": [], "return_type": "int"},
        "check": {"kind": "METHOD", "owner": box, "name": "check", "access": 9, "static": True,
                  "parameters": [box, "int"], "return_type": "void"},
    }
    p = module.RuleProgram(api, [], {}, owner="example.Bridge", target_path="src/main/java/example/Bridge.java")
    obj = p.value("new", [], symbol_ref="new")
    one = p.value("literal", type="int", value=1)
    p.effect("call", [obj, one], symbol_ref="set")
    observed = p.value("call", [obj], symbol_ref="get")
    p.effect("call", [obj, observed], symbol_ref="check")
    two = p.value("literal", type="int", value=2)
    p.effect("call", [obj, two], symbol_ref="set")
    p.effect("call", [obj, observed], symbol_ref="check")
    zero = p.value("literal", type="int", value=0)
    # A value constructed while planning a short-circuit RHS must not execute
    # unless that RHS is reached. Division by zero makes eagerness observable.
    bad = p.value("binary", [one, zero], operator="/")
    condition = p.value("binary", [bad, zero], operator=">")
    no = p.value("literal", type="boolean", value=False)
    guarded = p.value("binary", [no, condition], operator="&&")
    p.begin_if(guarded)
    p.effect("call", [obj, bad], symbol_ref="set")
    p.end()
    source = "package example; public class Bridge { public boolean run() {\n" + p.render() + "\n} }"
    folder = tmp_path / "example"
    folder.mkdir()
    (folder / "Bridge.java").write_text(source)
    (folder / "Box.java").write_text('''package example; public class Box {
        static int created; int value; public Box(){created++;}
        public void setValue(int v){value=v;} public int getValue(){return value;}
        public static void check(Box b,int old){if(created!=1 || old!=1 || b.value<1) throw new AssertionError("identity/snapshot");}
    }''')
    (folder / "Probe.java").write_text('''package example; public class Probe {
        public static void main(String[] args){if(!new Bridge().run() || Box.created!=1) throw new AssertionError();}
    }''')
    result = subprocess.run(["javac", "-d", str(tmp_path), *map(str, folder.glob("*.java"))], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(["java", "-cp", str(tmp_path), "example.Probe"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_author_can_narrow_a_computed_clock_and_construct_typed_null():
    from collections import deque
    p = program()
    clock = p.value("call", [p.inputs[0]], symbol_ref="clock")
    replies = deque([
        ("rule_value_operation", {"operation": "cast"}),
        ("rule_cast_type", {"type": "int"}),
        ("rule_operand", {"value_ref": clock}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v2"}),
    ])

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            expected, reply = replies.popleft()
            assert data_name == expected
            return reply

    module = importlib.import_module("minecraft_mod_ai.host_rule_author")
    author = module.RuleAuthor(Router(), p)
    value = author.value("Use the computed clock as an int index", expected="int")
    assert p.kind(value) == "int"
    replies.extend([
        ("rule_value_operation", {"operation": "literal"}),
        ("rule_literal_type", {"type": "example.World"}),
        ("rule_literal", {"value": None}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v3"}),
    ])
    absent = author.value("A typed null to compare a nullable world", expected="example.World")
    assert p.kind(absent) == "example.World" and p.source(absent) == "null"


def test_rule_author_can_build_new_rhs_while_old_numeric_operand_exists():
    from collections import deque
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    p = program()
    p.value("literal", type="long", value=6)
    decisions = deque([
        ("rule_value_operation", {"operation": "binary"}),
        ("rule_binary_operator", {"operator": "/"}),
        ("rule_operand", {"value_ref": "v1"}),
        ("rule_operand", {"value_ref": "build"}),
        ("rule_value_operation", {"operation": "literal"}),
        ("rule_literal_type", {"type": "long"}),
        ("rule_literal", {"value": 3}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v2"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v3"}),
    ])

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            expected, output = decisions.popleft()
            assert data_name == expected
            if data_name == "rule_operand" and output.get("value_ref") == "build":
                assert "build" in parameters["properties"]["value_ref"]["enum"]
            return output

    result = RuleAuthor(Router(), p).value("A computed ratio", expected="long")
    assert result == "v3"
    assert p.kind(result) == "long"
    assert "3L" in p.declarations()
    assert not decisions


def test_rule_api_query_accepts_static_access_bit_without_static_metadata():
    from types import SimpleNamespace
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    method = {"kind": "METHOD", "owner": "example.Utility",
              "name": "readSeason", "access": 9,
              "parameters": [], "return_type": "int"}
    stub = SimpleNamespace(symbols={"readSeason": method}, values={}, active=set(),
                           checker=SimpleNamespace(assignable=lambda *_: False))
    stub._symbol = lambda key, **kwargs: method
    author = RuleAuthor(None, stub)
    author.ask = lambda name, purpose, properties, **kwargs: (
        {"query": "readSeason"} if name == "rule_api_query" else {"symbol": "s0"})
    assert author.symbol("Read the inspected static API", "METHOD") == ("readSeason", method)



def test_rule_program_can_use_materialized_declared_state_accessors():
    from minecraft_mod_ai.host_typed_state_access import verified_state_accessor_symbols
    from minecraft_mod_ai.host_rule_program import RuleProgram

    owner = "example.AuthoredStateModel"
    accessors = verified_state_accessor_symbols([{
        "owner": owner,
        "methods": [
            {"name": "readState_phase",
             "signature": "public static synchronized long readState_phase();"},
            {"name": "writeState_phase",
             "signature": "public static synchronized void writeState_phase(long value);"},
            {"name": "writeState_phase",
             "signature": "public static synchronized void writeState_phase("
                          "long value, java.util.Map<String, Object> context);"},
            {"name": "writeState_unknown",
             "signature": "public static void unrelated(long value);"},
        ],
    }])
    assert accessors[owner]["kind"] == "CLASS"
    methods = {key: row for key, row in accessors.items() if row["kind"] == "METHOD"}
    assert len(methods) == 3
    assert not any(row["name"] == "writeState_unknown" for row in methods.values())
    assert any(row["parameters"] == ["long", "java.util.Map"]
               for row in methods.values())

    classpath = {**symbols(), **accessors}
    classpath["java.util.Map"] = {
        "kind": "CLASS", "owner": "java.util.Map", "access": 1537,
    }
    p = RuleProgram(classpath, [{"type": "example.World", "name": "arg0"}],
        {"behavior_id": "phase-transition", "state_reads": ["phase"],
         "state_writes": ["phase"]},
        owner="example.Bridge", target_path="src/main/java/example/Bridge.java",
        state_owners=[owner])
    read = next(key for key, row in methods.items()
                if row["name"] == "readState_phase")
    write = next(key for key, row in methods.items()
                 if row["name"] == "writeState_phase" and row["parameters"] == ["long"])
    old_value = p.value("call", [], symbol_ref=read)
    p.effect("call", [old_value], symbol_ref=write)
    assert p.written_states == {"phase"}
    assert "writeState_phase" in p.render()

    forbidden = RuleProgram(classpath, [], {
        "behavior_id": "no-state-permission", "state_reads": [], "state_writes": [],
    }, owner="example.Bridge", target_path="src/main/java/example/Bridge.java",
        state_owners=[owner])
    with pytest.raises(ValueError, match="STATE_AUTHORITY"):
        forbidden.value("call", [], symbol_ref=read)



def test_rule_literal_type_menu_respects_required_boolean():
    # Production MiMo previously chose ServerLevel + null 48 times while
    # the enclosing task explicitly required a boolean.
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name == "rule_literal_type"
            allowed = parameters["properties"]["type"]["enum"]
            assert allowed == ["boolean"], allowed
            return {"type": "boolean"}

    p = program()
    author = RuleAuthor(Router(), p)
    # A nonprimitive world input is present in p.menu(), but it cannot become
    # a null literal in a boolean-typed predicate.
    author._literal("Boolean intermediary", expected="boolean")


def test_gameplay_trigger_cannot_be_only_a_boolean_literal():
    from minecraft_mod_ai.host_rule_author import RuleAuthor
    p = program()
    constant = p.value("literal", type="boolean", value=True)
    author = RuleAuthor(None, p)
    assert not author._runtime_grounded(constant)
    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name == "rule_value_operation"
            operations = parameters["properties"]["operation"]["enum"]
            assert "literal" not in operations
            assert "use" not in operations
            assert "call" in operations
            raise RuntimeError("menu inspected")

    author.router = Router()
    with pytest.raises(RuntimeError, match="menu inspected"):
        author.value("Original trigger and preconditions, including every branch",
                     expected="boolean", require_grounded=True)


def test_checked_runtime_observation_can_ground_computed_boolean_gate():
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    p = program()
    time = p.value("call", [p.inputs[0]], symbol_ref="clock")
    threshold = p.value("literal", type="long", value=100)
    gate = p.value("binary", [time, threshold], operator=">=")
    pure = p.value("binary", [
        p.value("literal", type="long", value=2),
        p.value("literal", type="long", value=1)], operator=">")
    author = RuleAuthor(None, p)
    assert p.kind(gate) == "boolean"
    assert author._runtime_grounded(gate)
    assert not author._runtime_grounded(pure)
