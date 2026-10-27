"""Structured value choices render and execute without model-authored Java."""
from __future__ import annotations

from copy import deepcopy
import json
import shutil
import subprocess

import pytest

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.incremental_java_model import initial_scope
from minecraft_mod_ai.incremental_java_schema import choice


class ValueCoder:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        task = json.loads(messages[-1]["content"])
        self.requests.append((task, parameters))
        suffix, answer = next(self.steps)
        assert task["requested_page"] == suffix, (task["requested_page"], suffix)
        return answer(parameters, task) if callable(answer) else answer

    def generate_java_source(self, *args, **kwargs):
        raise AssertionError("Structured values must never request Java source")


class ValueBuilder(IncrementalJavaBuilder):
    """Real builder with the small type picker supplied by the task interface."""

    def select_java_type(self, suffix, *, expected="", reference_only=False, allow_void=False, scalar_only=False):
        types = ["boolean", "byte", "short", "char", "int", "long", "float", "double",
                 "java.lang.String", "java.lang.Object", "java.lang.Math", "java.lang.Integer",
                 "java.lang.StringBuilder", "java.util.ArrayList<java.lang.String>",
                 "java.util.List<java.lang.String>", "int[]", "int[][]", "Feature", "MmmType1"]
        if reference_only:
            types = [value for value in types if "." in value or value.endswith("[]") or value in {"Feature", "MmmType1"}]
        if scalar_only:
            types = [value for value in types if value in {
                "boolean", "byte", "short", "char", "int", "long", "float", "double", "java.lang.String"}]
        if allow_void:
            types.append("void")
        return self.ask(choice("java_type", types), suffix)["java_type"]

    def expression(self, slot, expected="java.lang.Object", *, allow_update=False):
        from minecraft_mod_ai.incremental_java_values import author_value
        return author_value(self, slot, expected, allow_update=allow_update)

    def arguments(self, node):
        from minecraft_mod_ai.incremental_java_calls import _argument_types
        from minecraft_mod_ai.incremental_java_context import _ground_argument

        node["arguments"] = []
        for index, expected in enumerate(_argument_types(self, node["call_contract"])):
            node["arguments"].append(self.expression("argument:" + str(index), _ground_argument(expected, "")))
        return node["arguments"]


def builder(steps, *, symbols=None, members=(), locals=()):
    spec = {"target_path": "src/main/java/demo/Feature.java",
            "bindings": {"package_name": "demo", "class_name": "Feature"}}
    result = ValueBuilder(ValueCoder(steps), spec, [], {"symbols": symbols or {}},
                          [{"behavior_id": "values", "intent": "Compute a structured value"}])
    result.program["members"].extend(deepcopy(list(members)))
    result.member = result.program["members"][0]
    result.scope = initial_scope(result.member)
    result.scope["locals"].extend(deepcopy(list(locals)))
    result.value_stack = []
    return result


def literal(value, java_type="int"):
    return [("value_kind", {"kind": "literal"}), ("literal_type", {"java_type": java_type}),
            ("literal", {"value": value})]


def local(name):
    return [("value_kind", {"kind": "local"}), ("local_value", {"name": name})]


def member_named(name, **matches):
    def select(schema, task):
        candidates = task["member_candidates"]
        selected = next(identifier for identifier, row in candidates.items()
                        if row["name"] == name and all(row.get(key) == value for key, value in matches.items()))
        return {"member_id": selected}
    return select


def method(owner, name, parameters=(), *, returns="int", **extra):
    return {"kind": "METHOD", "owner": owner, "name": name, "access": 1,
            "parameters": list(parameters), "return_type": returns, **extra}


def symbols(owner, *members, **extra):
    return {owner: {"kind": "CLASS", "owner": owner, "access": 1, **extra},
            **{"entry" + str(index): row for index, row in enumerate(members)}}


def call(receiver, name, **matches):
    return [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": receiver}),
            ("call_member", member_named(name, **matches))]


def execute(tmp_path, expression, *, declarations="", locals="", expected="12", java_type="int"):
    java = shutil.which("java")
    if java is None:
        pytest.skip("Java runtime required")
    source = ("public class ValueHarness {" + declarations
              + " public static void main(String[] args) {" + locals
              + java_type + " result = " + expression + "; if (!java.util.Objects.deepEquals(result, "
              + expected + ")) throw new AssertionError(String.valueOf(result)); }}")
    path = tmp_path / "ValueHarness.java"
    path.write_text(source)
    compiled = subprocess.run([java, "-m", "jdk.compiler/com.sun.tools.javac.Main", str(path)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr[:3000] + "\n" + source[:1000]
    ran = subprocess.run([java, "-cp", str(tmp_path), "ValueHarness"], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr


def test_string_source_fragment_remains_literal_data_and_has_no_length_cap(tmp_path):
    text = "privatevoidmmmHelper3()//TODO:implement\n\"\\\t" + "x" * 500
    result = builder(literal(text, "java.lang.String"))

    value = result.expression("value", "java.lang.String")

    assert value.startswith('"privatevoidmmmHelper3()//TODO:implement\\n\\"\\\\\\t')
    execute(tmp_path, value, expected=json.dumps(text), java_type="java.lang.String")
    assert result.value_stack == []


@pytest.mark.parametrize("java_type,value,expected", [("long", -(2**63), "Long.MIN_VALUE"),
    ("byte", -128, "(byte)-128"), ("short", 32767, "(short)32767"),
    ("char", "'", "'\\\''"), ("float", 1.25, "1.25f"), ("boolean", True, "true")])
def test_literal_payloads_use_java_types_and_ranges(tmp_path, java_type, value, expected):
    result = builder(literal(value, java_type))
    execute(tmp_path, result.expression("value", java_type), expected=expected, java_type=java_type)


def test_nested_math_calls_and_array_index_execute_with_parentheses(tmp_path):
    api = symbols("java.lang.Math", method("java.lang.Math", "max", ["int", "int"], static=True, access=9))
    steps = [("value_kind", {"kind": "binary"}), ("binary_operator", {"operator": "*"}),
             ("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "static"}),
             ("static_owner", {"java_type": "java.lang.Math"}), ("call_member", member_named("max")),
             *literal(2), *literal(3), ("value_kind", {"kind": "index"}), *local("numbers"), *literal(1)]
    result = builder(steps, symbols=api, locals=[{"name": "numbers", "type": "int[]"}])

    value = result.expression("value", "int")

    execute(tmp_path, value, locals="int[] numbers = {8,4};")
    assert result.value_stack == []


def test_constructor_and_instance_field_are_chosen_from_exact_index(tmp_path):
    api = symbols("java.lang.StringBuilder", {"kind": "CONSTRUCTOR", "owner": "java.lang.StringBuilder",
        "name": "<init>", "parameters": ["java.lang.String"], "access": 1},
        method("java.lang.StringBuilder", "length"))
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             ("value_kind", {"kind": "new"}), ("new_type", {"java_type": "java.lang.StringBuilder"}),
             ("constructor_member", member_named("<init>")), *literal("twelve chars", "java.lang.String"),
             ("call_member", member_named("length"))]
    execute(tmp_path, builder(steps, symbols=api).expression("value", "int"))


def test_generated_instance_field_is_selected_by_id_and_writable(tmp_path):
    field = {"id": "f1", "kind": "field", "owner": "Feature", "name": "mmmField1",
             "metadata": {"type": "int", "static": False, "final": False}}
    steps = [("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "this"}),
             ("field_member", {"member_id": "f1"})]
    result = builder(steps, members=[field])

    target = result.expression("target")

    assert target == "this.mmmField1"
    kinds = result.router.requests[0][1]["properties"]["kind"]["enum"]
    assert set(kinds) == {"field", "index", "lookup"}
    execute(tmp_path, "new ValueHarness().check()", declarations="int mmmField1; int check(){"
            + target + " = 12; return mmmField1;}")


def test_assignment_target_excludes_final_fields_and_final_locals():
    field = {"id": "f1", "kind": "field", "owner": "Feature", "name": "fixed",
             "metadata": {"type": "int", "final": True}}

    def inspect_members(schema, task):
        assert "f1" not in task["member_candidates"]
        raise RuntimeError("inspected")

    result = builder([("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "this"}),
                      ("field_member", inspect_members)], members=[field])
    with pytest.raises(RuntimeError, match="inspected"):
        result.expression("target")
    assert result.value_stack == []


def test_value_call_filters_void_and_selects_exact_overload():
    api = symbols("java.lang.Math", method("java.lang.Math", "discard", returns="void", static=True, access=9),
                  method("java.lang.Math", "max", ["int", "int"], static=True, access=9),
                  method("java.lang.Math", "max", ["long", "long"], returns="long", static=True, access=9))

    def choose_long(schema, task):
        assert all(row.get("return_type") != "void" for row in task["member_candidates"].values())
        return member_named("max", parameters=["long", "long"])(schema, task)

    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "static"}),
             ("static_owner", {"java_type": "java.lang.Math"}), ("call_member", choose_long),
             *literal(3, "long"), *literal(7, "long")]
    result = builder(steps, symbols=api)
    assert result.expression("value", "long") == "java.lang.Math.max(3L, 7L)"
    assert not any(task["requested_page"] == "arguments_count" for task, _ in result.router.requests)


def test_call_head_allows_void_and_does_not_author_arguments():
    from minecraft_mod_ai.incremental_java_values import author_call_head

    api = symbols("java.lang.Math", method("java.lang.Math", "discard", ["int"], returns="void", static=True, access=9))
    result = builder([("call_receiver", {"receiver_kind": "static"}),
                      ("static_owner", {"java_type": "java.lang.Math"}),
                      ("call_member", member_named("discard"))], symbols=api)
    node = {"op": "call"}
    author_call_head(result, node)
    assert node["method"] == "discard" and node["call_contract"]["return_type"] == "void"
    assert node["receiver_type"] == "java.lang.Math"
    assert "arguments" not in node


def test_parameterized_instance_contract_is_grounded_and_preserved():
    api = symbols("java.util.List", method("java.util.List", "set", ["int", "java.lang.Object"],
            returns="java.lang.Object", generic_parameters=["int", "E"], generic_return_type="E"), type_parameters=["E"])
    result = builder([("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
                      *local("values"), ("call_member", member_named("set")),
                      *literal(0), *literal("new", "java.lang.String")], symbols=api,
                     locals=[{"name": "values", "type": "java.util.List<java.lang.String>"}])
    assert result.expression("value", "java.lang.String") == 'values.set(0, "new")'
    pages = [task for task, _ in result.router.requests if task["requested_page"] == "value_kind"]
    assert pages[-1]["value_stack"][-2]["call_contract"]["generic_parameters"] == ["int", "java.lang.String"]
    assert pages[-1]["value_stack"][-2]["arguments"] == ["0"]


def test_varargs_use_selected_generic_component_and_fixed_arity(tmp_path):
    api = symbols("java.lang.String", method("java.lang.String", "format", ["java.lang.String", "java.lang.Object[]"],
        returns="java.lang.String", generic_parameters=["java.lang.String", "java.lang.Object[]"],
        static=True, access=0x89, varargs=True))
    result = builder([("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "static"}),
                      ("static_owner", {"java_type": "java.lang.String"}), ("call_member", member_named("format")),
                      ("arguments_count", {"count": 2}), *literal("%s", "java.lang.String"),
                      *literal("twelve", "java.lang.String")], symbols=api)
    execute(tmp_path, result.expression("value", "java.lang.String"), expected='"twelve"', java_type="java.lang.String")


def test_large_catalog_exposes_next_previous_and_search_without_full_index():
    api = symbols("java.lang.Math", *(method("java.lang.Math", "method" + str(index), static=True, access=9)
                                        for index in range(35)))

    def next_page(schema, task):
        assert len(task["member_candidates"]) <= 10
        assert "next_page" in schema["properties"]["member_id"]["enum"]
        return {"member_id": "next_page"}

    result = builder([("call_receiver", {"receiver_kind": "static"}),
                      ("static_owner", {"java_type": "java.lang.Math"}),
                      ("call_member", next_page), ("call_member", {"member_id": "previous_page"}),
                      ("call_member", {"member_id": "search"}), ("member_search", {"query": "method34"}),
                      ("call_member", member_named("method34"))], symbols=api)
    from minecraft_mod_ai.incremental_java_values import author_call_head
    node = {}
    author_call_head(result, node)
    assert node["method"] == "method34"


@pytest.mark.parametrize("mode", ["elements", "sizes"])
def test_array_creation_executes(tmp_path, mode):
    steps = [("value_kind", {"kind": "array"}), ("array_type", {"java_type": "int[]"}),
             ("array_mode", {"mode": mode})]
    steps += [("array_elements_count", {"count": 2}), *literal(3), *literal(4)] if mode == "elements" else literal(2)
    result = builder(steps)
    value = result.expression("value", "int[]")
    execute(tmp_path, value + ".length", expected="2")


def test_cast_instanceof_conditional_unary_and_class_reference_execute(tmp_path):
    steps = [("value_kind", {"kind": "conditional"}), ("value_kind", {"kind": "instanceof"}),
             *local("text"), ("instanceof_type", {"java_type": "java.lang.String"}),
             ("value_kind", {"kind": "unary"}), ("unary_operator", {"operator": "-"}),
             ("value_kind", {"kind": "cast"}), ("cast_type", {"java_type": "int"}), *literal(-12, "long"), *literal(0)]
    result = builder(steps, locals=[{"name": "text", "type": "java.lang.Object"}])
    execute(tmp_path, result.expression("value", "int"), locals='Object text = "yes";')
    reference = builder([("value_kind", {"kind": "class_literal"}),
                         ("class_literal_type", {"java_type": "java.lang.String"})]).expression("value", "java.lang.Class")
    execute(tmp_path, reference, expected="String.class", java_type="Class<?>")


def test_unbound_method_reference_selects_instance_api(tmp_path):
    api = symbols("java.lang.String", method("java.lang.String", "length"))
    steps = [("value_kind", {"kind": "method_reference"}), ("reference_kind", {"reference_kind": "method"}),
             ("call_receiver", {"receiver_kind": "static"}), ("static_owner", {"java_type": "java.lang.String"}),
             ("call_member", member_named("length"))]
    value = builder(steps, symbols=api).expression("value", "java.util.function.ToIntFunction<java.lang.String>")
    execute(tmp_path, "((java.util.function.ToIntFunction<String>)" + value + ').applyAsInt("twelve chars")')


def test_new_helper_finishes_implementation_before_arguments_and_restores_context(tmp_path):
    steps = [("value_kind", {"kind": "new_helper"}), ("helper_return_type", {"java_type": "int"}),
             ("value_helper", {"purpose": "Return twelve", "static": True}), ("parameters_count", {"count": 0}),
             ("kind", {"op": "return"}), *literal(12)]
    result = builder(steps)
    result.value_stack = [{"slot": "parent", "kind": "binary", "left": "2"}]

    value = result.expression("value", "int")

    helper = result.program["members"][-1]
    assert helper["implementation_status"] == "complete" and helper["body"][0]["value"] == "12"
    assert result.value_stack == [{"slot": "parent", "kind": "binary", "left": "2"}]
    assert helper["name"] in value
    execute(tmp_path, value.replace("Feature.", "ValueHarness."),
            declarations="static int " + helper["name"] + "(){return 12;}")


def test_lookup_returns_to_value_choices_and_keeps_exact_page():
    result = builder([("value_kind", {"kind": "lookup"}), ("lookup", {"query": "@declarations", "offset": 0}),
                      *literal(12)])
    assert result.expression("value", "int") == "12"
    assert result.latest_api["generated_declarations"][0]["id"] == "m1"


def test_update_is_offered_only_when_enabled_and_targets_writable_local(tmp_path):
    result = builder([("value_kind", {"kind": "update"}),
                      ("update_operator", {"operator": "++", "position": "postfix"}), *local("counter")],
                     locals=[{"name": "counter", "type": "int"}])
    value = result.expression("update", "int", allow_update=True)
    execute(tmp_path, value, locals="int counter = 12;")
    assert "update" in result.router.requests[0][1]["properties"]["kind"]["enum"]


def test_local_catalog_paginates_exact_names_and_restores_shadowed_type():
    locals = [{"name": "value" + str(index), "type": "int"} for index in range(24)]
    locals.append({"name": "value23", "type": "long"})

    def next_page(schema, task):
        assert len(task["local_candidates"]) == 10
        return {"name": "next_page"}

    result = builder([("value_kind", {"kind": "local"}), ("local_value", next_page),
                      ("local_value", next_page), ("local_value", {"name": "previous_page"}),
                      ("local_value", next_page), ("local_value", {"name": "value23"})], locals=locals)

    assert result.expression("value", "long") == "value23"
    assert result.last_value_type == "long"


def test_primitive_value_kind_excludes_null_and_empty_local_scope():
    result = builder(literal(12))
    assert result.expression("value", "int") == "12"
    kinds = result.router.requests[0][1]["properties"]["kind"]["enum"]
    assert "null" not in kinds and "local" not in kinds
    assert result.last_value_type == "int"


def test_array_length_is_intrinsic_read_only_member(tmp_path):
    result = builder([("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "value"}),
                      *local("numbers"), ("field_member", {"member_id": "array:length"})],
                     locals=[{"name": "numbers", "type": "int[]"}])
    value = result.expression("value", "int")
    execute(tmp_path, value, locals="int[] numbers = new int[12];")
    assert result.last_value_type == "int"
