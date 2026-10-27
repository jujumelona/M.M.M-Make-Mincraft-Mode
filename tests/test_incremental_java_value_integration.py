"""Structured pages reach legal Java in the real author and repair paths."""
from __future__ import annotations

from copy import deepcopy
import shutil
import subprocess

import pytest

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.incremental_java_model import initial_scope
from minecraft_mod_ai.incremental_java_operations import author_operation
from minecraft_mod_ai.incremental_java_render import render_member_fragment
from test_incremental_java_values import ValueCoder, member_named, method, symbols


def real_builder(steps, *, api=None, members=(), locals=()):
    spec = {"target_path": "src/main/java/demo/Feature.java",
            "bindings": {"package_name": "demo", "class_name": "Feature"}}
    builder = IncrementalJavaBuilder(ValueCoder(steps), spec, [], {"symbols": api or {}},
                                     [{"behavior_id": "integration", "intent": "Compute a value"}])
    builder.program["members"].extend(deepcopy(list(members)))
    builder.member = builder.program["members"][0]
    builder.scope = initial_scope(builder.member)
    builder.scope["locals"].extend(deepcopy(list(locals)))
    return builder


def compile_run(tmp_path, sources, main):
    paths = []
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        paths.append(str(path))
    java = shutil.which("java")
    assert java, "Java runtime required for the integration harness"
    result = subprocess.run([java, "-m", "jdk.compiler/com.sun.tools.javac.Main", "-d", str(tmp_path), *paths],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    ran = subprocess.run([java, "-cp", str(tmp_path), main], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr


def literal(value, java_type="int"):
    return [("value_kind", {"kind": "literal"}), ("literal_type", {"type": java_type}),
            ("literal", {"value": value})]


def local(name):
    return [("value_kind", {"kind": "local"}), ("local_value", {"name": name})]


def test_real_for_author_renders_a_legal_update_statement(tmp_path):
    steps = [("for", {"type": "int"}), *literal(0), ("value_kind", {"kind": "binary"}),
             ("binary_operator", {"operator": "<"}), *local("mmmValue1"), *literal(3),
             ("value_kind", {"kind": "update"}),
             ("update_operator", {"operator": "++", "position": "postfix"}),
             *local("mmmValue1"), ("kind", {"op": "complete"})]
    builder = real_builder(steps)
    builder.member["body"] = [author_operation(builder, "for")]
    body = render_member_fragment(builder.program, builder.member["id"])
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {" + body
        + "public static void main(String[] args){new Feature()." + builder.member["name"] + "();}}"}, "demo.Feature")


def test_real_super_call_can_select_protected_api(tmp_path):
    api = symbols("sample.Base", method("sample.Base", "protectedCall", returns="int", access=4))
    nested = {"id": "t1", "kind": "class", "owner": "Feature", "name": "MmmType1", "body": [],
              "metadata": {"superclass": "sample.Base", "static": True}}
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "super"}),
             ("call_member", member_named("protectedCall"))]
    builder = real_builder(steps, api=api, members=[nested])
    builder.member = {**builder.member, "owner": "MmmType1"}
    value = builder.expression("value", "int")
    compile_run(tmp_path, {"sample/Base.java": "package sample; public class Base {protected int protectedCall(){return 12;}}",
        "demo/Feature.java": "package demo; public class Feature {static class MmmType1 extends sample.Base {int check(){return "
        + value + ";}} public static void main(String[] args){if(new MmmType1().check()!=12)throw new AssertionError();}}"}, "demo.Feature")


def test_generic_inner_constructor_keeps_its_simple_type_arguments(tmp_path):
    api = symbols("sample.Outer", {"kind": "CLASS", "owner": "sample.Outer.Inner", "access": 1,
        "enclosing_owner": "sample.Outer", "type_parameters": ["T"]},
        {"kind": "CONSTRUCTOR", "owner": "sample.Outer.Inner", "name": "<init>", "parameters": [],
         "access": 1, "enclosing_instance_type": "sample.Outer"})
    steps = [("value_kind", {"kind": "new"}),
             ("new_type", {"type": "sample.Outer.Inner<java.lang.String>"}),
             ("constructor_member", member_named("<init>")), *local("outer")]
    builder = real_builder(steps, api=api, locals=[{"name": "outer", "type": "sample.Outer"}])
    value = builder.expression("value", "sample.Outer.Inner<java.lang.String>")
    compile_run(tmp_path, {"sample/Outer.java": "package sample; public class Outer {public class Inner<T> {}}",
        "demo/Feature.java": "package demo; public class Feature {public static void main(String[] args){sample.Outer outer=new sample.Outer();"
        + "sample.Outer.Inner<String> result=" + value + ";if(result==null)throw new AssertionError();}}"}, "demo.Feature")


def test_top_level_static_helper_is_selectable_again(tmp_path):
    helper = {"id": "m2", "kind": "method", "owner": "Feature", "name": "mmmHelper2", "body": [],
              "metadata": {"return_type": "int", "parameters": [], "static": True}}
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "static"}),
             ("static_owner", {"type": "Feature"}), ("call_member", {"member_id": "m2"})]
    value = real_builder(steps, members=[helper]).expression("value", "int")
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {static int mmmHelper2(){return 12;}"
        + "public static void main(String[] args){if(" + value + "!=12)throw new AssertionError();}}"}, "demo.Feature")


@pytest.mark.parametrize("slot", ["receiver", "method"])
def test_call_head_repair_uses_structured_member_choices(slot):
    from minecraft_mod_ai.incremental_java_repair_slots import replace_node
    helper = {"id": "m2", "kind": "method", "owner": "Feature", "name": "mmmHelper2", "body": [],
              "metadata": {"return_type": "void", "parameters": [], "static": False}}
    builder = real_builder([("call_receiver", {"receiver_kind": "this"}),
                            ("call_member", {"member_id": "m2"})], members=[helper])
    node = {"id": "n1", "op": "call", "receiver": "missing", "method": "missing", "arguments": [], "result_type": ""}
    replace_node(builder, node, slot)
    assert node["receiver"] == "this" and node["method"] == "mmmHelper2"
    assert node["call_contract"]["return_type"] == "void"


def test_call_repair_changes_arguments_when_the_selected_abi_changes():
    from minecraft_mod_ai.incremental_java_repair_slots import replace_node
    helper = {"id": "m2", "kind": "method", "owner": "Feature", "name": "mmmHelper2", "body": [],
              "metadata": {"return_type": "void", "parameters": [{"type": "int", "name": "arg0"}], "static": False}}
    builder = real_builder([("call_receiver", {"receiver_kind": "this"}),
                            ("call_member", {"member_id": "m2"}), *literal(12)], members=[helper])
    node = {"id": "n1", "op": "call", "receiver": "this", "method": "oldMethod", "arguments": ['"old"'],
            "result_type": "", "call_contract": {"kind": "METHOD", "return_type": "void", "parameters": ["String"]}}
    replace_node(builder, node, "method")
    assert node["call_contract"]["parameters"] == [{"type": "int", "name": "arg0"}]
    assert node["arguments"] == ["12"]


def test_call_repair_retains_arguments_when_the_selected_abi_is_unchanged():
    from minecraft_mod_ai.incremental_java_repair_slots import replace_node
    helper = {"id": "m2", "kind": "method", "owner": "Feature", "name": "mmmHelper2", "body": [],
              "metadata": {"return_type": "void", "parameters": [{"type": "int", "name": "arg0"}], "static": False}}
    builder = real_builder([("call_receiver", {"receiver_kind": "this"}),
                            ("call_member", {"member_id": "m2"})], members=[helper])
    node = {"id": "n1", "op": "call", "receiver": "this", "method": "oldMethod", "arguments": ["12"],
            "result_type": "", "call_contract": {"kind": "METHOD", "return_type": "void", "parameters": ["int"]}}
    replace_node(builder, node, "method")
    assert node["arguments"] == ["12"]


def test_varargs_parameter_is_an_array_value_without_changing_signature(tmp_path):
    builder = real_builder([("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "value"}),
                            *local("arg0"), ("field_member", member_named("length"))])
    builder.member["metadata"]["parameters"] = [{"type": "int...", "name": "arg0"}]
    builder.scope = initial_scope(builder.member)
    value = builder.expression("value", "int")
    assert builder.member["metadata"]["parameters"][0]["type"] == "int..."
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {static int check(int... arg0){return "
        + value + ";}public static void main(String[] args){if(check(1,2,3)!=3)throw new AssertionError();}}"}, "demo.Feature")


def test_generic_type_argument_pages_identify_base_index_and_prior_arguments():
    api = symbols("java.util.Map", type_parameters=["K", "V"])
    seen = []

    def argument(schema, task):
        frame = task["type_stack"][-2]
        assert frame["base"] == "java.util.Map"
        assert frame["formals"] == ["K", "V"]
        index = frame["argument_index"]
        seen.append(index)
        assert frame["arguments"] == ([] if index == 0 else ["String"])
        return {"type": "String" if index == 0 else "java.lang.Integer"}
    builder = real_builder([("local_type", {"type": "@parameterized"}),
                            ("local_type_base", {"type": "java.util.Map"}),
                            ("local_type_type_argument", argument), ("local_type_type_argument", argument)], api=api)
    builder.latest_api = {"declarations": [{"kind": "CLASS", "owner": "java.util.Map", "type_parameters": ["K", "V"]}]}
    assert builder.select_java_type("local_type") == "java.util.Map<String, java.lang.Integer>"
    assert seen == [0, 1]


def test_for_update_accepts_void_calls_and_only_statement_expression_kinds(tmp_path):
    helper = {"id": "m2", "kind": "method", "owner": "Feature", "name": "tick", "body": [],
              "metadata": {"return_type": "void", "parameters": [], "static": True}}

    def choose_call(schema, task):
        kinds = schema["properties"]["kind"]["enum"]
        assert not {"literal", "local", "binary", "array", "cast", "null"} & set(kinds)
        return {"kind": "call"}
    builder = real_builder([("value_kind", choose_call), ("call_receiver", {"receiver_kind": "static"}),
                            ("static_owner", {"type": "Feature"}), ("call_member", {"member_id": "m2"})], members=[helper])
    value = builder.expression("update", allow_update=True)
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {static int counter;static void tick(){counter++;}"
        + "public static void main(String[] args){for(int i=0;counter<3;" + value
        + "){}if(counter!=3)throw new AssertionError();}}"}, "demo.Feature")


def test_generated_type_aliases_do_not_evict_a_fresh_api_type_lookup():
    members = [{"id": "t" + str(index), "kind": "class", "owner": "Feature", "name": "MmmType" + str(index),
                "body": [], "metadata": {"static": True}} for index in range(4)]
    builder = real_builder([("local_type", {"type": "@lookup"}),
                            ("lookup", {"query": "example.World", "offset": 0}),
                            ("local_type", {"type": "example.World"})],
                           api=symbols("example.World"), members=members)
    assert builder.select_java_type("local_type") == "example.World"
    from minecraft_mod_ai.incremental_java_type_choices import suggested_types
    assert "example.World" in suggested_types(builder, "example.World")


def test_registration_repair_resumes_selection_after_cancel():
    from minecraft_mod_ai.incremental_java_repair_slots import replace_node
    event = {"event_id": "example.Events#ACTION", "event_owner": "example.Events", "event_field": "ACTION",
             "event_interface": "java.util.function.Consumer<String>", "method": "accept", "parameters": ["String"],
             "parameter_names": ["arg0"], "return_type": "void", "throws": [], "type_parameters": []}
    builder = real_builder([("register", {"event_id": "@cancel"}),
                            ("register", {"event_id": event["event_id"]})])
    builder.api._callbacks = [event]
    node = {"id": "n1", "op": "register", "event_owner": "example.Events", "event_field": "missing", "handler": "old"}
    replace_node(builder, node, "event_field")
    assert node["id"] == "n1" and node["event_field"] == "ACTION"
    listener = next(member for member in builder.program["members"] if member.get("name") == node["handler"])
    assert listener["event"] == event
