"""Generated declarations expose exact implicit and inherited Java contracts."""
from __future__ import annotations

import pytest

from test_incremental_java_value_integration import compile_run, literal, local, real_builder
from test_incremental_java_values import member_named, method, symbols


def generated(kind="class", **metadata):
    return {"id": "t1", "kind": kind, "owner": "Feature", "name": "MmmType1", "body": [],
            "metadata": {"static": True, **metadata}}


def instance_call(receiver, name):
    return [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": receiver}),
            ("call_member", member_named(name))]


def in_generated(builder):
    builder.member = {**builder.member, "owner": "MmmType1"}
    return builder


@pytest.mark.parametrize("receiver", ["this", "super"])
def test_generated_subclass_calls_protected_final_generic_api(tmp_path, receiver):
    api = symbols("sample.Base", method("sample.Base", "protectedCall", returns="java.lang.Object",
        generic_return_type="T", access=0x14), type_parameters=["T"])
    builder = in_generated(real_builder(instance_call(receiver, "protectedCall"), api=api,
        members=[generated(superclass="sample.Base<java.lang.String>")]))
    value = builder.expression("value", "java.lang.String")
    assert builder.last_value_type == "java.lang.String"
    compile_run(tmp_path, {"sample/Base.java": "package sample; public class Base<T> {"
        + "@SuppressWarnings(\"unchecked\") protected final T protectedCall(){return (T)\"ok\";}}",
        "demo/Feature.java": "package demo; public class Feature {static class MmmType1 extends sample.Base<String> {"
        + "String check(){return " + value + ";}}public static void main(String[] args){"
        + "if(!new MmmType1().check().equals(\"ok\"))throw new AssertionError();}}"}, "demo.Feature")


def test_generated_value_inherits_generic_interface_and_real_declaring_owner(tmp_path):
    api = symbols("sample.Root", method("sample.Root", "echo", ["java.lang.Object"],
        returns="java.lang.Object", generic_parameters=["T"], generic_return_type="T"),
        type_parameters=["T"], access=0x201)
    api["sample.Child"] = {"kind": "CLASS", "owner": "sample.Child", "access": 0x201,
        "interfaces": ["sample.Root"], "generic_interfaces": ["sample.Root<E>"], "type_parameters": ["E"]}
    def exact_owner(schema, task):
        row = next(row for row in task["member_candidates"].values() if row["name"] == "echo")
        assert row["owner"] == "sample.Root"
        return member_named("echo")(schema, task)
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             *local("item"), ("call_member", exact_owner), *literal("ok", "java.lang.String")]
    builder = real_builder(steps, api=api, members=[generated(interfaces=["sample.Child<java.lang.String>"])],
                           locals=[{"name": "item", "type": "MmmType1"}])
    value = builder.expression("value", "java.lang.String")
    assert builder.last_value_type == "java.lang.String"
    compile_run(tmp_path, {"sample/Root.java": "package sample; public interface Root<T> {default T echo(T value){return value;}}",
        "sample/Child.java": "package sample; public interface Child<E> extends Root<E> {}",
        "demo/Feature.java": "package demo; public class Feature {static class MmmType1 implements sample.Child<String> {}"
        + "public static void main(String[] args){MmmType1 item=new MmmType1();if(!("
        + value + ").equals(\"ok\"))throw new AssertionError();}}"}, "demo.Feature")


@pytest.mark.parametrize("receiver_type, allowed", [("MmmType1", True), ("sample.Base", False)])
def test_protected_access_respects_the_value_qualifier(tmp_path, receiver_type, allowed):
    api = symbols("sample.Base", method("sample.Base", "protectedCall", access=4),
                  method("sample.Base", "publicCall"))
    def choose(schema, task):
        names = {row["name"] for row in task["member_candidates"].values()}
        assert ("protectedCall" in names) is allowed
        return member_named("protectedCall" if allowed else "publicCall")(schema, task)
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             *local("other"), ("call_member", choose)]
    builder = in_generated(real_builder(steps, api=api, members=[generated(superclass="sample.Base")],
        locals=[{"name": "other", "type": receiver_type}]))
    value = builder.expression("value", "int")
    compile_run(tmp_path, {"sample/Base.java": "package sample; public class Base {protected int protectedCall(){return 12;}"
        + "public int publicCall(){return 12;}}", "demo/Feature.java": "package demo; public class Feature {"
        + "static class MmmType1 extends sample.Base {int check(" + receiver_type + " other){return " + value + ";}}"
        + "public static void main(String[] args){MmmType1 item=new MmmType1();if(item.check(item)!=12)throw new AssertionError();}}"}, "demo.Feature")


def test_record_accessor_is_derived_and_specialized_from_components(tmp_path):
    record = generated("record", type_parameters="T", components=[{"name": "item", "type": "T"}])
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             *local("record"), ("call_member", member_named("item", parameters=[]))]
    builder = real_builder(steps, members=[record], locals=[{"name": "record", "type": "MmmType1<java.lang.String>"}])
    value = builder.expression("value", "java.lang.String")
    assert builder.last_value_type == "java.lang.String"
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {record MmmType1<T>(T item) {}"
        + "public static void main(String[] args){MmmType1<String> record=new MmmType1<>(\"ok\");if(!("
        + value + ").equals(\"ok\"))throw new AssertionError();}}"}, "demo.Feature")


@pytest.mark.parametrize("name, kind, expected", [("FIRST", "field", "MmmType1.FIRST"),
    ("values", "call", "MmmType1.values()"), ("valueOf", "call", 'MmmType1.valueOf("FIRST")')])
def test_enum_implicit_contracts_compile(tmp_path, name, kind, expected):
    declaration = generated("enum", constants=[{"name": "FIRST", "arguments": ""}])
    suffix = "field" if kind == "field" else "call"
    steps = [("value_kind", {"kind": kind}), (suffix + "_receiver", {"receiver_kind": "static"}),
             ("static_owner", {"type": "MmmType1"}), (suffix + "_member", member_named(name))]
    if name == "valueOf":
        steps.extend(literal("FIRST", "java.lang.String"))
    builder = real_builder(steps, members=[declaration])
    value = builder.expression("value", "MmmType1[]" if name == "values" else "MmmType1")
    assert value == expected
    check = value + ".length!=1" if name == "values" else value + "!=MmmType1.FIRST"
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {enum MmmType1 {FIRST}"
        + "public static void main(String[] args){if(" + check + ")throw new AssertionError();}}"}, "demo.Feature")


def test_enum_constant_is_not_a_writable_field():
    def readonly(schema, task):
        assert not task["member_candidates"]
        raise RuntimeError("read only")
    steps = [("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "static"}),
             ("static_owner", {"type": "MmmType1"}), ("field_member", readonly)]
    builder = real_builder(steps, members=[generated("enum", constants=[{"name": "FIRST"}])])
    with pytest.raises(RuntimeError, match="read only"):
        builder.expression("target", "MmmType1")


@pytest.mark.parametrize("nested", [False, True])
def test_generated_implicit_constructor_has_its_real_enclosing_context(tmp_path, nested):
    owner = "MmmType1" if nested else "Feature"
    steps = [("value_kind", {"kind": "new"}), ("new_type", {"type": owner}),
             ("constructor_member", member_named(owner))]
    if nested:
        steps.extend(local("outer"))
    builder = real_builder(steps, members=[generated(static=False)] if nested else [],
                           locals=[{"name": "outer", "type": "Feature"}])
    value = builder.expression("value", owner)
    assert value == "(outer).new MmmType1()" if nested else value == "new Feature()"
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {class MmmType1 {}"
        + "public static void main(String[] args){Feature outer=new Feature();Object value=" + value
        + ";if(value==null)throw new AssertionError();}}"}, "demo.Feature")


def test_record_canonical_constructor_survives_an_explicit_overload(tmp_path):
    declaration = generated("record", components=[{"name": "item", "type": "int"}])
    overload = {"id": "m2", "kind": "constructor", "name": "MmmType1", "owner": "MmmType1", "body": [],
                "metadata": {"parameters": [], "visibility": "private"}}
    def canonical(schema, task):
        assert "m2" in task["member_candidates"]
        assert "t1" in task["member_candidates"]
        return {"member_id": "t1"}
    steps = [("value_kind", {"kind": "new"}), ("new_type", {"type": "MmmType1"}),
             ("constructor_member", canonical), *literal(12)]
    builder = real_builder(steps, members=[declaration, overload])
    value = builder.expression("value", "MmmType1")
    compile_run(tmp_path, {"demo/Feature.java": "package demo; public class Feature {record MmmType1(int item) {"
        + "MmmType1(){this(0);}}public static void main(String[] args){if((" + value
        + ").item()!=12)throw new AssertionError();}}"}, "demo.Feature")


def test_external_api_owner_with_the_root_class_name_is_not_generated(tmp_path):
    api = symbols("sample.Feature", method("sample.Feature", "result"))
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             *local("external"), ("call_member", member_named("result"))]
    builder = real_builder(steps, api=api, locals=[{"name": "external", "type": "sample.Feature"}])
    value = builder.expression("value", "int")
    compile_run(tmp_path, {"sample/Feature.java": "package sample; public class Feature {public int result(){return 12;}}",
        "demo/Feature.java": "package demo; public class Feature {public static void main(String[] args){"
        + "sample.Feature external=new sample.Feature();if(" + value + "!=12)throw new AssertionError();}}"}, "demo.Feature")


def test_generated_and_inherited_pages_keep_exact_ids_and_navigation(tmp_path):
    api = symbols("sample.Base", *(method("sample.Base", "method" + str(index)) for index in range(24)),
        {"kind": "CONSTRUCTOR", "owner": "sample.Base", "name": "<init>", "access": 1, "parameters": ["int"]})
    own = {"id": "m2", "kind": "method", "owner": "MmmType1", "name": "own", "body": [],
           "metadata": {"return_type": "int", "parameters": [], "static": False}}
    def generated_page(schema, task):
        assert list(task["member_candidates"]) == ["m2"]
        return {"member_id": "next_page"}
    def api_page(schema, task):
        assert len(task["member_candidates"]) == 10
        assert all(row["owner"] == "sample.Base" and row["kind"] == "METHOD"
                   for row in task["member_candidates"].values())
        return {"member_id": "next_page"}
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             *local("item"), ("call_member", generated_page), ("call_member", api_page),
             ("call_member", {"member_id": "previous_page"}), ("call_member", {"member_id": "previous_page"}),
             ("call_member", generated_page), ("call_member", {"member_id": "search"}),
             ("member_search", {"query": "method23"}), ("call_member", member_named("method23"))]
    builder = real_builder(steps, api=api, members=[generated(superclass="sample.Base"), own],
                           locals=[{"name": "item", "type": "MmmType1"}])
    value = builder.expression("value", "int")
    compile_run(tmp_path, {"sample/Base.java": "package sample; public class Base {public int method23(){return 12;}}",
        "demo/Feature.java": "package demo; public class Feature {static class MmmType1 extends sample.Base {int own(){return 0;}}"
        + "public static void main(String[] args){MmmType1 item=new MmmType1();if(" + value
        + "!=12)throw new AssertionError();}}"}, "demo.Feature")


def test_inherited_final_field_is_filtered_for_writes():
    api = symbols("sample.Base", {"kind": "FIELD", "owner": "sample.Base", "name": "fixed",
        "return_type": "int", "access": 0x11}, {"kind": "FIELD", "owner": "sample.Base", "name": "mutable",
        "return_type": "int", "access": 1})
    def mutable(schema, task):
        assert {row["name"] for row in task["member_candidates"].values()} == {"mutable"}
        return member_named("mutable")(schema, task)
    steps = [("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "value"}),
             *local("item"), ("field_member", mutable)]
    builder = real_builder(steps, api=api, members=[generated(superclass="sample.Base")],
                           locals=[{"name": "item", "type": "MmmType1"}])
    assert builder.expression("target", "int") == "item.mutable"


@pytest.mark.parametrize("foreign_name", ["Feature", "MmmType1"])
def test_foreign_same_simple_owner_does_not_grant_protected_access(tmp_path, foreign_name):
    owner = "sample." + foreign_name
    api = symbols(owner, method(owner, "protectedCall", access=4), method(owner, "publicCall"))
    def public_only(schema, task):
        assert {row["name"] for row in task["member_candidates"].values()} == {"publicCall"}
        return member_named("publicCall")(schema, task)
    steps = [("value_kind", {"kind": "call"}), ("call_receiver", {"receiver_kind": "value"}),
             *local("external"), ("call_member", public_only)]
    builder = real_builder(steps, api=api, members=[generated()],
                           locals=[{"name": "external", "type": owner}])
    if foreign_name == "MmmType1":
        in_generated(builder)
    value = builder.expression("value", "int")
    compile_run(tmp_path, {"sample/" + foreign_name + ".java": "package sample; public class " + foreign_name
        + " {protected int protectedCall(){return 0;}public int publicCall(){return 12;}}",
        "demo/Feature.java": "package demo; public class Feature {static class MmmType1 {int check("
        + owner + " external){return " + value + ";}}public static void main(String[] args){"
        + "if(new MmmType1().check(new " + owner + "())!=12)throw new AssertionError();}}"}, "demo.Feature")
