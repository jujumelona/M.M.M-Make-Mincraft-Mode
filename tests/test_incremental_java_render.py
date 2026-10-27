"""Executable contracts for host-owned incremental Java syntax."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from minecraft_mod_ai.incremental_java_render import (
    render_program,
    validate_expression,
    validate_type,
)


@pytest.mark.parametrize("text", [
    "new java.util.HashMap<String, java.util.List<Integer>>()",
    "(java.util.List<? extends Number>) values",
    "java.util.List::<String>of",
    "java.util.ArrayList<String>::new",
    "new int[] {1, 2, 3}",
    '"a; class Fake {} -> \\\""',
    "'{'",
    "(left instanceof String item) && !item.isEmpty()",
    '"""\nmultiline\ntext\n"""',
    '"' + "x" * 18000 + '"',
])
def test_expression_accepts_java_expressions_without_source_length_limits(text: str) -> None:
    assert validate_expression("  " + text + "  ") == text


@pytest.mark.parametrize("text", [
    "",
    "return answer;",
    "answer; doOtherThing()",
    "new Object() { void run() {} }",
    "(Runnable) (() -> doOtherThing())",
    "java.util.List.of(() -> doOtherThing())",
    "value); } class Injected {} class Extra { Object value = (null",
    "value); int sibling = 1; Object other = (value",
    '"\\u0022); } class Injected {} class Extra { Object value = (null//"',
    "value++",
    "target = value",
])
def test_expression_rejects_statements_bodies_and_scope_escapes(text: str) -> None:
    with pytest.raises(ValueError):
        validate_expression(text)


def test_expression_update_is_available_only_in_its_explicit_slot() -> None:
    assert validate_expression("i++", allow_update=True) == "i++"
    assert validate_expression("i += 2", allow_update=True) == "i += 2"
    with pytest.raises(ValueError):
        validate_expression("i++; launch()", allow_update=True)


@pytest.mark.parametrize("text", [
    "int", "void", "java.util.Map<String, java.util.List<? extends Number>>[]",
    "@Deprecated String", "String...",
])
def test_type_is_parser_backed_and_preserves_generic_spelling(text: str) -> None:
    assert validate_type("  " + text + "  ") == text


@pytest.mark.parametrize("text", [
    "", "int count", "String; void escaped() {} String", "new Thing()", "List<>",
    "String value = null", "String) {} class Escaped {} class More { void x(String",
])
def test_type_rejects_declarations_and_expressions(text: str) -> None:
    with pytest.raises(ValueError):
        validate_type(text)


def _method(name: str, body: list[dict], *, owner: str = "Demo", return_type: str = "void",
            parameters: list[dict] | None = None, **extra) -> dict:
    return {"id": name, "owner": owner, "kind": "method", "name": name,
            "metadata": {"visibility": "public", "static": False, "return_type": return_type,
                         "parameters": parameters or [], "type_parameters": "", "throws": [],
                         "annotations": []},
            "purpose": name, "behavior": {"purpose": name}, "body": body, **extra}


def _program(members: list[dict]) -> dict:
    return {"schema": "mmm/incremental-java-v1", "package": "sample", "class_name": "Demo",
            "interface": "Lifecycle", "lifecycle": "onInitialize", "members": members}


def _field(name: str, type_name: str, value: str, *, owner: str = "Demo") -> dict:
    return {"id": owner + ":" + name, "owner": owner, "kind": "field", "name": name,
            "metadata": {"visibility": "private", "static": False, "final": False,
                         "type": type_name}, "initializer": value}


def _java(tmp_path: Path, source: str, harness: str = "") -> subprocess.CompletedProcess:
    java = shutil.which("java")
    if java is None:
        pytest.skip("Java runtime is unavailable")
    package = tmp_path / "sample"
    package.mkdir(exist_ok=True)
    (package / "Demo.java").write_text(source, encoding="utf-8")
    (package / "Harness.java").write_text(
        "package sample; interface Lifecycle { void onInitialize(); }\n" + harness,
        encoding="utf-8",
    )
    return subprocess.run(
        [java, "-m", "jdk.compiler/com.sun.tools.javac.Main", "-d", str(tmp_path),
         str(package / "Demo.java"), str(package / "Harness.java")],
        text=True, capture_output=True, check=False, timeout=30,
    )


def test_rendered_generic_operations_execute_through_registered_lifecycle(tmp_path: Path) -> None:
    body = [
        {"id": "sum", "op": "local", "type": "int", "name": "sum", "value": "0"},
        {"id": "loop", "op": "for", "type": "int", "name": "i", "value": "0",
         "condition": "i < 4", "update": "i++", "body": [
             {"id": "skip", "op": "if", "condition": "i == 1",
              "body": [{"id": "continue", "op": "continue"}], "otherwise": []},
             {"id": "addIndex", "op": "assign", "target": "sum", "operator": "+=", "value": "i"}]},
        {"id": "each", "op": "foreach", "type": "int", "name": "item",
         "iterable": "new int[] {2, 3}", "body": [
             {"id": "addItem", "op": "assign", "target": "sum", "operator": "+=", "value": "item"}]},
        {"id": "while", "op": "while", "condition": "sum < 12", "body": [
             {"id": "increment", "op": "assign", "target": "sum", "operator": "+=", "value": "1"}]},
        {"id": "do", "op": "do", "condition": "false", "body": [
             {"id": "decrement", "op": "assign", "target": "sum", "operator": "-=", "value": "1"}]},
        {"id": "choose", "op": "switch", "selector": "sum", "cases": [
            {"label": "11", "body": [
                {"id": "caseAdd", "op": "assign", "target": "sum", "operator": "+=", "value": "4"},
                {"id": "break", "op": "break"}]},
            {"label": "", "body": [
                {"id": "wrongBranch", "op": "throw", "value": "new IllegalStateException()"}]}]},
        {"id": "parse", "op": "try", "body": [
            {"id": "parseCall", "op": "call", "receiver": "Integer", "method": "parseInt",
             "arguments": ['"7"'], "result_type": "int", "name": "parsed"},
            {"id": "parsedAdd", "op": "assign", "target": "sum", "operator": "+=", "value": "parsed"}],
         "catches": [{"type": "RuntimeException", "name": "failure", "body": [
             {"id": "rethrow", "op": "throw", "value": "failure"}]}], "finally": [
             {"id": "finallyAdd", "op": "assign", "target": "sum", "operator": "+=", "value": "1"}]},
        {"id": "lock", "op": "synchronized", "monitor": "this", "body": [
            {"id": "block", "op": "block", "body": [
                {"id": "lockedAdd", "op": "assign", "target": "sum", "operator": "+=", "value": "2"}]}]},
        {"id": "double", "op": "lambda", "type": "java.util.function.IntUnaryOperator",
         "name": "transform", "parameters": [{"type": "int", "name": "value"}],
         "return_type": "int", "body": [
             {"id": "lambdaReturn", "op": "return", "value": "value * 2"}]},
        {"id": "nested", "op": "local", "type": "Counter", "name": "counter",
         "value": "new Counter(sum)"},
        {"id": "read", "op": "call", "receiver": "counter", "method": "amount",
         "arguments": [], "result_type": "int", "name": "base"},
        {"id": "apply", "op": "call", "receiver": "transform", "method": "applyAsInt",
         "arguments": ["base"], "result_type": "int", "name": "doubled"},
        {"id": "store", "op": "assign", "target": "this.total", "operator": "+=", "value": "doubled"},
        {"id": "done", "op": "return", "value": ""},
    ]
    registrar = _method("wire", [{"id": "register", "op": "register", "event_owner": "Harness",
                                  "event_field": "EVENT", "handler": "onTick"}], registrar=True)
    counter = {"id": "Counter", "owner": "Demo", "kind": "class", "name": "Counter",
               "metadata": {"visibility": "private", "static": True, "superclass": "",
                            "interfaces": [], "fields": [], "constructors": [], "methods": []}, "body": []}
    constructor = {"id": "counterInit", "owner": "Counter", "kind": "constructor", "name": "Counter",
                   "metadata": {"visibility": "private", "parameters": [{"type": "int", "name": "value"}],
                                "annotations": [], "throws": []}, "body": [
                       {"id": "super", "op": "constructor_call", "target": "super", "arguments": []},
                       {"id": "setValue", "op": "assign", "target": "this.value", "operator": "=", "value": "value"}]}
    identity = _method("identity", [{"id": "idReturn", "op": "return", "value": "value"}],
                       return_type="T", parameters=[{"type": "T", "name": "value"}])
    identity["metadata"]["type_parameters"] = "T extends Number & Comparable<T>"
    ignored = _method("notARegistrar", [{"id": "mustNotRun", "op": "assign", "target": "total",
                                         "operator": "=", "value": "999"}])
    program = _program([_field("total", "int", "0"), registrar, ignored, _method("onTick", body),
                        _method("result", [{"id": "result", "op": "return", "value": "total"}], return_type="int"),
                        identity, counter, _field("value", "int", "", owner="Counter"), constructor,
                        _method("amount", [{"id": "amount", "op": "return", "value": "value"}],
                                owner="Counter", return_type="int")])
    original = deepcopy(program)
    source, mapping = render_program(program)
    assert render_program(program) == (source, mapping)
    assert program == original
    harness = """
public class Harness {
    static final Event EVENT = new Event();
    static final class Event {
        Runnable action;
        void register(Runnable next) { action = next; }
        void fire() { action.run(); }
    }
    public static void main(String[] args) {
        Demo demo = new Demo();
        demo.onInitialize();
        EVENT.fire(); EVENT.fire();
        if (demo.result() != 100) throw new AssertionError(demo.result());
        System.out.println(demo.identity(17));
    }
}
"""
    compiled = _java(tmp_path, source, harness)
    assert compiled.returncode == 0, compiled.stderr + "\n" + source
    executed = subprocess.run([shutil.which("java"), "-cp", str(tmp_path), "sample.Harness"],
                              text=True, capture_output=True, check=False, timeout=30)
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.strip() == "17"
    assert any(item == {"member_id": "onTick", "node_id": "lambdaReturn", "slot": "value"}
               for item in mapping.values())


@pytest.mark.parametrize(("broken", "member_id", "node_id", "slot"), [
    ({"id": "bad", "op": "local", "type": "int", "name": "value", "value": "missingValue"},
     "work", "bad", "value"),
    ({"id": "bad", "op": "call", "receiver": "Integer", "method": "parseInt",
      "arguments": ["missingValue"], "result_type": "int", "name": "value"},
     "work", "bad", "argument:0"),
])
def test_javac_error_lines_point_to_expression_operands(tmp_path: Path, broken: dict,
                                                       member_id: str, node_id: str, slot: str) -> None:
    source, mapping = render_program(_program([_method("work", [broken])]))
    compiled = _java(tmp_path, source)
    assert compiled.returncode != 0
    match = re.search(r"Demo\.java:(\d+): error", compiled.stderr)
    assert match, compiled.stderr
    assert mapping[int(match[1])] == {"member_id": member_id, "node_id": node_id, "slot": slot}


def test_unfinished_return_is_a_compiler_diagnostic_not_a_renderer_gate(tmp_path: Path) -> None:
    source, mapping = render_program(_program([_method("unfinished", [], return_type="int")]))
    compiled = _java(tmp_path, source)
    assert compiled.returncode != 0
    match = re.search(r"Demo\.java:(\d+): error", compiled.stderr)
    assert match, compiled.stderr
    assert mapping[int(match[1])]["member_id"] == "unfinished"


def test_registrars_require_an_explicit_flag_and_behavior() -> None:
    marked = _method("marked", [], registrar=True)
    legacy = _method("compatible", [])
    legacy["metadata"]["host_registrar"] = True
    no_behavior = _method("noBehavior", [], registrar=True)
    no_behavior["behavior"] = {}
    source, _ = render_program(_program([marked, legacy, no_behavior, _method("helper", [])]))
    initialize = source.split("void onInitialize()", 1)[1].split("}", 1)[0]
    assert "this.marked();" in initialize
    assert "this.compatible();" in initialize
    assert "this.noBehavior();" not in initialize
    assert "this.helper();" not in initialize


@pytest.mark.parametrize(("op", "field", "value"), [
    ("local", "name", "x; run()"),
    ("call", "method", "run(); sneaky"),
    ("local", "value", "new Object() { int escaped; }"),
    ("local", "value", "java.util.List.of((Runnable) () -> run())"),
])
def test_renderer_applies_operand_boundaries(op: str, field: str, value: str) -> None:
    node = ({"id": "node", "op": "local", "type": "Object", "name": "value", "value": "null"}
            if op == "local" else
            {"id": "node", "op": "call", "receiver": "this", "method": "run", "arguments": [],
             "result_type": "", "name": ""})
    node[field] = value
    with pytest.raises(ValueError):
        render_program(_program([_method("work", [node])]))


def test_member_fragment_preserves_nested_owner_and_overload_identity() -> None:
    from minecraft_mod_ai.incremental_java_render import render_member_fragment

    first = _method("value", [{"id": "one", "op": "return", "value": "1"}], return_type="int")
    first["id"] = "first"
    second = _method("value", [{"id": "two", "op": "return", "value": "input"}],
                     return_type="int", parameters=[{"type": "int", "name": "input"}])
    second["id"] = "second"
    fragment = render_member_fragment(_program([first, second]), "second")
    assert "value(int input)" in fragment
    assert "value()" not in fragment
    assert "return\n" in fragment
    assert "input" in fragment
    assert "public final class Demo" not in fragment
    with pytest.raises(ValueError):
        render_member_fragment(_program([first, second]), "missing")


@pytest.mark.parametrize("text", [
    "@A(value = new Object() { void leak() {} }) String",
    "@A(value = () -> leak()) String",
])
def test_type_annotations_cannot_smuggle_authored_bodies(text: str) -> None:
    with pytest.raises(ValueError):
        validate_type(text)


def test_declaration_annotations_cannot_smuggle_authored_bodies() -> None:
    member = _method("work", [])
    member["metadata"]["annotations"] = ["A(new Object() { void leak() {} })"]
    with pytest.raises(ValueError):
        render_program(_program([member]))


def test_nested_generic_class_record_and_enum_compile_and_execute(tmp_path: Path) -> None:
    def type_member(kind: str, name: str, **metadata) -> dict:
        return {"id": name, "owner": "Demo", "name": name, "kind": kind, "body": [],
                "metadata": {"visibility": "private", "static": True, "interfaces": [],
                             "fields": [], "constructors": [], "methods": [], **metadata}}

    box = type_member("class", "Box", superclass="", type_parameters="T")
    pair = type_member("record", "Pair", components=[{"type": "int", "name": "first"},
                                                       {"type": "int", "name": "second"}])
    mode = type_member("enum", "Mode", constants=[{"name": "A", "arguments": "3"},
                                                  {"name": "B", "arguments": "7"}])
    constructor = {"id": "boxConstructor", "owner": "Box", "name": "Box", "kind": "constructor",
                   "metadata": {"visibility": "private", "parameters": [{"type": "T", "name": "value"}]},
                   "body": [{"id": "boxSet", "op": "assign", "target": "this.payload", "operator": "=", "value": "value"}]}
    enum_constructor = {"id": "modeConstructor", "owner": "Mode", "name": "Mode", "kind": "constructor",
                        "metadata": {"visibility": "private", "parameters": [{"type": "int", "name": "value"}]},
                        "body": [{"id": "modeSet", "op": "assign", "target": "this.number", "operator": "=", "value": "value"}]}
    members = [box, pair, mode, _field("payload", "T", "", owner="Box"), constructor,
               _method("read", [{"id": "readPayload", "op": "return", "value": "payload"}], owner="Box", return_type="T"),
               _field("number", "int", "", owner="Mode"), enum_constructor,
               _method("number", [{"id": "readNumber", "op": "return", "value": "number"}], owner="Mode", return_type="int"),
               _method("result", [{"id": "createTypes", "op": "return",
                                   "value": 'new Box<String>(new Pair(2, 3).toString() + Mode.B.number()).read()'}],
                       return_type="String")]
    source, _ = render_program(_program(members))
    compiled = _java(tmp_path, source, """
public class Harness {
    public static void main(String[] args) {
        System.out.println(new Demo().result());
    }
}
""")
    assert compiled.returncode == 0, compiled.stderr + "\n" + source
    executed = subprocess.run([shutil.which("java"), "-cp", str(tmp_path), "sample.Harness"],
                              text=True, capture_output=True, check=False, timeout=30)
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.strip() == "Pair[first=2, second=3]7"


def test_super_receiver_and_multi_catch_are_native_java_operands(tmp_path: Path) -> None:
    assert validate_type("IllegalArgumentException | IllegalStateException") == "IllegalArgumentException | IllegalStateException"
    method = _method("work", [
        {"id": "superHash", "op": "call", "receiver": "super", "method": "hashCode",
         "arguments": [], "result_type": "int", "name": "ignored"},
        {"id": "result", "op": "local", "type": "int", "name": "result", "value": "0"},
        {"id": "catch", "op": "try", "body": [
            {"id": "throw", "op": "throw", "value": "new IllegalArgumentException()"}],
         "catches": [{"type": "IllegalArgumentException | IllegalStateException", "name": "failure", "body": [
             {"id": "recover", "op": "assign", "target": "result", "operator": "=", "value": "7"}]}], "finally": []},
        {"id": "return", "op": "return", "value": "result"}], return_type="int")
    source, _ = render_program(_program([method]))
    compiled = _java(tmp_path, source, """
public class Harness {
    public static void main(String[] args) {
        if (new Demo().work() != 7) throw new AssertionError();
    }
}
""")
    assert compiled.returncode == 0, compiled.stderr + "\n" + source
    executed = subprocess.run([shutil.which("java"), "-cp", str(tmp_path), "sample.Harness"],
                              text=True, capture_output=True, check=False, timeout=30)
    assert executed.returncode == 0, executed.stderr


def test_member_fragment_keeps_unicode_literal_line_separators() -> None:
    from minecraft_mod_ai.incremental_java_render import render_member_fragment

    member = _field("separator", "String", '"a\u2028b"')
    fragment = render_member_fragment(_program([member]), "Demo:separator")
    assert '"a\u2028b"' in fragment
    assert fragment.rstrip().endswith(";")


def test_compiler_mapping_handles_carriage_returns_in_expression(tmp_path: Path) -> None:
    program = _program([_method("work", [
        {"id": "multiline", "op": "local", "type": "int", "name": "total", "value": "1\r + 2\r + 3\r + 4\r + 5"},
        {"id": "bad", "op": "local", "type": "int", "name": "broken", "value": "missingValue"},
    ])])
    source, mapping = render_program(program)
    compiled = _java(tmp_path, source)
    assert compiled.returncode != 0
    match = re.search(r"Demo\.java:(\d+): error", compiled.stderr)
    assert match, compiled.stderr
    assert mapping[int(match[1])] == {"member_id": "work", "node_id": "bad", "slot": "value"}
