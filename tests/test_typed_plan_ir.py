"""The planner boundary rejects invalid operations before Java generation."""

import copy
import hashlib
import shutil
import subprocess

import pytest

from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java


def literal(value, type_="int"):
    return {"op": "literal", "type": type_, "value": value}


def ref(name):
    return {"op": "ref", "name": name}


def binary(operator, left, right):
    return {"op": "binary", "operator": operator, "left": left, "right": right}


def plan_for(body, return_type="int", parameters=None):
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(b"authored design").hexdigest(),
        "functions": [{"id": "compute", "parameters": parameters or [],
                       "return_type": return_type, "body": body,
                       "covers": ["requirements:0"]}],
        "initialize": [],
    }


def run_java(tmp_path, plan, harness, *, capabilities=None, state=False):
    if not shutil.which("javac") or not shutil.which("java"):
        pytest.skip("JDK is required for executable compiler fixtures")
    source = render_typed_plan_java(plan, package="fixture", capabilities=capabilities)
    (tmp_path / "AuthoredProgram.java").write_text(source, encoding="utf-8")
    (tmp_path / "Harness.java").write_text(
        "package fixture; public class Harness { public static void main(String[] args) {"
        + harness + "}}", encoding="utf-8",
    )
    sources = ["AuthoredProgram.java", "Harness.java"]
    if state:
        (tmp_path / "AuthoredStateModel.java").write_text(
            "package fixture; public class AuthoredStateModel {"
            "private static final java.util.Map<String,Object> values = new java.util.HashMap<>();"
            "public static Object getState(String name, java.util.Map<String,Object> context) {"
            'return values.get(context.get("player") + ":" + name);}'
            "public static void setState(String name,Object value,java.util.Map<String,Object> context) {"
            'values.put(context.get("player") + ":" + name,value);}}', encoding="utf-8",
        )
        sources.append("AuthoredStateModel.java")
    compiled = subprocess.run(["javac", "-encoding", "UTF-8", "-d", str(tmp_path), *sources],
                              cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(["java", "-cp", str(tmp_path), "fixture.Harness"],
                              capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stderr
    return executed.stdout


def test_validation_is_pure_and_generation_is_deterministic():
    plan = plan_for([{"op": "return", "value": literal(42)}])
    original = copy.deepcopy(plan)
    assert validate_typed_plan_ir(plan) == original
    assert plan == original
    assert render_typed_plan_java(plan, package="fixture") == render_typed_plan_java(plan, package="fixture")


@pytest.mark.parametrize("body,error", [
    ([{"op": "raw_java", "source": "System.exit(0);"}], "unsupported"),
    ([{"op": "return", "value": ref("missing")}], "unknown"),
    ([{"op": "return", "value": literal(True, "boolean")}], "type"),
    ([{"op": "let", "name": "x", "type": "int", "value": literal(1)}], "return"),
    ([{"op": "return", "value": literal(1)}, {"op": "return", "value": literal(2)}], "unreachable"),
    ([{"op": "return", "value": {"op": "capability", "id": "minecraft.teleport", "args": []}}], "capability"),
    ([{"op": "return", "value": {"op": "call", "function": "compute", "args": []}}], "recursive"),
    ([{"op": "return", "value": literal(2147483648)}], "range"),
])
def test_invalid_ir_fails_closed(body, error):
    with pytest.raises(ValueError, match=error):
        validate_typed_plan_ir(plan_for(body))


def test_rejects_unknown_fields_and_invalid_scopes():
    plan = plan_for([{"op": "return", "value": literal(1)}])
    plan["java_source"] = "public class Bad {}"
    with pytest.raises(ValueError, match="unknown"):
        validate_typed_plan_ir(plan)
    plan = plan_for([
        {"op": "if", "condition": literal(True, "boolean"),
         "then": [{"op": "let", "name": "x", "type": "int", "value": literal(1)}], "else": []},
        {"op": "return", "value": ref("x")},
    ])
    with pytest.raises(ValueError, match="unknown"):
        validate_typed_plan_ir(plan)


def test_compiled_control_flow_dispatch_and_strict_arguments(tmp_path):
    plan = plan_for([
        {"op": "let", "name": "sum", "type": "int", "value": literal(0)},
        {"op": "let", "name": "i", "type": "int", "value": literal(0)},
        {"op": "while", "condition": binary("<", ref("i"), ref("limit")), "max_iterations": 10,
         "body": [{"op": "set", "name": "sum", "value": binary("+", ref("sum"), ref("i"))},
                  {"op": "set", "name": "i", "value": binary("+", ref("i"), literal(1))}]},
        {"op": "return", "value": ref("sum")},
    ], parameters=[{"name": "limit", "type": "int"}])
    assert run_java(tmp_path, plan, '''
        System.out.print(AuthoredProgram.invoke("compute", java.util.Map.of("limit", 5)));
        try { AuthoredProgram.invoke("compute", java.util.Map.of("limit", "5")); throw new AssertionError(); }
        catch (IllegalArgumentException expected) {}
        try { AuthoredProgram.invoke("compute", java.util.Map.of("limit", 5, "extra", 1)); throw new AssertionError(); }
        catch (IllegalArgumentException expected) {}
        try { AuthoredProgram.invoke("missing", java.util.Map.of()); throw new AssertionError(); }
        catch (IllegalArgumentException expected) {}
        try { AuthoredProgram.invoke("compute", java.util.Map.of("limit", 12)); throw new AssertionError(); }
        catch (IllegalStateException expected) {}
    ''') == "10"


def test_compiled_collections_calls_and_host_capability(tmp_path):
    plan = plan_for([
        {"op": "let", "name": "total", "type": "int", "value": literal(0)},
        {"op": "foreach", "name": "item", "type": "int", "collection": {"op": "list", "items": [literal(-2), literal(3)]},
         "body": [{"op": "set", "name": "total", "value": binary("+", ref("total"),
             {"op": "capability", "id": "math.abs", "args": [ref("item")]})}]},
        {"op": "return", "value": {"op": "call", "function": "double", "args": [ref("total")]}},
    ])
    plan["functions"].append({"id": "double", "parameters": [{"name": "input", "type": "int"}],
                              "return_type": "int", "covers": [], "body": [
                                  {"op": "return", "value": binary("*", ref("input"), literal(2))}]})
    capabilities = {"math.abs": {"owner": "java.lang.Math", "method": "abs", "parameters": ["int"], "return_type": "int"}}
    assert run_java(tmp_path, plan,
        'System.out.print(AuthoredProgram.invoke("compute", java.util.Map.of()));', capabilities=capabilities) == "10"


def test_compiled_state_operations_bind_host_context(tmp_path):
    context = {"op": "map", "entries": [{"key": literal("player", "string"), "value": literal("alex", "string")}]}
    plan = plan_for([
        {"op": "state_set", "key": literal("coins", "string"), "value": literal(9), "context": context},
        {"op": "return", "value": {"op": "state_get", "key": literal("coins", "string"), "type": "int", "context": context}},
    ])
    assert run_java(tmp_path, plan, 'System.out.print(AuthoredProgram.invoke("compute", java.util.Map.of()));', state=True) == "9"


def test_compiled_string_literals_cannot_inject_source(tmp_path):
    value = '\\u0022; System.exit(9); //\n한글\x00'
    plan = plan_for([{"op": "return", "value": literal(value, "string")}], "string")
    assert run_java(tmp_path, plan, 'System.out.print(AuthoredProgram.invoke("compute", java.util.Map.of()));') == value
