"""UI values must reach their typed server function without prose dispatch."""

from copy import deepcopy
import shutil
import subprocess

import pytest

from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java


def action_plan():
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "a" * 64,
        "initialize": [],
        "functions": [
            {
                "id": "apply_amount",
                "parameters": [
                    {"name": "source", "type": "object"},
                    {"name": "amount", "type": "int"},
                    {"name": "label", "type": "string"},
                ],
                "return_type": "int",
                "covers": ["resources_and_ui.interactions"],
                "body": [
                    {
                        "op": "state_set",
                        "key": {"op": "literal", "type": "string", "value": "amount"},
                        "value": {"op": "ref", "name": "amount"},
                        "context": {"op": "map", "entries": []},
                    },
                    {
                        "op": "state_set",
                        "key": {"op": "literal", "type": "string", "value": "label"},
                        "value": {"op": "ref", "name": "label"},
                        "context": {"op": "map", "entries": []},
                    },
                    {
                        "op": "return",
                        "value": {"op": "literal", "type": "int", "value": 1},
                    },
                ],
            }
        ],
        "event_bindings": [
            {
                "event": "command",
                "function": "apply_amount",
                "entry_point_index": 0,
                "config": {
                    "literal": "mmm_ui_example",
                    "permission_level": 0,
                    "action": {
                        "id": "example",
                        "label": "Change values",
                        "inputs": [
                            {
                                "name": "amount",
                                "type": "int",
                                "label": "Amount",
                                "minimum": 1,
                                "maximum": 10,
                            },
                            {
                                "name": "label",
                                "type": "string",
                                "label": "Label",
                                "max_length": 40,
                                "choices": [],
                            },
                        ],
                    },
                },
            }
        ],
    }


def test_typed_action_signature_includes_real_input_arguments():
    plan = action_plan()
    normalized = validate_typed_plan_ir(plan)
    assert (
        normalized["event_bindings"][0]["config"]["action"]["inputs"][0]["type"]
        == "int"
    )
    source = render_typed_plan_java(plan, package="test", minecraft_version="26.2")
    assert "StringArgumentType.word()" in source
    assert "AuthoredActions.submitEncoded(" in source
    assert '"example"' in source
    assert "ui_action_dispatch" not in source


@pytest.mark.parametrize("mutation", ["type", "name", "range", "duplicate", "object"])
def test_action_contract_rejects_unbound_or_invalid_inputs(mutation):
    plan = action_plan()
    inputs = plan["event_bindings"][0]["config"]["action"]["inputs"]
    if mutation == "type":
        inputs[0]["type"] = "long"
    if mutation == "name":
        inputs[0]["name"] = "invented"
    if mutation == "range":
        inputs[0]["minimum"] = 11
    if mutation == "duplicate":
        inputs[1]["name"] = "amount"
    if mutation == "object":
        inputs[0]["type"] = "object"
    with pytest.raises(ValueError, match="ACTION_"):
        validate_typed_plan_ir(plan)


def test_action_ids_and_transport_literals_are_unique():
    plan = action_plan()
    second = deepcopy(plan["event_bindings"][0])
    second["entry_point_index"] = 1
    plan["event_bindings"].append(second)
    with pytest.raises(ValueError, match="ACTION_.*DUPLICATE"):
        validate_typed_plan_ir(plan)


def test_generated_action_transport_executes_distinct_inputs_and_rejects_invalid_before_writes(
    tmp_path,
):
    from minecraft_mod_ai.typed_action_java import render_action_dispatch_java

    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")
    plan = action_plan()
    adapter = render_action_dispatch_java(plan, package="test")
    # The adapter is exactly the production transport owner. Its target is the
    # same rendered typed function; only Minecraft's registration is omitted.
    function_plan = deepcopy(plan)
    function_plan["event_bindings"] = []
    program = render_typed_plan_java(function_plan, package="test")
    sources = {
        "AuthoredActions.java": adapter,
        "AuthoredProgram.java": program,
        "AuthoredStateModel.java": """package test;
public class AuthoredStateModel {
 public static int writes;
 public static final java.util.Map<String,Object> values = new java.util.HashMap<>();
 public static Object getState(String k, java.util.Map<String,Object> c) { return values.get(k); }
 public static void setState(String k, Object v, java.util.Map<String,Object> c) { writes++; values.put(k,v); }
}""",
        "Harness.java": """package test;
public class Harness {
 static void check(boolean value) { if (!value) throw new AssertionError(); }
 public static void main(String[] args) {
  Object actor = new Object();
  String text = "한글 \\\" / ; \\n";
  var values = java.util.Map.<String,Object>of("amount", 3, "label", text);
  String command = AuthoredActions.command("example", values);
  check(command.split(" ").length == 3);
  var tokens = java.util.Arrays.asList(command.split(" ")).subList(1, 3);
  check(AuthoredActions.invokeEncoded("example", actor, tokens) == 1);
  check(AuthoredStateModel.values.get("amount").equals(3));
  check(AuthoredStateModel.values.get("label").equals(text));
  AuthoredActions.invoke("example", actor, java.util.Map.of("amount", 7, "label", ""));
  check(AuthoredStateModel.values.get("amount").equals(7));
  check(AuthoredStateModel.values.get("label").equals(""));
  int before = AuthoredStateModel.writes;
  for (Object amount : new Object[]{0, 11, 1.5, Double.NaN, 4294967297L, true}) {
   try { AuthoredActions.invoke("example", actor, java.util.Map.of("amount", amount, "label", "x"));
         throw new AssertionError("invalid amount accepted: " + amount); }
   catch (IllegalArgumentException expected) { }
  }
  try { AuthoredActions.invokeEncoded("example", actor, java.util.List.of("x%%%", "x"));
        throw new AssertionError("bad encoding accepted"); } catch (IllegalArgumentException expected) { }
  try { AuthoredActions.invoke("example", actor, java.util.Map.of("amount", 2, "label", "x", "extra", 9));
        throw new AssertionError("extra key accepted"); } catch (IllegalArgumentException expected) { }
  check(AuthoredStateModel.writes == before);
  check(AuthoredStateModel.values.get("amount").equals(7));
 }
}""",
    }
    folder = tmp_path / "test"
    folder.mkdir()
    for name, source in sources.items():
        (folder / name).write_text(source)
    compiled = subprocess.run(
        [javac, *map(str, folder.glob("*.java"))], capture_output=True, text=True
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [java, "-cp", str(tmp_path), "test.Harness"], capture_output=True, text=True
    )
    assert executed.returncode == 0, executed.stderr
