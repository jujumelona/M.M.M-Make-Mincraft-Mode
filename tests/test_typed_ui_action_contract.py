"""UI values must reach their typed server function without prose dispatch."""

from copy import deepcopy
import shutil
import subprocess
import os

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
    assert "context.getSource().getPlayerOrException();" in source
    assert "context.getSource(), java.util.List.of(" in source


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


def test_complete_action_contract_must_fit_the_command_packet():
    from minecraft_mod_ai.typed_action_contract import validate_action_contract

    contract = action_plan()["event_bindings"][0]["config"]["action"]
    contract["inputs"] = [
        {
            "name": f"text_{i}",
            "type": "string",
            "label": "Text",
            "max_length": 4096,
            "choices": [],
        }
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="ACTION_COMMAND_TOO_LARGE"):
        validate_action_contract(contract)


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
  try { AuthoredActions.command("example", java.util.Map.of("amount", 3, "label", "\\ud800")); throw new AssertionError(); }
  catch (IllegalArgumentException expected) {}
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


def test_client_form_is_generated_from_the_same_action_contract():
    from minecraft_mod_ai.typed_action_client import render_action_client_java

    source = render_action_client_java(
        action_plan(), package="test", mod_id="testmod", minecraft_version="26.2"
    )
    assert "AuthoredActions.commandFromText(" in source
    assert "sendCommand(command)" in source
    assert "getValue()" in source
    assert '"amount"' in source and '"label"' in source
    assert "AuthoredProgram.invoke" not in source
    assert "AuthoredStateModel" not in source
    assert "gui.setScreen(" in source
    assert "extractRenderState(" in source


def test_production_writes_both_sides_and_registers_the_client_form(tmp_path):
    import json
    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.generator import FabricProjectGenerator
    from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
    from minecraft_mod_ai.typed_plan_production import (
        generate_typed_plan_module,
        validate_typed_plan_generation_contract,
    )
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS

    FabricProjectGenerator().generate(
        ModSpec(
            mod_id="actiontest",
            mod_name="Actions",
            package_name="test.actions",
            version="1.0.0",
            summary="Action contract test",
            contents=(
                ContentSpec(
                    content_id="marker",
                    kind=ContentKind.ITEM,
                    display_name_en="Marker",
                    display_name_ko="표시",
                ),
            ),
        ),
        tmp_path,
    )
    state = {
        "constraint_evidence_refs": [],
        "specification": {
            **{key: [] for key in DETAIL_RECORDS["state_model"]},
            "inapplicable_concerns": [],
            "variables": [
                {
                    "name": "amount",
                    "type": "int",
                    "default": "0",
                    "owner": "player",
                    "domain": "count",
                    "unit": "count",
                },
                {
                    "name": "label",
                    "type": "string",
                    "default": "",
                    "owner": "player",
                    "domain": "text",
                    "unit": "text",
                },
            ],
        },
    }
    module = ProductionModule(
        "authored_typed_plan",
        "typed_host",
        {
            "typed_plan_ir": action_plan(),
            "typed_plan_package": "test.actions",
            "minecraft_version": "26.2",
            "typed_plan_state_section": state,
        },
    )
    dry = validate_typed_plan_generation_contract(
        module, package_name="test.actions", mod_id="actiontest"
    )
    receipt = generate_typed_plan_module(tmp_path, module=module)
    client = "src/client/java/test/actions/AuthoredActionsClient.java"
    assert client in dry["paths"] and client in receipt["touched_paths"]
    assert "src/main/java/test/actions/AuthoredActions.java" in receipt["touched_paths"]
    fabric = json.loads((tmp_path / "src/main/resources/fabric.mod.json").read_text())
    assert "test.actions.AuthoredActionsClient" in fabric["entrypoints"]["client"]
    common = (tmp_path / "src/main/java/test/actions/AuthoredActions.java").read_text()
    assert "net.minecraft.client" not in common


@pytest.mark.skipif(
    not os.environ.get("MMM_ACTION_COMPILE_CLASSPATH"),
    reason="Exact Minecraft/Fabric SDK classpath required",
)
def test_generated_client_and_server_compile_against_actual_minecraft_26_2(tmp_path):
    from minecraft_mod_ai.typed_action_client import render_action_client_java
    from minecraft_mod_ai.typed_action_java import render_action_dispatch_java
    from minecraft_mod_ai.production_state_compiler import render_production_state_java
    from minecraft_mod_ai.typed_host_capabilities import (
        render_typed_host_capabilities_java,
    )
    from minecraft_mod_ai.typed_plan_production import (
        _state_lifecycle_files,
        _state_persistence_java,
    )
    from minecraft_mod_ai.system_templates_common import _persistent_store_java
    from minecraft_mod_ai.typed_state_context import bind_player_state_context
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS

    p = action_plan()
    state = {
        "specification": {
            **{concern: [] for concern in DETAIL_RECORDS["state_model"]},
            "inapplicable_concerns": [],
            "variables": [
                {
                    "name": name,
                    "type": kind,
                    "owner": "player",
                    "default": default,
                    "unit": "value",
                    "domain": "test",
                }
                for name, kind, default in (
                    ("amount", "int", "0"),
                    ("label", "string", ""),
                )
            ],
        },
        "constraint_evidence_refs": [],
    }
    p, caps = bind_player_state_context(p, state, {})
    sources = {
        "AuthoredProgram.java": render_typed_plan_java(
            p, package="test", minecraft_version="26.2", capabilities=caps
        ),
        "AuthoredActions.java": render_action_dispatch_java(p, package="test"),
        "AuthoredActionsClient.java": render_action_client_java(
            p, package="test", mod_id="testmod", minecraft_version="26.2"
        ),
        "AuthoredStateModel.java": render_production_state_java(
            state, package_name="test"
        ),
        "AuthoredHostCapabilities.java": render_typed_host_capabilities_java(
            "test", minecraft_version="26.2"
        ),
        "AuthoredStateLifecycle.java": next(
            iter(_state_lifecycle_files("test").values())
        ),
        "AuthoredStatePersistence.java": _state_persistence_java(
            "test",
            "actions",
            ("amount", "label"),
            {
                "schema_version": "2",
                "transfer_on_respawn": True,
                "migrations": [
                    {
                        "from_version": "1",
                        "to_version": "2",
                        "operation": "set_default",
                        "destination_key": "amount",
                        "value": 2,
                    }
                ],
            },
            minecraft_version="26.2",
        ),
        "system/MmmPersistentStore.java": _persistent_store_java(
            "test", "testmod", minecraft_version="26.2"
        ),
    }
    for name, source in sources.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    compiled = subprocess.run(
        [
            shutil.which("javac"),
            "-proc:none",
            "-cp",
            os.environ["MMM_ACTION_COMPILE_CLASSPATH"],
            *map(str, tmp_path.rglob("*.java")),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
