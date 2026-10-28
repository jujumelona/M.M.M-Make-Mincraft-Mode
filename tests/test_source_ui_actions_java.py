"""Compile the original typed UI route through a registered source listener."""
import shutil
import subprocess
from copy import deepcopy

import pytest
from test_source_ui_actions import source_sections

from minecraft_mod_ai.source_ui_actions import (
    source_ui_body,
    source_ui_callback,
    source_ui_capabilities,
)
from minecraft_mod_ai.source_ui_actions_java import render_source_ui_action_files
from minecraft_mod_ai.typed_action_java import render_action_dispatch_java
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java


def source_plan():
    sections = source_sections()
    record = sections["resources_and_ui"]["specification"]["interactions"][0]
    caps = source_ui_capabilities({}, sections)
    callback = source_ui_callback(record, package="test")
    inputs = record["server_action"]["inputs"]
    plan = {
        "schema_version": "mmm/typed-plan-ir-v1", "source_sha256": "a" * 64,
        "initialize": [], "functions": [{
            "id": "source_action", "parameters": [
                {"name": "source", "type": "object"},
                *({"name": r["name"], "type": r["type"]} for r in inputs),
            ], "return_type": "int", "covers": ["resources_and_ui.interactions"],
            "body": source_ui_body(record, sections, caps),
        }], "event_bindings": [{
            "event": "command", "function": "source_action", "entry_point_index": 0,
            "config": {"literal": "original_action", "permission_level": 0, "action": {
                "id": callback["action_id"], "label": record["server_action"]["intent"],
                "inputs": inputs,
            }},
        }],
    }
    return plan, caps, callback


def test_renderer_selects_only_the_exact_host_delegate_body():
    plan, _, callback = source_plan()
    files = render_source_ui_action_files(plan, package="test", minecraft_version="26.2")
    assert list(files) == ["src/main/java/test/AuthoredSourceActions.java"]
    assert "// MMM:SOURCE_UI_ACTIONS_OWNER" in next(iter(files.values()))
    assert callback["event_field"] in next(iter(files.values()))
    for body in [[], [{"op": "return", "value": {"op": "literal", "type": "int", "value": 1}}]]:
        other = deepcopy(plan)
        other["functions"][0]["body"] = body
        assert render_source_ui_action_files(other, package="test", minecraft_version="26.2") == {}
    for index in range(3):
        other = deepcopy(plan)
        other["functions"][0]["body"][0]["value"]["args"][index] = {"op": "literal", "type": "string", "value": "forged"}
        assert render_source_ui_action_files(other, package="test", minecraft_version="26.2") == {}


def _event_fixture():
    return {
        "net/fabricmc/fabric/api/event/Event.java": """package net.fabricmc.fabric.api.event;
public interface Event<T> { void register(T listener); T invoker(); }""",
        "net/fabricmc/fabric/api/event/EventFactory.java": """package net.fabricmc.fabric.api.event;
public final class EventFactory {
 public static <T> Event<T> createArrayBacked(Class<T> type, java.util.function.Function<T[],T> factory) {
  return new Event<T>() {
   private final java.util.List<T> listeners = new java.util.ArrayList<>();
   public void register(T listener) { listeners.add(listener); }
   public T invoker() { return factory.apply(listeners.toArray(size -> (T[])java.lang.reflect.Array.newInstance(type,size))); }
  };
 }
}""",
        "net/minecraft/commands/CommandSourceStack.java": """package net.minecraft.commands;
public class CommandSourceStack { public int effect; }""",
    }


def test_registered_callback_executes_once_through_existing_input_codec(tmp_path):
    plan, caps, callback = source_plan()
    files = render_source_ui_action_files(plan, package="test", minecraft_version="26.2")
    runtime = deepcopy(plan)
    runtime["event_bindings"] = []  # Only Minecraft command registration is outside this fixture.
    files.update({
        "test/AuthoredProgram.java": render_typed_plan_java(runtime, package="test", capabilities=caps),
        "test/AuthoredActions.java": render_action_dispatch_java(plan, package="test"),
        **_event_fixture(),
    })
    field, identity = callback["event_field"], callback["action_id"]
    files["test/Harness.java"] = '''package test;
public class Harness {
 static void check(boolean ok) { if (!ok) throw new AssertionError(); }
 static void absent(Runnable task) {
  try { task.run(); throw new AssertionError("accepted listener count"); }
  catch (IllegalStateException expected) {}
 }
 public static void main(String[] args) {
  var source = new net.minecraft.commands.CommandSourceStack();
  var values = java.util.Map.<String,Object>of("amount", 3);
  absent(() -> AuthoredActions.invoke("IDENTITY", source, values));
  AuthoredSourceActions.FIELD.register((actor, amount) -> { actor.effect += amount; return amount + 20; });
  String command = AuthoredActions.commandFromText("IDENTITY", java.util.Map.of("amount", "3"));
  check(command.startsWith("original_action "));
  var tokens = java.util.Arrays.asList(command.split(" ")).subList(1, 2);
  check(AuthoredActions.invokeEncoded("IDENTITY", source, tokens) == 23);
  check(source.effect == 3);
  try { AuthoredActions.invoke("IDENTITY", new Object(), values); throw new AssertionError("bad actor"); }
  catch (IllegalArgumentException expected) {}
  for (Object bad : new Object[]{0, 11, 1.5, true, "3"}) {
   try { AuthoredActions.invoke("IDENTITY", source, java.util.Map.of("amount", bad)); throw new AssertionError("bad input"); }
   catch (IllegalArgumentException expected) {}
  }
  try { AuthoredActions.invoke("IDENTITY", source, java.util.Map.of("amount",3,"extra",1)); throw new AssertionError("extra key"); }
  catch (IllegalArgumentException expected) {}
  check(source.effect == 3);
  AuthoredSourceActions.FIELD.register((actor, amount) -> { actor.effect++; return 99; });
  absent(() -> AuthoredActions.invoke("IDENTITY", source, values));
  check(source.effect == 3);
 }
}'''.replace("IDENTITY", identity).replace("FIELD", field)
    for name, source in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    java = shutil.which("java")
    assert java, "Java runtime with module compiler is required"
    compiled = subprocess.run([java, "-m", "jdk.compiler/com.sun.tools.javac.Main", "-d", str(tmp_path),
                               *map(str, tmp_path.rglob("*.java"))], check=False, text=True, capture_output=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([java, "-cp", str(tmp_path), "test.Harness"], check=False, text=True, capture_output=True, timeout=30)
    assert executed.returncode == 0, executed.stderr


def test_production_renders_and_materializes_source_file_with_strict_ownership(tmp_path):
    from test_typed_plan_production import _project

    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.typed_plan_production import (
        generate_typed_plan_module,
        validate_typed_plan_generation_contract,
    )

    plan, caps, _ = source_plan()
    package = "ai.minecraft.typedtest"
    module = ProductionModule(module_id="authored_typed_plan", kind="typed_host", config={
        "typed_plan_ir": plan, "typed_plan_package": package,
        "typed_plan_capabilities": caps, "minecraft_version": "26.2",
    }, required_gates=("target_compile",))
    receipt = validate_typed_plan_generation_contract(module, package_name=package, mod_id="typedtest")
    assert any("AuthoredSourceActions.java" in str(value) for value in receipt.values())
    root = _project(tmp_path)
    target = root / "src/main/java/ai/minecraft/typedtest/AuthoredSourceActions.java"
    target.write_text("package ai.minecraft.typedtest; public class AuthoredSourceActions {}")
    with pytest.raises(ValueError, match="OWNERSHIP_CONFLICT"):
        generate_typed_plan_module(root, module=module)
    target.write_text("// MMM:SOURCE_UI_ACTIONS_OWNER\nold")
    generate_typed_plan_module(root, module=module)
    assert "EventFactory.createArrayBacked" in target.read_text()
