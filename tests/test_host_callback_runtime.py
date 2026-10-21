"""Execute host-compiled non-void callbacks, including concrete result subtypes."""
import shutil
import subprocess

import pytest

from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


def test_nonvoid_callback_executes_effect_only_between_predicate_and_success(tmp_path):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required for compiled callback behavior probe")
    event = "net.fabricmc.fabric.api.event.Event"
    hook = "net.fabricmc.fabric.api.event.player.TestHook"
    symbols = {
        "register": {"kind": "METHOD", "owner": event, "name": "register",
                     "access": 1, "parameters": ["java.lang.Object"], "return_type": "void"},
        hook: {"kind": "CLASS", "owner": hook, "access": 1537},
        "hook.field": {"kind": "FIELD", "owner": hook, "name": "EVENT", "access": 25,
                       "return_type": event, "generic_return_type": event + "<" + hook + ">"},
        "hook.sam": {"kind": "METHOD", "owner": hook, "name": "run", "access": 1025,
                     "parameters": ["example.Target"], "return_type": "example.Result"},
        "target": {"kind": "CLASS", "owner": "example.Target", "access": 1},
        "result": {"kind": "CLASS", "owner": "example.Result", "access": 1537},
        "success": {"kind": "CLASS", "owner": "example.Success", "access": 17,
                    "interfaces": ["example.Result"]},
        "pass": {"kind": "FIELD", "owner": "example.Result", "name": "PASS",
                 "access": 25, "return_type": "example.Result"},
        "ok": {"kind": "FIELD", "owner": "example.Result", "name": "SUCCESS",
               "access": 25, "return_type": "example.Success"},
    }

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name == "select_verified_fabric_event":
                return {"event_binding": hook + "#EVENT"}
            if "_predicate_" in data_name:
                return {"expression": "target.active"}
            if "_operation_0_assign_" in data_name:
                return {"target": "target.effects", "operator": "+=", "value": "1"}
            if "_operation_0_" in data_name:
                return {"action": "assign"}
            if "_operation_1_" in data_name:
                return {"action": "complete"}
            if data_name.startswith("select_callback_result_"):
                return {"expression": "example.Result." + (
                    "PASS" if data_name.endswith("unhandled") else "SUCCESS")}
            raise AssertionError("Unexpected Java-source authoring request: " + data_name)

    source = compose_first_pass_java(Router(), {
        "language": "java", "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
        "bindings": {"request_feature_entrypoint": "example.Bridge"},
    }, [], construction_context={"symbols": symbols}, behaviors=[{
        "behavior_id": "generic.effect", "intent": "Increment effects when active",
        "trigger": "target is active", "effects": "Increment target effects once"}])
    files = {
        "example/Bridge.java": source,
        "example/Target.java": "package example; public class Target { public boolean active; public int effects; }",
        "example/Result.java": "package example; public interface Result { Result PASS=new Result(){}; Success SUCCESS=new Success(); }",
        "example/Success.java": "package example; public final class Success implements Result {}",
        "net/fabricmc/api/ModInitializer.java": "package net.fabricmc.api; public interface ModInitializer { void onInitialize(); }",
        "net/fabricmc/fabric/api/event/Event.java": (
            "package net.fabricmc.fabric.api.event; public class Event<T> { public T listener; "
            "public void register(T listener) { this.listener=listener; }}"),
        "net/fabricmc/fabric/api/event/player/TestHook.java": (
            "package net.fabricmc.fabric.api.event.player; public interface TestHook { "
            "net.fabricmc.fabric.api.event.Event<TestHook> EVENT=new net.fabricmc.fabric.api.event.Event<>(); "
            "example.Result run(example.Target target); }"),
        "example/Probe.java": (
            "package example; import net.fabricmc.fabric.api.event.player.TestHook; "
            "public class Probe { public static void main(String[] args) { "
            "new Bridge().onInitialize(); Target t=new Target(); "
            "if(TestHook.EVENT.listener.run(t)!=Result.PASS || t.effects!=0) throw new AssertionError(); "
            "t.active=true; if(TestHook.EVENT.listener.run(t)!=Result.SUCCESS || t.effects!=1) throw new AssertionError(); "
            "t.active=false; if(TestHook.EVENT.listener.run(t)!=Result.PASS || t.effects!=1) throw new AssertionError(); }}"),
    }
    paths = []
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        paths.append(str(path))
    classes = tmp_path / "classes"
    compiled = subprocess.run([javac, "-proc:none", "-d", str(classes), *paths],
                              capture_output=True, text=True, timeout=60, check=False)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([java, "-cp", str(classes), "example.Probe"],
                              capture_output=True, text=True, timeout=30, check=False)
    assert executed.returncode == 0, executed.stderr
