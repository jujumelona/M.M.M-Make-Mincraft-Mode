"""Compile and execute host rules through repeated observations and contexts."""
from collections import deque
import importlib
import shutil
import subprocess

import pytest


LEVEL = "net.minecraft.server.level.ServerLevel"
TICKS = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"
EVENT = "net.fabricmc.fabric.api.event.Event"


def index():
    return {
        "java.lang.String": {"kind": "CLASS", "owner": "java.lang.String", "access": 17},
        LEVEL: {"kind": "CLASS", "owner": LEVEL, "access": 1},
        TICKS: {"kind": "CLASS", "owner": TICKS, "access": 17},
        TICKS + ".EndTick": {"kind": "CLASS", "owner": TICKS + ".EndTick", "access": 1545},
        EVENT: {"kind": "CLASS", "owner": EVENT, "access": 1},
        "register": {"kind": "METHOD", "owner": EVENT, "name": "register", "access": 1,
                     "parameters": ["java.lang.Object"], "return_type": "void"},
        "event": {"kind": "FIELD", "owner": TICKS, "name": "END_LEVEL_TICK", "access": 25,
                  "static": True, "return_type": EVENT, "generic_return_type": EVENT + "<" + TICKS + ".EndTick>"},
        "sam": {"kind": "METHOD", "owner": TICKS + ".EndTick", "name": "onEndTick", "access": 1025,
                "parameters": [LEVEL], "return_type": "void"},
        "ready": {"kind": "METHOD", "owner": LEVEL, "name": "isReady", "access": 1,
                  "parameters": [], "return_type": "boolean"},
        "clock": {"kind": "METHOD", "owner": LEVEL, "name": "getGameTime", "access": 1,
                  "parameters": [], "return_type": "long"},
        "effects": {"kind": "FIELD", "owner": LEVEL, "name": "effects", "access": 1, "return_type": "int"},
    }


def formula_replies(mode="changed", first="baseline"):
    replies = [
        ("rule_observation", {"observation": "world_state"}),
        ("rule_activation", {"mode": mode}),
        ("rule_value_operation", {"operation": "call"}),
        ("rule_api_query", {"query": "isReady"}),
        ("rule_api_page", {"symbol": "s0"}),
        ("rule_operand", {"value_ref": "v0"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v1"}),
    ]
    field_id = "v2"
    if mode == "changed":
        replies += [
            ("rule_value_operation", {"operation": "call"}),
            ("rule_api_query", {"query": "getGameTime"}),
            ("rule_api_page", {"symbol": "s0"}),
            ("rule_operand", {"value_ref": "v0"}),
            ("rule_value_operation", {"operation": "literal"}),
            ("rule_literal_type", {"type": "long"}),
            ("rule_literal", {"value": 10}),
            ("rule_value_operation", {"operation": "binary"}),
            ("rule_binary_operator", {"operator": "/"}),
            ("rule_operand", {"value_ref": "v2"}),
            ("rule_operand", {"value_ref": "v3"}),
            ("rule_value_operation", {"operation": "use"}),
            ("rule_operand", {"value_ref": "v4"}),
        ]
        field_id = "v5"
    if mode != "level":
        replies += [("rule_tracking_context", {"value_ref": "v0"}),
                    ("rule_first_observation", {"policy": first})]
    literal_id = "v" + str(int(field_id[1:]) + 1)
    replies += [
        ("rule_effect_operation", {"operation": "assign"}),
        ("rule_value_operation", {"operation": "field"}),
        ("rule_api_query", {"query": "effects"}),
        ("rule_api_page", {"symbol": "s0"}),
        ("rule_operand", {"value_ref": "v0"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": field_id}),
        ("rule_operand", {"value_ref": "build"}),
        ("rule_value_operation", {"operation": "literal"}),
        ("rule_literal_type", {"type": "int"}),
        ("rule_literal", {"value": 1}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": literal_id}),
        ("rule_assignment_operator", {"operator": "+="}),
        ("rule_effect_operation", {"operation": "complete"}),
    ]
    return replies


class Router:
    def __init__(self, mode="changed", first="baseline"):
        self.replies = deque(formula_replies(mode, first))

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        expected, reply = self.replies.popleft()
        assert data_name == expected
        return reply

    def generate_java_source(self, *args, **kwargs):
        raise AssertionError("Rule production must never request Java source")


def compose(tmp_path, mode="changed", first="baseline"):
    module = importlib.import_module("minecraft_mod_ai.host_rule_feature")
    router = Router(mode, first)
    source = module.compose_rule_feature(router, {
        "language": "java", "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
        "bindings": {"request_feature_entrypoint": "example.Bridge"},
    }, {"symbols": index(), "project_root": str(tmp_path)}, package="example", name="Bridge",
        behaviors=[{"behavior_id": "world-boundary", "intent": "Observe a clock boundary",
                    "trigger": "Game time reaches a boundary", "effects": "Increment effects",
                    "state_reads": [], "state_writes": []}], state_access=[])
    assert not router.replies
    return source


def compile_run(tmp_path, source, probe, *, callback_return="void"):
    assert shutil.which("javac"), "JDK is required; do not skip execution proof"
    files = {
        "example/Bridge.java": source,
        "net/fabricmc/api/ModInitializer.java": "package net.fabricmc.api; public interface ModInitializer { void onInitialize(); }",
        "net/fabricmc/fabric/api/event/Event.java": "package net.fabricmc.fabric.api.event; public class Event<T> { public T listener; public void register(T t) { listener=t; } }",
        "net/fabricmc/fabric/api/event/lifecycle/v1/ServerTickEvents.java":
            "package net.fabricmc.fabric.api.event.lifecycle.v1; import net.fabricmc.fabric.api.event.Event; "
            "public class ServerTickEvents { public interface EndTick { " + callback_return + " onEndTick(net.minecraft.server.level.ServerLevel w); } "
            "public static final Event<EndTick> END_LEVEL_TICK=new Event<>(); }",
        "net/minecraft/server/level/ServerLevel.java": "package net.minecraft.server.level; public class ServerLevel { "
            "public boolean ready=true; public long time; public int effects; public int weaponInventory; "
            "public long getGameTime() { return time; } public boolean isReady() { return ready; } " +
            "public int getInventory(){return weaponInventory;} }",
        "example/Probe.java": "package example; import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents; "
            "import net.minecraft.server.level.ServerLevel; public class Probe { "
            "static void tick(ServerLevel w,long t) { w.time=t; ServerTickEvents.END_LEVEL_TICK.listener.onEndTick(w); } "
            "static void check(boolean b) { if(!b) throw new AssertionError(); } "
            "public static void main(String[] args) { new Bridge().onInitialize(); " + probe + " } }",
    }
    paths = []
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        paths.append(str(p))
    classes = tmp_path / "classes"
    result = subprocess.run(["javac", "-proc:none", "-d", str(classes), *paths], text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(["java", "-cp", str(classes), "example.Probe"], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_changed_clock_is_scoped_and_does_not_repeat_effects(tmp_path):
    source = compose(tmp_path)
    from minecraft_mod_ai.construction_file_author import _check_entrypoint
    import json
    _check_entrypoint(source, {"bindings": {"request_feature_entrypoint": "example.Bridge",
        "construction_behaviors_json": json.dumps([{"behavior_id": "world-boundary"}])}})
    compile_run(tmp_path, source, """
        ServerLevel a=new ServerLevel(), b=new ServerLevel();
        tick(a,0); tick(a,1); tick(a,9); check(a.effects==0);
        tick(a,10); tick(a,11); tick(a,19); check(a.effects==1);
        tick(a,20); check(a.effects==2);
        tick(b,20); check(b.effects==0); tick(b,30); check(b.effects==1);
        a.ready=false; tick(a,30); check(a.effects==2);
        a.ready=true; tick(a,31); check(a.effects==2);
        tick(a,40); check(a.effects==3);
    """)
    assert list((tmp_path / ".mmm/construction_rule_ir").glob("*.json"))


@pytest.mark.parametrize("mode", ["level", "rising"])
def test_gate_and_rising_edge_execute_real_effects_with_explicit_first_policy(tmp_path, mode):
    source = compose(tmp_path, mode, "fire")
    expected = 2 if mode == "level" else 1
    compile_run(tmp_path, source, f"""
        ServerLevel a=new ServerLevel(); a.ready=false; tick(a,0); check(a.effects==0);
        a.ready=true; tick(a,1); tick(a,2); check(a.effects=={expected});
        a.ready=false; tick(a,3); a.ready=true; tick(a,4); check(a.effects=={expected + 1});
    """)


@pytest.mark.parametrize("compiler_failure", [False, True])
def test_production_construction_uses_rules_and_never_source_repair(tmp_path, monkeypatch, compiler_failure):
    from minecraft_mod_ai.construction_host_source import compose_host_java_source

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy API/source authoring must not run for production rules")

    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._evidence", forbidden)
    monkeypatch.setattr("minecraft_mod_ai.first_pass_java_source.compose_first_pass_java", forbidden)
    monkeypatch.setattr("minecraft_mod_ai.construction_host_member_repair.repair_compiler_member", forbidden)
    monkeypatch.setattr("minecraft_mod_ai.construction_coverage.record_authored_source_calls", lambda *a, **k: None)

    def verify(source, *args, **kwargs):
        if compiler_failure:
            raise RuntimeError("target-compiler-diagnostic")
        compile_run(tmp_path, source, "ServerLevel w=new ServerLevel(); tick(w,0); tick(w,10); check(w.effects==1);")

    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._verify_full_source", verify)
    context = {"symbols": index(), "project_root": str(tmp_path),
               "target": {"loader": "fabric"}, "java_classpath": [str(tmp_path)],
               "prerequisite_hashes": {}}
    spec = {"language": "java", "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
            "bindings": {"request_feature_entrypoint": "example.Bridge"}}
    def call():
        return compose_host_java_source(Router(), spec, context, package="example", name="Bridge", owner="example.Bridge",
            behaviors=[{"behavior_id": "world-boundary", "trigger": "Game time reaches a boundary",
                        "intent": "Observe a boundary", "effects": "Increment effects", "state_reads": [], "state_writes": []}])
    if compiler_failure:
        with pytest.raises(RuntimeError, match="target-compiler-diagnostic"):
            call()
        assert not list((tmp_path / ".mmm/construction_host_files").glob("*.json"))
    else:
        source = call()
        assert "handleRule" in source
        assert list((tmp_path / ".mmm/construction_host_files").glob("*.json"))


def test_tracking_context_excludes_scalar_temporaries():
    from minecraft_mod_ai.host_rule_feature import _tracking
    from minecraft_mod_ai.host_rule_program import RuleProgram
    p = RuleProgram(index(), [{"type": LEVEL, "name": "arg0"}], {},
        owner="example.Bridge", target_path="src/main/java/example/Bridge.java")
    gate = p.value("literal", type="boolean", value=True)
    temporary = p.value("literal", type="java.lang.String", value="all-worlds")

    class Author:
        def ask(self, name, purpose, properties, **context):
            if name == "rule_tracking_context":
                assert temporary not in properties["value_ref"]["enum"]
                return {"value_ref": p.inputs[0]}
            return {"policy": "baseline"}

    _tracking(Author(), p, gate, "rising")


def test_initial_true_with_baseline_requires_a_new_rising_edge(tmp_path):
    source = compose(tmp_path, "rising", "baseline")
    compile_run(tmp_path, source, """
        ServerLevel w=new ServerLevel(); tick(w,0); tick(w,1); check(w.effects==0);
        w.ready=false; tick(w,2); w.ready=true; tick(w,3); check(w.effects==1);
    """)


def test_nonvoid_callback_returns_handled_only_after_matching_effects(tmp_path):
    from minecraft_mod_ai.host_rule_feature import compose_rule_feature
    api = index()
    api["sam"]["return_type"] = "boolean"
    router = Router("level")
    router.replies[0] = ("rule_observation", {"observation": "native_event"})
    router.replies.insert(1, ("rule_native_event_0", {"event_id": TICKS + "#END_LEVEL_TICK"}))
    router.replies.extend([
        ("select_callback_result_unhandled", {"expression": "false"}),
        ("select_callback_result_handled", {"expression": "true"}),
    ])
    source = compose_rule_feature(router, {
        "side": "COMMON", "target_path": "src/main/java/example/Bridge.java", "bindings": {},
    }, {"symbols": api, "project_root": str(tmp_path)}, package="example", name="Bridge",
        behaviors=[{"behavior_id": "native-result", "trigger": "Only when ready",
                    "effects": "Increment effects", "state_reads": [], "state_writes": []}], state_access=[])
    assert not router.replies
    compile_run(tmp_path, source, """
        ServerLevel w=new ServerLevel(); w.ready=false;
        check(!ServerTickEvents.END_LEVEL_TICK.listener.onEndTick(w)); check(w.effects==0);
        w.ready=true; check(ServerTickEvents.END_LEVEL_TICK.listener.onEndTick(w)); check(w.effects==1);
    """, callback_return="boolean")



def test_rule_materializes_computed_inventory_and_executes_real_gate(tmp_path):
    """A missing read-only planner label becomes inspected executable Java.

    Its getter is getInventory, not getWeaponInventory or the exact logical
    label. The host must compile an actual expression and require that value
    in the gate instead of accepting an unrelated world flag.
    """
    import json
    from minecraft_mod_ai.host_rule_feature import compose_rule_feature
    symbols = index()
    symbols["inventory"] = {
        "kind": "METHOD", "owner": LEVEL, "name": "getInventory",
        "access": 1, "parameters": [], "return_type": "int",
    }
    replies = deque([
        ("rule_observation", {"observation": "world_state"}),
        ("rule_value_operation", {"operation": "call"}),
        ("rule_api_query", {"query": "getInventory"}),
        ("rule_api_page", {"symbol": "s0"}),
        ("rule_operand", {"value_ref": "v0"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v1"}),
        ("rule_activation", {"mode": "level"}),
        ("rule_value_operation", {"operation": "binary"}),
        ("rule_binary_operator", {"operator": ">"}),
        ("rule_operand", {"value_ref": "v1"}),
        ("rule_operand", {"value_ref": "build"}),
        ("rule_value_operation", {"operation": "literal"}),
        ("rule_literal_type", {"type": "int"}),
        ("rule_literal", {"value": 0}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v2"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v3"}),
        ("rule_effect_operation", {"operation": "assign"}),
        ("rule_value_operation", {"operation": "field"}),
        ("rule_api_query", {"query": "effects"}),
        ("rule_api_page", {"symbol": "s0"}),
        ("rule_operand", {"value_ref": "v0"}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v4"}),
        ("rule_operand", {"value_ref": "build"}),
        ("rule_value_operation", {"operation": "literal"}),
        ("rule_literal_type", {"type": "int"}),
        ("rule_literal", {"value": 1}),
        ("rule_value_operation", {"operation": "use"}),
        ("rule_operand", {"value_ref": "v5"}),
        ("rule_assignment_operator", {"operator": "+="}),
        ("rule_effect_operation", {"operation": "complete"}),
    ])

    class Replay:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert replies, f"Unexpected extra inference for {data_name}"
            expected, value = replies.popleft()
            assert data_name == expected, (data_name, expected)
            return value

        def generate_java_source(self, *args, **kwargs):
            raise AssertionError("Host must own Java syntax")

    source = compose_rule_feature(Replay(), {
        "language": "java", "side": "COMMON",
        "target_path": "src/main/java/example/Bridge.java",
        "bindings": {
            "runtime_state_bindings_json": json.dumps({
                "Weapon inventory": {
                    "binding": "world_state",
                    "original_name": "Weapon inventory",
                    "observation": "Weapon inventory",
                    "semantic_status": "host_classified_unbound_read",
                }
            })
        }
    }, {"symbols": symbols, "project_root": str(tmp_path)},
        package="example", name="Bridge", behaviors=[{
            "behavior_id": "inventory-effect", "intent": "React to live inventory",
            "trigger": "Inventory is nonempty", "effects": "Increment effects",
            "state_reads": [], "state_writes": [],
            "runtime_state_reads": ["Weapon inventory"],
        }], state_access=[])
    assert not replies
    assert ".getInventory()" in source
    compile_run(tmp_path, source,
                "ServerLevel w=new ServerLevel(); "
                "tick(w,0); check(w.effects==0); "
                "w.weaponInventory=3; tick(w,1); check(w.effects==1); "
                "w.weaponInventory=0; tick(w,2); check(w.effects==1);")
