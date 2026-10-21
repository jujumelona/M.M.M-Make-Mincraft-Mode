"""Domain conditions are tested before effects on the verified observation hook."""
import json
import shutil
import subprocess

import pytest

from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


EVENT = "net.fabricmc.fabric.api.event.Event"
TICKS = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"
LISTENER = TICKS + ".EndTick"
SERVER = "net.minecraft.server.MinecraftServer"
BEHAVIOR = {"behavior_id": "behavior_07000b123c21eb0a40a4",
            "intent": "Fight aliens in space.", "trigger": "Alien encounter in space",
            "effects": "Apply combat effects only during an alien encounter"}
SPEC = {"language": "java", "side": "COMMON",
        "target_path": "src/main/java/example/Bridge.java",
        "bindings": {"request_feature_entrypoint": "example.Bridge"}}


def symbols():
    return {
        "register": {"kind": "METHOD", "owner": EVENT, "name": "register", "access": 1,
                     "parameters": ["java.lang.Object"], "return_type": "void"},
        TICKS: {"owner": TICKS, "kind": "CLASS", "access": 17},
        LISTENER: {"owner": LISTENER, "kind": "CLASS", "access": 1545},
        "field": {"owner": TICKS, "kind": "FIELD", "name": "END_SERVER_TICK", "access": 25,
                  "return_type": EVENT, "generic_return_type": EVENT + "<" + LISTENER + ">"},
        "sam": {"owner": LISTENER, "kind": "METHOD", "name": "onEndTick", "access": 1025,
                "parameters": [SERVER], "return_type": "void"},
    }


class ConditionRouter:
    def __init__(self):
        self.tasks = []
        self.selections = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        assert role == "coder"
        if data_name in {"select_verified_fabric_event_index",
                         "select_verified_fabric_event_owner",
                         "select_verified_fabric_event"}:
            self.selections.append((data_name, json.loads(messages[-1]["content"])))
            if data_name == "select_verified_fabric_event_owner":
                return {"event_owner": TICKS}
            return {"event_binding": TICKS + "#END_SERVER_TICK"}
        member = json.loads(messages[1]["content"])
        if "_predicate_try_" in data_name or "_operation_0_try_" in data_name:
            self.tasks.append(member)
        if "_predicate_try_" in data_name:
            return {"expression": "minecraftServer.active"}
        if "_operation_0_try_" in data_name:
            return {"action": "assign"}
        if "_operation_0_assign_try_" in data_name:
            return {"target": "minecraftServer.effects", "operator": "+=", "value": "1"}
        if "_operation_1_try_" in data_name:
            return {"action": "complete"}
        raise AssertionError(data_name)


def compose(router, index=None):
    return compose_first_pass_java(
        router, SPEC, [], construction_context={"symbols": index or symbols()},
        behaviors=[BEHAVIOR])


def test_observation_callback_has_separate_typed_condition_and_effect_members():
    router = ConditionRouter()
    source = compose(router)
    assert "private boolean matchesBehavior" in source
    assert "private boolean applyBehavior" in source
    assert "if (this.matchesBehavior" in source
    assert len(router.tasks) == 2
    assert {t["member_role"] for t in router.tasks} == {
        "test_callback_condition", "implement_callback_effects"}
    assert all(t["assigned_behavior"] == BEHAVIOR for t in router.tasks)
    assert source.count(TICKS + ".END_SERVER_TICK.register(this::handleBehavior") == 1


def test_global_event_index_is_selected_before_sam_details_are_sent():
    index = symbols()
    for i in range(12):
        owner = "net.fabricmc.fabric.api.event.player.Example" + str(i)
        index[owner] = {"owner": owner, "kind": "CLASS", "access": 1537}
        index[owner + "field"] = {"owner": owner, "kind": "FIELD", "name": "EVENT",
            "access": 25, "return_type": EVENT, "generic_return_type": EVENT + "<" + owner + ">"}
        index[owner + "method"] = {"owner": owner, "kind": "METHOD", "access": 1025,
            "name": "interact", "parameters": ["int"], "return_type": "void"}
    router = ConditionRouter()
    compose(router, index)
    assert [name for name, _ in router.selections] == [
        "select_verified_fabric_event_index"]
    candidate_index = router.selections[0][1]
    assert len(candidate_index["verified_hook_index"]) == 13
    assert "verified_listeners" not in candidate_index
    assert "verified_event_groups" not in candidate_index
    assert all(set(item) == {"event_id", "inputs"}
               for item in candidate_index["verified_hook_index"])
    assert candidate_index["trigger"] == BEHAVIOR["trigger"]


def test_callback_primitive_parameters_receive_legal_stable_host_names():
    index = symbols()
    index["sam"]["parameters"] = [SERVER, "float", "float", "boolean"]
    source = compose(ConditionRouter(), index)
    assert "float eventArg1, float eventArg2, boolean eventArg3" in source


def test_false_domain_condition_never_executes_effects_on_tick(tmp_path):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required for compiled dispatch proof")
    source = compose(ConditionRouter())
    files = {
        "example/Bridge.java": source,
        "net/fabricmc/api/ModInitializer.java":
            "package net.fabricmc.api; public interface ModInitializer { void onInitialize(); }",
        "net/fabricmc/fabric/api/event/Event.java":
            "package net.fabricmc.fabric.api.event; public class Event<T> {"
            "public T listener; public void register(T listener) { this.listener=listener; }}",
        "net/fabricmc/fabric/api/event/lifecycle/v1/ServerTickEvents.java":
            "package net.fabricmc.fabric.api.event.lifecycle.v1;"
            "import net.fabricmc.fabric.api.event.Event;"
            "public final class ServerTickEvents {"
            "public interface EndTick { void onEndTick(net.minecraft.server.MinecraftServer server); }"
            "public static final Event<EndTick> END_SERVER_TICK = new Event<>(); }",
        "net/minecraft/server/MinecraftServer.java":
            "package net.minecraft.server; public class MinecraftServer {"
            "public boolean active; public int effects; }",
        "example/Probe.java":
            "package example; import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;"
            "public class Probe { public static void main(String[] args) {"
            "new Bridge().onInitialize(); var server=new net.minecraft.server.MinecraftServer();"
            "ServerTickEvents.END_SERVER_TICK.listener.onEndTick(server);"
            "if (server.effects!=0) throw new AssertionError(\"false condition ran effects\");"
            "server.active=true; ServerTickEvents.END_SERVER_TICK.listener.onEndTick(server);"
            "if (server.effects!=1) throw new AssertionError(\"true condition lost effects\");"
            "server.active=false; ServerTickEvents.END_SERVER_TICK.listener.onEndTick(server);"
            "if (server.effects!=1) throw new AssertionError(\"later false condition ran effects\"); }}",
    }
    paths = []
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        paths.append(str(path))
    dest = tmp_path / "classes"
    built = subprocess.run([javac, "-d", str(dest), *paths], capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([java, "-cp", str(dest), "example.Probe"], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
