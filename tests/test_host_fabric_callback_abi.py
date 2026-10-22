"""Regression from production-logs 17: unbound 'player' in lifecycle register."""
import json

import pytest

from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java
from minecraft_mod_ai.host_fabric_callback_abi import callback_interfaces

OWNER = "net.fabricmc.fabric.api.event.player.UseBlockCallback"
EVENT = "net.fabricmc.fabric.api.event.Event"
PARAMETERS = [
    "net.minecraft.world.entity.player.Player",
    "net.minecraft.world.level.Level",
    "net.minecraft.world.InteractionHand",
    "net.minecraft.world.phys.BlockHitResult",
]
INDEX = {
    "event.register": {
        "kind": "METHOD", "owner": EVENT, "name": "register",
        "access": 1, "parameters": [OWNER], "return_type": "void",
    },
    "callback.class": {
        "kind": "CLASS", "owner": OWNER, "access": 1537,
        "name": "UseBlockCallback",
    },
    "callback.event": {
        "kind": "FIELD", "owner": OWNER, "access": 25, "name": "EVENT",
        "return_type": EVENT, "generic_return_type": EVENT + "<" + OWNER + ">",
    },
    "callback.method": {
        "kind": "METHOD", "owner": OWNER, "access": 1025, "name": "interact",
        "return_type": "net.minecraft.world.InteractionResult",
        "parameters": PARAMETERS,
    },
}
SPEC = {
    "language": "java", "side": "COMMON",
    "target_path": "src/main/java/example/Bridge.java",
    "bindings": {"package_name": "example", "class_name": "Bridge",
                 "request_feature_entrypoint": "example.Bridge"},
    "requirement": "Build a ship by player block interaction",
}
BEHAVIORS = [{"behavior_id": "ship.build", "intent": "Build a ship",
              "trigger": "Player interacts with shipyard block",
              "effects": "Assemble ship modules"}]


class EventRouter:
    def __init__(self, selected=OWNER):
        self.selected = selected
        self.calls = []
        self.source_tasks = []
    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append(data_name)
        assert role == "coder"
        if data_name == "select_verified_fabric_event":
            assert set(parameters["properties"]["event_binding"]["enum"]) == {
                "UNBOUND", OWNER + "#EVENT",
            }
            return {"event_binding": "UNBOUND" if self.selected == "UNBOUND"
                    else self.selected + "#EVENT"}
        if data_name.startswith("select_callback_result_"):
            return {"expression": "net.minecraft.world.InteractionResult.PASS"}
        member = json.loads(messages[1]["content"])
        if "_predicate_try_" in data_name or "_operation_0_try_" in data_name:
            self.source_tasks.append(member)
            assert "Player player" in member["immutable_signature"]["prefix"]
            assert "Level level" in member["immutable_signature"]["prefix"]
        if "_predicate_try_" in data_name:
            return {"expression": "player != null"}
        if "_operation_0_call_try_" in data_name:
            return {"receiver": "player", "method": "setPos", "arguments": ["0.0", "0.0", "0.0"]}
        if "_operation_0_try_" in data_name:
            return {"action": "call"}
        if "_operation_1_try_" in data_name:
            return {"action": "complete"}
        raise AssertionError(data_name)


def test_actual_interface_field_and_sam_are_required():
    found = callback_interfaces(INDEX)
    assert len(found) == 1
    assert found[0]["parameters"] == PARAMETERS
    for badkey in ["event.register", "callback.class", "callback.event",
                   "callback.method"]:
        candidates = {k: v for k, v in INDEX.items() if k != badkey}
        assert callback_interfaces(candidates) == []


def test_host_registers_callback_with_method_reference_and_exact_abi():
    router = EventRouter()
    source = compose_first_pass_java(
        router, SPEC, [],
        construction_context={"symbols": INDEX, "authoring_contract": {}},
        behaviors=BEHAVIORS,
    )
    assert source.count(".EVENT.register(this::handleBehavior") == 1
    assert "private net.minecraft.world.InteractionResult handleBehavior" in source
    assert "Player player" in source
    assert source.count("void onInitialize()") == 1
    assert len(router.source_tasks) == 2
    assert "private boolean matchesBehavior" in source
    assert "private boolean applyBehavior" in source
    assert router.calls == [
        "select_verified_fabric_event",
        "submit_java_implementation_1_predicate_try_0",
        "submit_java_implementation_2_operation_0_try_0",
        "submit_java_implementation_2_operation_0_call_try_0",
        "submit_java_implementation_2_operation_1_try_0",
        "select_callback_result_unhandled",
        "select_callback_result_handled",
    ]
    assert router.source_tasks[0]["verified_event_callback"]["event_interface"] == OWNER


def test_host_does_not_invent_callback_when_no_real_trigger_matches():
    with pytest.raises(ValueError, match="JAVA_HOST_BEHAVIOR_EVENT_UNBOUND"):
        compose_first_pass_java(
            EventRouter(selected="UNBOUND"), SPEC, [],
            construction_context={"symbols": INDEX, "authoring_contract": {}},
            behaviors=BEHAVIORS,
        )


def test_callback_body_receives_exact_input_and_result_apis_when_text_search_is_empty():
    index = dict(INDEX)
    player, level, hand, hit = PARAMETERS
    result = "net.minecraft.world.InteractionResult"
    for owner in [player, level, hand, hit, result]:
        index[owner] = {"kind": "CLASS", "owner": owner,
                        "name": owner.rsplit(".", 1)[-1], "access": 1}
    index["player.id"] = {"kind": "METHOD", "owner": player, "name": "getUUID",
                           "parameters": [], "return_type": "java.util.UUID", "access": 1}
    index["result.pass"] = {"kind": "FIELD", "owner": result, "name": "PASS",
                             "return_type": result, "access": 25, "static": True}
    index["private"] = {"kind": "METHOD", "owner": player, "name": "privateHelper",
                         "parameters": [], "return_type": "void", "access": 2}
    router = EventRouter()
    compose_first_pass_java(router, SPEC, [],
        construction_context={"symbols": index, "authoring_contract": {}}, behaviors=BEHAVIORS)
    apis = router.source_tasks[0]["callback_input_api"]
    by_class = {record["verified_class"]: record for record in apis}
    assert {player, level, hand, hit, result} <= by_class.keys()
    assert any(member["name"] == "getUUID" for member in by_class[player]["members"])
    assert not any(member["name"] == "privateHelper" for member in by_class[player]["members"])
    assert any(member["name"] == "PASS" for member in by_class[result]["members"])



def test_method_reference_and_sam_compile_together(tmp_path):
    import shutil
    import subprocess
    javac = shutil.which("javac")
    if not javac:
        pytest.skip("A JDK compiler is required for the method-reference integration proof")
    router = EventRouter()
    generated = compose_first_pass_java(
        router, SPEC, [],
        construction_context={"symbols": INDEX, "authoring_contract": {}},
        behaviors=BEHAVIORS,
    )
    files = {
        "example/Bridge.java": generated,
        "net/fabricmc/api/ModInitializer.java":
            "package net.fabricmc.api; public interface ModInitializer { void onInitialize(); }",
        "net/fabricmc/fabric/api/event/Event.java":
            ("package net.fabricmc.fabric.api.event; "
             "public class Event<T> { public void register(T callback) {} }"),
        "net/fabricmc/fabric/api/event/player/UseBlockCallback.java":
            ("package net.fabricmc.fabric.api.event.player;"
             " public interface UseBlockCallback {"
             " net.fabricmc.fabric.api.event.Event<UseBlockCallback> EVENT"
             " = new net.fabricmc.fabric.api.event.Event<>();"
             " net.minecraft.world.InteractionResult interact("
             " net.minecraft.world.entity.player.Player player,"
             " net.minecraft.world.level.Level level,"
             " net.minecraft.world.InteractionHand hand,"
             " net.minecraft.world.phys.BlockHitResult hit); }"),
        "net/minecraft/world/entity/player/Player.java":
            "package net.minecraft.world.entity.player; public class Player { "
            "public void setPos(double x, double y, double z) {} }",
        "net/minecraft/world/level/Level.java":
            "package net.minecraft.world.level; public class Level {}",
        "net/minecraft/world/InteractionHand.java":
            "package net.minecraft.world; public enum InteractionHand { MAIN_HAND }",
        "net/minecraft/world/InteractionResult.java":
            "package net.minecraft.world; public enum InteractionResult { PASS }",
        "net/minecraft/world/phys/BlockHitResult.java":
            "package net.minecraft.world.phys; public class BlockHitResult {}",
    }
    paths = []
    for relative, content in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        paths.append(str(target))
    dest = tmp_path / "classes"
    dest.mkdir()
    finished = subprocess.run(
        [javac, "-proc:none", "-d", str(dest), *paths],
        capture_output=True, text=True, timeout=90,
    )
    assert finished.returncode == 0, finished.stderr
