"""Regression from production-logs 17: unbound 'player' in lifecycle register."""
import json

import pytest

from minecraft_mod_ai.host_fabric_callback_abi import callback_interfaces
from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


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
        "return_type": EVENT,
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
        assert data_name == "select_verified_fabric_event"
        assert set(parameters["properties"]["event_interface"]["enum"]) == {
            "UNBOUND", OWNER,
        }
        return {"event_interface": self.selected}
    def generate_java_source(self, role, messages, *, data_name, source_frame):
        self.source_tasks.append(json.loads(messages[-1]["content"]))
        assert role == "coder"
        assert data_name == "submit_java_implementation_1"
        assert "registerBehavior" not in source_frame["prefix"]
        assert "Player player" in source_frame["prefix"]
        assert "Level level" in source_frame["prefix"]
        return (source_frame["prefix"]
                + "return net.minecraft.world.InteractionResult.PASS;"
                + source_frame["suffix"])


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
    assert len(router.source_tasks) == 1
    assert router.calls == ["select_verified_fabric_event"]
    assert router.source_tasks[0]["verified_event_callback"]["event_interface"] == OWNER


def test_host_does_not_invent_callback_when_no_real_trigger_matches():
    with pytest.raises(ValueError, match="JAVA_HOST_BEHAVIOR_EVENT_UNBOUND"):
        compose_first_pass_java(
            EventRouter(selected="UNBOUND"), SPEC, [],
            construction_context={"symbols": INDEX, "authoring_contract": {}},
            behaviors=BEHAVIORS,
        )



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
            "package net.minecraft.world.entity.player; public class Player {}",
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
