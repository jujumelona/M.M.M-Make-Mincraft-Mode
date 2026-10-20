"""Fabric events are fields of Event<T>, not interfaces named *Callback.EVENT."""
from copy import deepcopy
import shutil
import subprocess

import pytest

from minecraft_mod_ai.construction_context import build_target_symbols
from minecraft_mod_ai.host_fabric_callback_abi import callback_interfaces


EVENT = "net.fabricmc.fabric.api.event.Event"
TICKS = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"
LISTENER = TICKS + ".EndTick"


def event_symbols(owner=TICKS, listener=LISTENER, field="END_SERVER_TICK", side="COMMON"):
    return {
        "register": {"owner": EVENT, "kind": "METHOD", "name": "register",
                     "parameters": ["java.lang.Object"], "return_type": "void", "access": 1},
        owner: {"owner": owner, "kind": "CLASS", "access": 17, "side": side},
        listener: {"owner": listener, "kind": "CLASS", "access": 1545, "side": side},
        "field": {"owner": owner, "kind": "FIELD", "name": field, "access": 25,
                  "return_type": EVENT, "generic_return_type": EVENT + "<" + listener + ">",
                  "side": side},
        "sam": {"owner": listener, "kind": "METHOD", "name": "onEndTick", "access": 1025,
                "parameters": ["net.minecraft.server.MinecraftServer"], "return_type": "void",
                "side": side},
    }


def test_finds_nested_listener_from_event_field_generic_type():
    events = callback_interfaces(event_symbols())
    assert len(events) == 1
    assert events[0]["event_owner"] == TICKS
    assert events[0]["event_interface"] == LISTENER
    assert events[0]["event_field"] == "END_SERVER_TICK"
    assert events[0]["event_id"] == TICKS + "#END_SERVER_TICK"
    assert events[0]["parameters"] == ["net.minecraft.server.MinecraftServer"]


def test_indexes_entity_package_and_distinct_fields_using_same_listener():
    owner = "net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents"
    symbols = event_symbols(owner, owner + ".AfterDamage", "AFTER_DAMAGE")
    symbols["second"] = {**symbols["field"], "name": "OTHER_DAMAGE"}
    events = callback_interfaces(symbols)
    assert {e["event_id"] for e in events} == {owner + "#AFTER_DAMAGE", owner + "#OTHER_DAMAGE"}


@pytest.mark.parametrize("missing", ["register", TICKS, LISTENER, "sam"])
def test_missing_verified_declaration_does_not_create_binding(missing):
    symbols = event_symbols()
    del symbols[missing]
    assert callback_interfaces(symbols) == []


def test_rejects_erased_private_and_planned_event_fields():
    for change in [{"generic_return_type": EVENT}, {"access": 10},
                   {"provenance": "construction_plan"}]:
        symbols = event_symbols()
        symbols["field"].update(change)
        assert callback_interfaces(symbols) == []


def test_preserves_generic_sam_instead_of_advertising_erased_object_handler():
    symbols = event_symbols()
    symbols["sam"].update({
        "return_type": "java.lang.Object", "parameters": ["java.util.List"],
        "type_parameters": ["T extends java.lang.Object"],
        "generic_return_type": "T", "generic_parameters": ["java.util.List<T>"],
    })
    event = callback_interfaces(symbols)[0]
    assert event["type_parameters"] == ["T extends java.lang.Object"]
    assert event["return_type"] == "T"
    assert event["parameters"] == ["java.util.List<T>"]


def test_common_side_never_offers_client_event_to_server_feature():
    symbols = event_symbols(side="CLIENT")
    assert callback_interfaces(symbols) == []
    assert len(callback_interfaces(symbols, side="CLIENT")) == 1


@pytest.mark.parametrize("owner", [
    "net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents",
    "net.fabricmc.fabric.api.event.client.player.ClientPreAttackCallback",
])
def test_client_package_is_rejected_even_when_jar_has_no_side_annotations(owner):
    assert callback_interfaces(event_symbols(owner, owner + ".Listener")) == []


def test_discovers_event_from_real_classfile_signature(tmp_path):
    javac = shutil.which("javac")
    if not javac:
        pytest.skip("JDK required for bytecode inventory proof")
    sources = {
        "net/fabricmc/fabric/api/event/Event.java":
            "package net.fabricmc.fabric.api.event; public class Event<T> {"
            "public void register(T listener) {} }",
        "net/fabricmc/fabric/api/event/lifecycle/v1/ServerTickEvents.java":
            "package net.fabricmc.fabric.api.event.lifecycle.v1;"
            "import net.fabricmc.fabric.api.event.Event;"
            "public final class ServerTickEvents {"
            "public interface EndTick { void onEndTick(int tick); }"
            "public static final Event<EndTick> END_SERVER_TICK = new Event<>(); }",
    }
    paths = []
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        paths.append(str(path))
    classes = tmp_path / "classes"
    built = subprocess.run([javac, "-d", str(classes), *paths], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    symbols = build_target_symbols([classes], java_version=25, namespace="OFFICIAL", classpath_sides={})
    before = deepcopy(symbols)
    events = callback_interfaces(symbols)
    assert [(e["event_owner"], e["event_field"], e["parameters"]) for e in events] == [
        (TICKS, "END_SERVER_TICK", ["int"])]
    assert symbols == before
