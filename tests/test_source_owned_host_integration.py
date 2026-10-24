"""End-to-end Python host lowering of inspected Fabric hook to typed Java source.

This fixture is intentionally not a Minecraft playtest. It verifies the real
Python call chain rather than testing the graph and registrar in isolation.
"""
import json

from minecraft_mod_ai.feature_source_work_graph import build_source_work_graph
from minecraft_mod_ai.feature_registration_proof import validate_registered_event_sources
from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


def test_source_owned_runtime_producer_work_reaches_typed_handler(monkeypatch):
    event = {
        "event_id": "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents#END_SERVER_TICK",
        "event_owner": "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents",
        "event_field": "END_SERVER_TICK",
        "event_interface": "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents$EndTick",
        "method": "onEndTick",
        "parameters": ["net.minecraft.server.MinecraftServer"],
        "return_type": "void",
        "type_parameters": [],
        "throws": [],
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.host_fabric_callback_abi.callback_interfaces",
        lambda symbols, side: [event],
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.host_java_imports.host_add_verified_imports",
        lambda source, **kwargs: source,
    )
    behavior = {
        "behavior_id": "generic_ship_work", "intent": "Operate a world vehicle",
        "trigger": "Vehicle construction condition is satisfied",
        "effects": "Change the vehicle's state",
        "source_quote": "Build a vehicle and use it to travel",
        "runtime_state_reads": ["selected_section"],
    }
    graph = build_source_work_graph(
        [behavior],
        producer_obligations=[{
            "behavior_id": behavior["behavior_id"],
            "name": "selected_section", "reason": "planning_only_world_state",
        }],
        inspected_hooks=[event],
    )
    work_seen = []
    def author_body(router, messages, *, data_name, frame, predicate,
                    effect_receipt=False):
        current = json.loads(messages[-1]["content"])
        scope = current["source_owned_work"]
        work_seen.append(scope)
        assert any(row["kind"] == "runtime_producer" for row in scope["source_tasks"])
        assert all(row["behavior_id"] == behavior["behavior_id"]
                   for row in scope["source_tasks"])
        body = "return true;" if predicate else (
            "java.lang.System.out.println(\"test only\");\nreturn true;"
            if effect_receipt else "java.lang.System.out.println(\"test only\");")
        return frame["prefix"] + body + frame["suffix"]

    monkeypatch.setattr(
        "minecraft_mod_ai.host_callback_body.author_callback_body",
        author_body,
    )

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert role == "planner"
            assert data_name == "select_source_owned_observation_hook"
            assert event["event_id"] in parameters["properties"]["event_id"]["enum"]
            return {"event_id": event["event_id"]}

        def generate_java_source(self, *args, **kwargs):
            raise AssertionError("host must own callbacks, not freeform Java")

    spec = {
        "language": "java",
        "target_path": "src/main/java/example/Feature.java",
        "requirement": behavior["source_quote"],
        "bindings": {"package_name": "example", "class_name": "Feature",
                     "request_feature_entrypoint": True},
    }
    source = compose_first_pass_java(
        Router(), spec, [{"role": "user", "content": json.dumps({
            "assigned_behaviors": [behavior], "inspected_target_api": {},
        })}],
        construction_context={
            "symbols": {}, "host_source_owned": True,
            "source_owned_work_graph": graph, "authoring_contract": {},
        }, behaviors=[behavior],
    )
    validate_registered_event_sources(source, [behavior], [event])
    assert "implements net.fabricmc.api.ModInitializer" in source
    assert event["event_owner"] + "." + event["event_field"] + ".register(" in source
    assert "matchesBehavior" in source
    assert "applyBehavior" in source
    # The host now authors a separate typed method for the missing runtime
    # observation. Event registration alone cannot satisfy that obligation.
    from minecraft_mod_ai.feature_source_work_graph import runtime_observer_methods
    observer_name = runtime_observer_methods(graph, behavior["behavior_id"])[0]["name"]
    assert observer_name in source
    assert "this." + observer_name + "(minecraftServer)" in source
    assert len(work_seen) == 3  # original predicate, observer, gameplay effect
    assert graph["completion_verified"] is False
