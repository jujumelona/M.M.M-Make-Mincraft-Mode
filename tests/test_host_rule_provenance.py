"""Replays missing-runtime-producer obligations from production-logs (14).

These tests must never approve an unrelated Fabric/world boolean just
because the Java method exists or compiles.
"""
import json

import pytest

from minecraft_mod_ai.host_rule_provenance import (
    RuleMissingRuntimeProducer, validate_runtime_read_producers,
)


def test_unbound_world_observations_fail_before_model_java_requests(tmp_path):
    from minecraft_mod_ai.host_rule_feature import compose_rule_feature
    # Minimal inspected Fabric tick ABI; deliberately independent of tests
    # being imported in any particular order or pytest import mode.
    owner = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"
    event = "net.fabricmc.fabric.api.event.Event"
    callback = owner + ".EndTick"
    level = "net.minecraft.server.level.ServerLevel"
    symbols = {
        owner: {"kind": "CLASS", "owner": owner, "access": 1},
        event: {"kind": "CLASS", "owner": event, "access": 1},
        callback: {"kind": "CLASS", "owner": callback, "access": 1545},
        level: {"kind": "CLASS", "owner": level, "access": 1},
        "register": {"kind": "METHOD", "owner": event, "name": "register",
                     "access": 1, "parameters": ["java.lang.Object"], "return_type": "void"},
        "tick": {"kind": "FIELD", "owner": owner, "name": "END_LEVEL_TICK",
                 "access": 25, "return_type": event,
                 "generic_return_type": event + "<" + callback + ">"},
        "sam": {"kind": "METHOD", "owner": callback, "name": "onEndTick",
                "access": 1025, "parameters": [level], "return_type": "void"},
    }

    class NoRequests:
        def generate_structured_data(self, *args, **kwargs):
            raise AssertionError("Missing source producer must be detected before model calls")

    required = ["Alien presence", "Weapon inventory"]
    bindings = {name: {"original_name": name, "binding": "world_state",
                       "observation": name,
                       "semantic_status": "host_classified_unbound_read"}
                for name in required}
    with pytest.raises(RuleMissingRuntimeProducer) as caught:
        compose_rule_feature(NoRequests(), {
            "side": "COMMON", "target_path": "src/main/java/example/Bridge.java",
            "bindings": {"runtime_state_bindings_json": json.dumps(bindings)},
        }, {"symbols": symbols, "project_root": str(tmp_path)},
            package="example", name="Bridge",
            behaviors=[{"behavior_id": "combat", "trigger": "Alien encounter in space",
                        "effects": "Combat with aliens",
                        "runtime_state_reads": required,
                        "state_reads": [], "state_writes": []}],
            state_access=[])
    assert {row["name"] for row in caught.value.missing} == set(required)
    assert all(row["reason"] == "planning_only_world_state" for row in caught.value.missing)
    assert "RULE_RUNTIME_PRODUCER_REQUIRED" in str(caught.value)


def test_semantic_world_label_is_not_java_producer():
    bindings = {"selected_section": {"binding": "world_state",
                "observation": "selected_section",
                "semantic_status": "model_authored_unverified"}}
    symbols = {"pvp": {"kind": "METHOD", "owner": "example.World",
                       "name": "isPvpAllowed", "return_type": "boolean",
                       "parameters": [], "access": 1}}
    with pytest.raises(RuleMissingRuntimeProducer) as error:
        validate_runtime_read_producers({"behavior_id": "ship", "runtime_state_reads": [
            "selected_section"]}, bindings, symbols)
    assert error.value.missing[0]["name"] == "selected_section"


def test_checked_source_id_requires_existing_readable_java_signature():
    bindings = {"world_time": {"binding": "world_state",
                        "verified_symbol_ref": "clock",
                        "semantic_status": "host_verified_observer"}}
    symbols = {"example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
               "clock": {"kind": "METHOD", "owner": "example.World",
                         "name": "getWorldTime", "return_type": "long",
                         "parameters": [], "access": 1}}
    expected = {"behavior_id": "time", "runtime_state_reads": ["world_time"]}
    assert validate_runtime_read_producers(
        expected, bindings, symbols, input_types=["example.World"]
    ) == {"world_time": "clock"}
    for row in ({}, {"clock": {**symbols["clock"], "return_type": "void"}},
                {"clock": {**symbols["clock"], "parameters": ["int"]}}):
        with pytest.raises(RuleMissingRuntimeProducer):
            validate_runtime_read_producers(
                expected, bindings, {**symbols, **row}, input_types=["example.World"])

    # A valid but semantically unrelated symbol cannot be made authoritative
    # by inserting 'host_verified_observer' into a planner-controlled record.
    malicious = {"world_time": {**bindings["world_time"],
                 "verified_symbol_ref": "pvp"}}
    with pytest.raises(RuleMissingRuntimeProducer):
        validate_runtime_read_producers(
            expected, malicious, {**symbols, "pvp": {**symbols["clock"],
                "name": "isPvpAllowed"}}, input_types=["example.World"])


def test_unrelated_world_boolean_cannot_cover_verified_observation():
    from minecraft_mod_ai.host_rule_program import RuleProgram
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    symbols = {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "pvp": {"kind": "METHOD", "owner": "example.World", "access": 1,
                "name": "isPvpAllowed", "return_type": "boolean", "parameters": []},
        "presence": {"kind": "METHOD", "owner": "example.World", "access": 1,
                     "name": "isAlienPresent", "return_type": "boolean", "parameters": []},
    }
    program = RuleProgram(symbols, [{"type": "example.World", "name": "arg0"}],
        {"behavior_id": "combat", "state_reads": [], "state_writes": [],
         "verified_runtime_observers": {"Alien presence": "presence"}},
        owner="example.Bridge", target_path="src/main/java/example/Bridge.java")
    pvp = program.value("call", [program.inputs[0]], symbol_ref="pvp")
    presence = program.value("call", [program.inputs[0]], symbol_ref="presence")
    author = RuleAuthor(None, program)
    assert not author._runtime_grounded(pvp)
    assert author._runtime_grounded(presence)



def test_checked_observation_menu_never_reoffers_identical_getter_receiver():
    from minecraft_mod_ai.host_rule_program import RuleProgram
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    symbols = {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "clock": {"kind": "METHOD", "owner": "example.World", "access": 1,
                  "name": "getGameTime", "parameters": [], "return_type": "long"},
    }
    p = RuleProgram(symbols, [{"type": "example.World", "name": "arg0"}],
        {"behavior_id": "day-boundary", "trigger": "World game time changes",
         "state_reads": [], "state_writes": []},
        owner="example.Bridge", target_path="src/main/java/example/Bridge.java")
    author = RuleAuthor(None, p)
    first = author._checked_observation_candidates("Actual game time")
    assert len(first) == 1 and first[0]["symbol_ref"] == "clock"
    old = p.value("call", [p.inputs[0]], symbol_ref="clock")
    assert p.source(old)
    assert author._checked_observation_candidates("Actual game time") == []


def test_checked_observation_menu_uses_bound_signal_producer_not_pvp():
    from minecraft_mod_ai.host_rule_program import RuleProgram
    from minecraft_mod_ai.host_rule_author import RuleAuthor

    symbols = {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "pvp": {"kind": "METHOD", "owner": "example.World", "access": 1,
                "name": "isPvpAllowed", "return_type": "boolean", "parameters": []},
        "presence": {"kind": "METHOD", "owner": "example.World", "access": 1,
                     "name": "isAlienPresent", "return_type": "boolean", "parameters": []},
    }
    p = RuleProgram(symbols, [{"type": "example.World", "name": "arg0"}],
        {"behavior_id": "combat", "trigger": "Alien presence",
         "verified_runtime_observers": {"Alien presence": "presence"},
         "state_reads": [], "state_writes": []},
        owner="example.Bridge", target_path="src/main/java/example/Bridge.java")
    choices = RuleAuthor(None, p)._checked_observation_candidates("Alien presence")
    assert [row["symbol_ref"] for row in choices] == ["presence"]



def test_deterministic_exact_name_getter_resolves_without_model_api_guess():
    symbols = {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "worldtime": {"kind": "METHOD", "owner": "example.World", "access": 1,
                      "name": "getWorldTime", "parameters": [],
                      "return_type": "long"},
        "unrelated": {"kind": "METHOD", "owner": "example.World", "access": 1,
                      "name": "isPvpAllowed", "parameters": [],
                      "return_type": "boolean"},
    }
    state = "world_time"
    bindings = {state: {"binding": "world_state", "observation": state,
                        "semantic_status": "host_classified_unbound_read"}}
    assert validate_runtime_read_producers({
        "behavior_id": "world-rule", "runtime_state_reads": [state],
    }, bindings, symbols, input_types=["example.World"]) == {
        state: "worldtime",
    }
    with pytest.raises(RuleMissingRuntimeProducer):
        validate_runtime_read_producers({
            "behavior_id": "world-rule", "runtime_state_reads": [state],
        }, bindings, symbols, input_types=["example.Other"])


def test_ambiguous_exact_name_sources_require_explicit_provenance():
    owner = "example.World"
    symbols = {owner: {"kind": "CLASS", "owner": owner, "access": 1}}
    for key in ("first", "second"):
        symbols[key] = {"kind": "METHOD", "owner": owner, "access": 1,
                        "name": "getWorldTime", "parameters": [],
                        "return_type": "long"}
    with pytest.raises(RuleMissingRuntimeProducer) as caught:
        validate_runtime_read_producers({
            "behavior_id": "world-rule", "runtime_state_reads": ["world_time"],
        }, {"world_time": {"binding": "world_state",
                           "semantic_status": "host_classified_unbound_read"}},
           symbols, input_types=[owner])
    assert caught.value.missing[0]["reason"] == "ambiguous_runtime_observer"
    assert caught.value.missing[0]["matching_candidate_count"] == 2
