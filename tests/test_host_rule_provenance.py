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
    from test_host_rule_feature import index

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
        }, {"symbols": index(), "project_root": str(tmp_path)},
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
    bindings = {"time": {"binding": "world_state",
                         "verified_symbol_ref": "clock",
                         "semantic_status": "host_verified_observer"}}
    symbols = {"clock": {"kind": "METHOD", "owner": "example.World",
                         "name": "getGameTime", "return_type": "long",
                         "parameters": [], "access": 1}}
    assert validate_runtime_read_producers({
        "behavior_id": "time", "runtime_state_reads": ["time"],
    }, bindings, symbols) == {"time": "clock"}
    for row in ({}, {"clock": {**symbols["clock"], "return_type": "void"}},
                {"clock": {**symbols["clock"], "parameters": ["int"]}}):
        with pytest.raises(RuleMissingRuntimeProducer):
            validate_runtime_read_producers({"behavior_id": "time",
                "runtime_state_reads": ["time"]}, bindings, row)


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
