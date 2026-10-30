"""Direct feature activation and native mutation source-proof regression."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.incremental_java_execution_proof import (
    verify_incremental_feature_execution, native_write_targets,
)


CALLBACK = {"event_id": "example.Events#ACTION", "event_owner": "example.Events",
            "event_field": "ACTION", "parameters": ["String"], "return_type": "void"}
BEHAVIOR = {"behavior_id": "resource", "trigger": "Player acquires a resource",
            "runtime_state_writes": []}
OWNER = "net.minecraft.world.World"


def _program(*, register=True, native_effect=None):
    registrar = {"id": "m1", "kind": "method", "name": "registerBehavior",
                 "owner": "Feature", "registrar": True,
                 "behavior": dict(BEHAVIOR),
                 "body": ([{"id": "n1", "op": "register", "event_owner": "example.Events",
                           "event_field": "ACTION", "handler": "handleEvent"}]
                          if register else [])}
    handler = {"id": "m2", "kind": "method", "name": "handleEvent",
               "owner": "Feature", "event": dict(CALLBACK), "behavior": dict(BEHAVIOR),
               "body": []}
    if native_effect:
        handler["body"] = [{"id": "n2", "op": "call", "method": "setBlockState",
                            "receiver": "arg0", "arguments": ["pos", "newState"],
                            "call_contract": {"kind": "METHOD", "owner": OWNER,
                                              "name": "setBlockState", "parameters": [
                                                  "BlockPos", "BlockState"]},
                            "native_state_effect": native_effect}]
    return {"members": [registrar, handler]}


def test_exact_event_hook_required_not_just_compiling_registrar():
    program = _program()
    verify_incremental_feature_execution(program, [BEHAVIOR], [CALLBACK], {}, {})
    with pytest.raises(ValueError, match="INSPECTED_OCCURRENCE_MISSING"):
        verify_incremental_feature_execution(_program(register=False),
                                             [BEHAVIOR], [CALLBACK], {}, {})
    with pytest.raises(ValueError, match="INSPECTED_OCCURRENCE_MISSING"):
        verify_incremental_feature_execution(program, [BEHAVIOR], [], {}, {})
    with pytest.raises(ValueError, match="INSPECTED_OCCURRENCE_MISSING"):
        verify_incremental_feature_execution(
            program, [{**BEHAVIOR, "event_binding": "other.Events#BLOCK"}], [CALLBACK], {}, {})


def test_native_world_state_requires_a_checked_live_mutator():
    native = {**BEHAVIOR, "runtime_state_writes": ["world block state"]}
    bindings = {"world block state": {"binding": "world_state",
                                   "semantic_status": "host_native_runtime_owner"}}
    expected = {"world#setBlock": {"kind": "METHOD", "owner": OWNER, "name": "setBlockState",
                                   "parameters": ["BlockPos", "BlockState"]}}
    assert native_write_targets(native, bindings) == ("world block state",)
    with pytest.raises(ValueError, match="NATIVE_MUTATION_UNPROVEN"):
        verify_incremental_feature_execution(_program(), [native], [CALLBACK], bindings, expected)
    program = _program(native_effect="world block state")
    with pytest.raises(ValueError, match="NATIVE_MUTATION_UNPROVEN"):
        verify_incremental_feature_execution(program, [native], [CALLBACK], bindings, {})
    verify_incremental_feature_execution(program, [native], [CALLBACK], bindings, expected)


def test_invalid_or_unreachable_mutation_is_not_causal_evidence():
    native = {**BEHAVIOR, "runtime_state_writes": ["world block state"]}
    bindings = {"world block state": {"semantic_status": "host_native_runtime_owner"}}
    symbols = {"s": {"kind": "METHOD", "owner": OWNER, "name": "setBlockState",
                     "parameters": ["BlockPos", "BlockState"]}}
    program = _program(native_effect="world block state")
    body = program["members"][1]["body"]
    program["members"][1]["body"] = [{"id": "n3", "op": "if", "condition": "false",
                                       "body": body, "otherwise": []}]
    with pytest.raises(ValueError, match="NATIVE_MUTATION_UNPROVEN"):
        verify_incremental_feature_execution(program, [native], [CALLBACK], bindings, symbols)
    program["members"][1]["body"] = body
    body[0]["call_contract"]["name"] = "getBlockState"
    with pytest.raises(ValueError, match="NATIVE_MUTATION_UNPROVEN"):
        verify_incremental_feature_execution(program, [native], [CALLBACK], bindings, symbols)


def test_unbound_runtime_write_fails_closed_but_declared_custom_state_is_separate():
    obligation = {**BEHAVIOR, "runtime_state_writes": ["claim_tracking"]}
    assert native_write_targets(obligation, {}) == ("claim_tracking",)
    declared = {"claim_tracking": {"semantic_status": "host_owned_state_model"}}
    assert native_write_targets(obligation, declared) == ()



def test_compiled_path_rechecks_event_and_native_effect_on_every_verification():
    from minecraft_mod_ai.incremental_java_pipeline import _verify_observable_program

    native = {**BEHAVIOR, "runtime_state_writes": ["world block state"]}
    binding = {"world block state": {"semantic_status": "host_native_runtime_owner"}}
    symbols = {"s": {"kind": "METHOD", "owner": OWNER, "name": "setBlockState",
                     "parameters": ["BlockPos", "BlockState"]}}
    program = _program(native_effect="world block state")
    program["members"][0]["behavior"] = native
    program["members"][1]["behavior"] = native
    context = {"incremental_state": {"program": program}, "behaviors": [native],
               "verified_fabric_callbacks": [CALLBACK], "runtime_state_bindings": binding,
               "symbols": symbols}
    _verify_observable_program(context)

    # A completed cached AST must be checked again, not just its compiled source.
    context["verified_fabric_callbacks"] = []
    with pytest.raises(ValueError, match="INSPECTED_OCCURRENCE_MISSING"):
        _verify_observable_program(context)
    context["verified_fabric_callbacks"] = [CALLBACK]
    context["symbols"] = {}
    with pytest.raises(ValueError, match="NATIVE_MUTATION_UNPROVEN"):
        _verify_observable_program(context)


def test_compiler_repair_maps_missing_native_effect_to_real_listener():
    from minecraft_mod_ai.incremental_java_state import _repair_entrypoint

    program = _program(native_effect=None)
    member, diagnostic = _repair_entrypoint(
        program, "INCREMENTAL_JAVA_NATIVE_MUTATION_UNPROVEN: resource states=['world block state']")
    assert member["id"] == "m2"
    assert diagnostic.startswith("CONSTRUCTION_FILE_BEHAVIOR_NOOP:")
    member, diagnostic = _repair_entrypoint(
        program, "INCREMENTAL_JAVA_INSPECTED_OCCURRENCE_MISSING: resource")
    assert member["id"] == "m1"
    assert diagnostic.startswith("CONSTRUCTION_FILE_BEHAVIOR_NOOP:")


def test_mutation_choice_is_attached_only_to_checked_source_call():
    from minecraft_mod_ai.incremental_java_operations import _record_checked_native_effect

    class API:
        bindings = {"world block state": {"semantic_status": "host_native_runtime_owner"}}
        symbols = {"s": {"kind": "METHOD", "owner": OWNER, "name": "setBlockState",
                         "parameters": ["BlockPos", "BlockState"]}}

    class Builder:
        api = API()
        member = {"behavior": {**BEHAVIOR, "runtime_state_writes": ["world block state"]}}
        def ask(self, schema, suffix, extra=None):
            assert suffix == "native_mutation_state"
            assert "world block state" in schema["properties"]["native_state"]["enum"]
            return {"native_state": "world block state"}

    node = _program(native_effect="world block state")["members"][1]["body"][0]
    node.pop("native_state_effect")
    _record_checked_native_effect(Builder(), node)
    assert node["native_state_effect"] == "world block state"
    node["call_contract"]["name"] = "getBlockState"
    node.pop("native_state_effect")
    _record_checked_native_effect(Builder(), node)
    assert "native_state_effect" not in node


def test_java_logging_and_event_registration_cannot_claim_native_world_mutation():
    from minecraft_mod_ai.incremental_java_execution_proof import _inspected_call

    for owner, method in (("java.io.PrintStream", "println"),
                          ("net.fabricmc.fabric.api.event.Event", "register")):
        node = {"op": "call", "method": method, "receiver": "arg0",
                "call_contract": {"kind": "METHOD", "owner": owner,
                                  "name": method, "parameters": ["String"]}}
        symbols = {"method": {"kind": "METHOD", "owner": owner,
                              "name": method, "parameters": ["String"]}}
        assert not _inspected_call(node, symbols)


def test_verified_source_ui_event_is_selectable_in_incremental_coder_context():
    from minecraft_mod_ai.incremental_java_context import _combine_checked_events

    original = [CALLBACK]
    custom = {"event_id": "example.OriginalOccurrence#EVENT",
              "event_owner": "example.OriginalOccurrence", "event_field": "EVENT",
              "behavior_id": "resource", "provenance": "host_typed_ui_action"}
    rows = _combine_checked_events(original, [custom])
    assert {row["event_id"] for row in rows} == {
        "example.Events#ACTION", "example.OriginalOccurrence#EVENT"}
    with pytest.raises(ValueError, match="JAVA_CALLBACK_EVENT_ABI_CONFLICT"):
        _combine_checked_events(original, [{**CALLBACK, "return_type": "boolean"}])


def test_real_fabric_event_must_observe_the_user_requested_occurrence():
    arrival = {**BEHAVIOR, "trigger": "Player enters another dimension"}
    program = _program()
    program["members"][0]["behavior"] = arrival
    program["members"][1]["behavior"] = arrival
    with pytest.raises(ValueError, match="INSPECTED_OCCURRENCE_MISSING"):
        verify_incremental_feature_execution(
            program, [arrival], [CALLBACK], {}, {})
    source_event = {**CALLBACK, "provenance": "verified_source_site_mixin",
                    "behavior_id": arrival["behavior_id"]}
    verify_incremental_feature_execution(program, [arrival], [source_event], {}, {})
