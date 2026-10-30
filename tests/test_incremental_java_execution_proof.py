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


def test_custom_derived_gameplay_state_does_not_claim_native_mutation_proof():
    assert native_write_targets({**BEHAVIOR, "runtime_state_writes": ["claim_tracking"]}, {}) == ()
