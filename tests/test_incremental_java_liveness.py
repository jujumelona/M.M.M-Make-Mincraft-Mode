"""Prompt-independent liveness regressions for host-owned Java authoring.

The original failure produced hundreds of empty nested blocks and repeated
active-helper declarations without a single executable statement.
"""
from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.incremental_java_schema import (
    MAX_CONTROL_FLOW_NESTING,
    operation_options,
)
from test_incremental_java_recovery import _builder


def _member():
    return {"id": "m1", "kind": "method", "owner": "Feature"}


def _scope():
    return {"loop_depth": 0, "switch_depth": 0}


def test_empty_block_is_not_a_source_of_new_empty_blocks():
    options = operation_options(_member(), _scope(), [], set())
    assert "block" not in options
    assert {"call", "if", "return", "complete", "register"} <= set(options)
    already_implemented = [{"id": "n1", "op": "call"}]
    assert "block" in operation_options(_member(), _scope(), already_implemented, set())


def test_control_flow_depth_gate_preserves_direct_operations():
    options = operation_options(
        _member(), _scope(), [], set(),
        nesting_depth=MAX_CONTROL_FLOW_NESTING,
    )
    assert not {"if", "while", "for", "foreach", "block",
                "synchronized", "switch", "try", "lambda"} & set(options)
    assert {"call", "assign", "return", "register", "complete"} <= set(options)
    below_limit = operation_options(
        _member(), _scope(), [], set(),
        nesting_depth=MAX_CONTROL_FLOW_NESTING - 1,
    )
    assert "if" in below_limit


def test_nested_block_does_not_reset_stalled_active_helper():
    builder = _builder(None)
    member = {**builder.member, "registrar": False}
    epoch = builder._progress_epoch.get(member["id"], 0)
    builder._stalled_declarations[(member["id"], "method")] = epoch

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            assert task["block_path"] == ["n1", "body"]
            assert "method" not in parameters["properties"]["op"]["enum"]
            return {"op": "complete"}

    builder.router = Coder()
    assert builder.author_block(member, builder.scope, ["n1", "body"]) == []
    builder._record_progress(member)
    assert builder._progress_epoch[member["id"]] == epoch + 1


def test_no_executable_progress_is_bounded_per_member_across_scopes():
    builder = _builder(None)
    member = builder.member
    with pytest.raises(RuntimeError, match="INCREMENTAL_JAVA_NO_SEMANTIC_PROGRESS"):
        for _ in range(96):
            builder._record_nonprogress(member)

    builder._record_progress(member)
    assert builder._nonprogress_choices[member["id"]] == 0
    builder._record_nonprogress(member)
    assert builder._nonprogress_choices[member["id"]] == 1


def test_deep_scope_cannot_reopen_recursive_control_flow():
    builder = _builder(None)

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            options = parameters["properties"]["op"]["enum"]
            assert "if" not in options
            assert "block" not in options
            assert "call" in options
            return {"op": "complete"}

    builder.router = Coder()
    builder.member.pop("registrar", None)  # This test targets generic nested helper scopes.
    path = [part for i in range(MAX_CONTROL_FLOW_NESTING)
            for part in (f"n{i + 1}", "body")]
    assert builder.author_block(builder.member, builder.scope, path) == []


@pytest.mark.parametrize("idle, lookup_available", [(12, True), (32, False)])
def test_stalled_model_is_guided_to_an_executable_action(idle, lookup_available):
    builder = _builder(None)
    member = builder.member
    builder._nonprogress_choices[member["id"]] = idle

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            options = parameters["properties"]["op"]["enum"]
            assert "method" not in options
            assert "block" not in options
            assert "if" not in options
            assert ("lookup" in options) is lookup_available
            assert "return" in options
            return {"op": "return"}

    builder.router = Coder()
    body = builder.author_block(member, builder.scope, [])
    assert [node["op"] for node in body] == ["return"]
    assert builder._nonprogress_choices[member["id"]] == 0


def test_noop_callback_and_lifecycle_are_not_marked_implemented():
    from minecraft_mod_ai.incremental_java_author import _assert_live_behavior_entrypoints
    with pytest.raises(RuntimeError, match="INCREMENTAL_JAVA_NO_OBSERVABLE_ACTION"):
        _assert_live_behavior_entrypoints({"members": [{
            "id": "m1", "registrar": True,
            "body": [{"id": "n1", "op": "block", "body": [
                {"id": "n2", "op": "return", "value": ""}]}],
        }]})
    with pytest.raises(RuntimeError, match="INCREMENTAL_JAVA_NO_OBSERVABLE_ACTION"):
        _assert_live_behavior_entrypoints({"members": [{
            "id": "m2", "event": {"id": "registered"},
            "body": [{"id": "n1", "op": "if", "body": [], "otherwise": []}],
        }]})


def test_real_nested_effect_passes_entrypoint_gate():
    from minecraft_mod_ai.incremental_java_author import _assert_live_behavior_entrypoints
    _assert_live_behavior_entrypoints({"members": [{
        "id": "m1", "registrar": True,
        "body": [{"id": "n1", "op": "register"}],
    }, {
        "id": "m2", "event": {"id": "registered"},
        "body": [{"id": "n2", "op": "if", "body": [
            {"id": "n3", "op": "assign", "target": "this.count",
             "operator": "+=", "value": "1"}], "otherwise": []}],
    }]})


def test_behavior_entrypoints_cannot_complete_without_observable_action():
    registrar = {**_member(), "registrar": True}
    assert "complete" not in operation_options(registrar, _scope(), [], set())
    wrapper_only = [{"id": "n1", "op": "if", "body": [], "otherwise": []}]
    assert "complete" not in operation_options(registrar, _scope(), wrapper_only, set())
    connected = [{"id": "n1", "op": "register"}]
    assert "complete" in operation_options(registrar, _scope(), connected, set())
    listener = {**_member(), "event": {"event_owner": "example.Events"}}
    assert "complete" not in operation_options(listener, _scope(), wrapper_only, set())
    effected = [{"id": "n1", "op": "if", "body": [
        {"id": "n2", "op": "assign", "target": "this.value", "value": "1"}],
        "otherwise": []}]
    assert "complete" in operation_options(listener, _scope(), effected, set())
