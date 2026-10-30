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
    assert "block" not in operation_options(_member(), _scope(), already_implemented, set())


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
    member = builder.member
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


def test_author_scope_restores_after_failed_model_page():
    builder = _builder(None)
    member_before, scope_before = builder.member, builder.scope
    path_before, block_before = builder.path, builder.block
    operation_before = builder.current_operation

    class FailingCoder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            raise RuntimeError("intentional model transport failure")

    builder.router = FailingCoder()
    with pytest.raises(RuntimeError, match="intentional model transport failure"):
        builder.author_block(member_before, dict(scope_before), ["n1", "body"])

    assert builder.member is member_before
    assert builder.scope is scope_before
    assert builder.path is path_before
    assert builder.block is block_before
    assert builder.current_operation == operation_before


def test_current_generated_helper_is_not_callable_but_other_pending_helpers_are():
    from minecraft_mod_ai.incremental_java_calls import _generated_methods
    from minecraft_mod_ai.incremental_java_model import method_metadata

    builder = _builder(None)
    active = builder.member
    active["kind"] = "method"
    active["name"] = "mmmHelper3"
    active["implementation_status"] = "authoring"
    assert _generated_methods(builder, active["owner"], "mmmHelper3") == []

    other = {"id": "m99", "kind": "method", "owner": active["owner"],
             "name": "distinctHelper", "implementation_status": "pending",
             "metadata": method_metadata(), "body": []}
    builder.program["members"].append(other)
    assert len(_generated_methods(builder, active["owner"], "distinctHelper")) == 1


def test_block_free_operation_catalog_still_allows_genuine_control_flow():
    body = [{"id": "n1", "op": "local", "name": "temporary", "value": "0"}]
    options = operation_options(_member(), _scope(), body, set(), nesting_depth=7)
    assert "block" not in options
    assert {"call", "assign", "if", "while", "try", "return", "complete"} <= set(options)


def test_active_helper_self_call_is_rejected_before_accepting_node():
    from minecraft_mod_ai.incremental_java_author import _reject_current_self_call

    member = {"id": "m3", "name": "mmmHelper3", "kind": "method"}
    with pytest.raises(RuntimeError, match="JAVA_ACTIVE_MEMBER_SELF_CALL: m3"):
        _reject_current_self_call(member, {
            "op": "call", "receiver": "this", "method": "mmmHelper3"})
    _reject_current_self_call(member, {
        "op": "call", "receiver": "this", "method": "otherHelper"})
    _reject_current_self_call(member, {"op": "local", "name": "temporary"})
    # A distinct overload is legal Java: it must remain callable.
    from minecraft_mod_ai.incremental_java_model import method_metadata
    method = {**member, "metadata": method_metadata()}
    _reject_current_self_call(method, {
        "op": "call", "receiver": "this", "method": "mmmHelper3",
        "call_contract": {"kind": "METHOD", "owner": "Feature",
                          "name": "mmmHelper3", "parameters": ["int"]}})


def test_active_method_is_absent_from_actual_generated_value_call_page():
    """Reproduce the exact model page that exposed m3 in production log 11."""
    from minecraft_mod_ai.incremental_java_values_members import _generated_page
    from minecraft_mod_ai.incremental_java_values_model import JavaReceiver
    from minecraft_mod_ai.incremental_java_model import method_metadata

    builder = _builder(None)
    active = builder.new_method(owner=builder.program["class_name"],
                                purpose="Work", behavior={},
                                metadata=method_metadata(), name="mmmHelper3")
    builder.member = active
    active["implementation_status"] = "authoring"
    other = builder.new_method(owner=active["owner"], purpose="Other work",
                               behavior={}, metadata=method_metadata(), name="otherHelper")
    receiver = JavaReceiver("this", active["owner"])
    rows, total = _generated_page(builder, receiver, "METHOD", "", 0,
                                  {"nonvoid": False, "writable": False})
    assert active["id"] not in rows
    assert other["id"] in rows
    assert total == len(rows)


def test_late_self_call_is_rejected_as_a_recoverable_choice(monkeypatch):
    from minecraft_mod_ai.incremental_java_author import _safe_author_operation
    from minecraft_mod_ai import incremental_java_operations as operations

    builder = _builder(None)
    member = builder.member
    member["name"] = "mmmHelper3"
    monkeypatch.setattr(operations, "author_operation", lambda *_: {
        "id": "n90", "op": "call", "receiver": "this",
        "method": "mmmHelper3", "call_contract": {"parameters": []}})
    result = _safe_author_operation(builder, member, "call")
    assert result is None
    assert builder.rejection["operand_diagnostic"] == (
        "JAVA_ACTIVE_MEMBER_SELF_CALL: " + member["id"])


def test_late_guard_does_not_discard_other_valid_overloads(monkeypatch):
    from minecraft_mod_ai.incremental_java_author import _safe_author_operation
    from minecraft_mod_ai import incremental_java_operations as operations

    builder = _builder(None)
    member = builder.member
    member["name"] = "mmmHelper3"
    node = {"id": "n91", "op": "call", "receiver": "this",
            "method": "otherHelper", "call_contract": {"parameters": []}}
    monkeypatch.setattr(operations, "author_operation", lambda *_: node)
    assert _safe_author_operation(builder, member, "call") == node


def test_each_model_page_preserves_host_state_work_order():
    """Gameplay obligations must survive the host-to-small-coder projection."""
    builder = _builder(None)
    scope = {"required_state_transition": "a verified gameplay state mutation",
             "missing_producer": "author the state producer before the dependent effect"}
    builder.api.context["host_state_work_order_scope"] = scope
    viewed = builder.api.view(builder.member, builder.scope, builder.program)
    assert viewed["host_state_work_order_scope"] == scope
    assert "assigned_behavior" in viewed
    assert "runtime_state_bindings" in viewed
