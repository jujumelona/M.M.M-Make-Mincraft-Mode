"""Regression coverage for semantic liveness and recursive Java expression depth."""
from __future__ import annotations

from contextlib import ExitStack
import pytest

from minecraft_mod_ai.incremental_java_author import _semantic_operation_key
from minecraft_mod_ai.incremental_java_values_model import (
    MAX_VALUE_EXPRESSION_NESTING, value_frame,
)
from test_incremental_java_recovery import _builder


def test_local_preparation_does_not_count_as_runtime_effect():
    scope = {"locals": [{"name": "temporary", "type": "int"}]}
    assert _semantic_operation_key(
        {"id": "n1", "op": "local", "name": "temporary", "value": "0"}, scope
    ) is None
    assert _semantic_operation_key(
        {"id": "n2", "op": "assign", "target": "temporary", "value": "1"}, scope
    ) is None
    assert _semantic_operation_key(
        {"id": "n3", "op": "call", "method": "calculate", "result_type": "int"}, scope
    ) is None
    assert _semantic_operation_key(
        {"id": "n4", "op": "assign", "target": "this.count", "value": "1"}, scope
    ) is not None


def test_repeated_operations_share_semantic_signature_not_node_identity():
    scope = {"locals": []}
    first = {"id": "n1", "op": "call", "receiver": "this",
             "method": "observe", "arguments": [], "result_type": ""}
    second = {**first, "id": "n2147"}
    assert _semantic_operation_key(first, scope) == _semantic_operation_key(second, scope)


def test_local_churn_followed_by_duplicate_call_reaches_stall_guard(monkeypatch):
    from minecraft_mod_ai import incremental_java_operations as operations

    class Coder:
        def __init__(self):
            self.requests = 0

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name.endswith("_kind")
            options = parameters["properties"]["op"]["enum"]
            self.requests += 1
            return {"op": "local" if "local" in options else "call"}

    coder = Coder()
    builder = _builder(coder)

    def fake_operation(builder, operation):
        identity = builder.allocate("n")
        if operation == "local":
            name = "mmmValue" + builder.allocate("v")[1:]
            builder.scope["locals"].append({"name": name, "type": "int"})
            return {"id": identity, "op": "local", "name": name, "value": "0"}
        return {"id": identity, "op": "call", "method": "observe",
                "receiver": "this", "arguments": [], "result_type": ""}

    monkeypatch.setattr(operations, "author_operation", fake_operation)
    with pytest.raises(RuntimeError, match="INCREMENTAL_JAVA_NO_SEMANTIC_PROGRESS"):
        builder.author_block(builder.member, builder.scope, [])
    assert coder.requests <= 130
    assert builder._progress_epoch.get(builder.member["id"], 0) == 1


def test_value_expression_nesting_is_bounded_without_mutating_stack():
    builder = _builder(None)
    builder.value_stack = []
    with ExitStack() as stack:
        for i in range(MAX_VALUE_EXPRESSION_NESTING):
            stack.enter_context(value_frame(builder, "operand:" + str(i), "int"))
        with pytest.raises(RuntimeError, match="JAVA_VALUE_EXPRESSION_NESTING_EXHAUSTED"):
            with value_frame(builder, "one-more", "int"):
                pass
        assert len(builder.value_stack) == MAX_VALUE_EXPRESSION_NESTING
    assert builder.value_stack == []


def test_local_catalog_page_cycle_is_rejected(monkeypatch):
    from minecraft_mod_ai import incremental_java_values as values

    builder = _builder(None)
    builder.scope["locals"] = [
        {"name": "local" + str(i), "type": "int"} for i in range(20)
    ]

    def choose(_builder, schema, suffix, extra=None):
        assert suffix == "local_value"
        return {"name": "next_page" if extra["local_offset"] == 0 else "previous_page"}

    monkeypatch.setattr(values, "request", choose)
    with pytest.raises(RuntimeError, match="JAVA_LOCAL_SELECTION_NAVIGATION_CYCLE"):
        values.ValueAuthor(builder)._local()


def test_member_catalog_page_cycle_is_rejected(monkeypatch):
    from minecraft_mod_ai import incremental_java_values_members as members
    from minecraft_mod_ai.incremental_java_values_model import JavaReceiver

    builder = _builder(None)

    def catalog(_builder, receiver, kind, query, location, filters):
        if location[1] == 0:
            return ({"m1": {"name": "entry"}}, {"total": 20}, ("api", 10))
        return ({"m2": {"name": "entry2"}}, {"total": 20}, None)

    def choose(_builder, schema, suffix, extra=None):
        assert suffix == "field_member"
        return {"member_id": "next_page" if not extra.get("member_page", {}).get("offset") else "previous_page"}

    # Supply the real page offset in the test catalog, so selection alternates.
    def catalog_with_offset(_builder, receiver, kind, query, location, filters):
        rows, page, following = catalog(_builder, receiver, kind, query, location, filters)
        return rows, {**page, "offset": location[1]}, following

    monkeypatch.setattr(members, "_member_page", catalog_with_offset)
    monkeypatch.setattr(members, "request", choose)
    with pytest.raises(RuntimeError, match="JAVA_MEMBER_SELECTION_NAVIGATION_CYCLE"):
        members.select_member(builder, JavaReceiver("this", "Feature"), "FIELD", "field_member")


def test_repair_discovery_cannot_replace_a_concrete_edit_forever():
    from minecraft_mod_ai.incremental_java_repair import _select_repair

    class Builder:
        lookups = 0

        def ask(self, schema, suffix):
            options = schema["properties"]["repair"]["enum"]
            return {"repair": "lookup" if "lookup" in options else "mapped_operand"}

        def lookup(self):
            self.lookups += 1

    builder = Builder()
    assert _select_repair(builder, ["mapped_operand"]) == "mapped_operand"
    assert builder.lookups == 2


def test_inserted_repair_operations_share_original_error_budget():
    from minecraft_mod_ai.incremental_java_repair import _count_attempt

    builder = _builder(None)
    member = builder.member
    _count_attempt(builder, member, {"id": "n1"}, "value")
    _count_attempt(builder, member, {"id": "n2", "_repair_origin": "n1"}, "value")
    with pytest.raises(ValueError, match="JAVA_INCREMENTAL_OPERATION_REPAIR_EXHAUSTED"):
        _count_attempt(builder, member, {"id": "n3", "_repair_origin": "n1"}, "value")
