"""Regression coverage for semantic liveness and recursive Java expression depth."""
from __future__ import annotations

from contextlib import ExitStack
import json

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
    assert coder.requests <= 100
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
