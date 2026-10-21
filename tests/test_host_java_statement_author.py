"""Structural regression: Fabric callback authoring no longer accepts whole Java methods.

Replays the real 2026-10-21 correction error: a single implementation field
contained several private methods and their registrations instead of the
interior of the host-owned handler.
"""
from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.host_java_statement_author import (
    parse_one_java_statement, author_java_callback_statements,
)


_FRAME = {
    "kind": "method",
    "prefix": "private int callback(java.lang.Object entity) {\n",
    "suffix": "\n}",
}


class FakeStatementRouter:
    def __init__(self, statements):
        self.statements = iter(statements)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        assert role == "coder"
        assert parameters["properties"]["statement"]["maxLength"] == 2048
        assert parameters["additionalProperties"] is False
        self.calls.append((data_name, messages))
        return {"statement": next(self.statements)}


def _messages():
    return [
        {"role": "user", "content": json.dumps({
            "inspected_target_api": {
                "target_api": [{"owner": "example.VerifiedApi", "kind": "METHOD",
                                "name": "apply"}],
                "class_members": [],
            },
        })},
        {"role": "user", "content": json.dumps({
            "response_source_frame": _FRAME,
            "assigned_behavior": {
                "behavior_id": "b1", "intent": "Handle a verified event",
                "effects": "Make a concrete effect"},
            "current_unit_purpose": "Perform the event effect",
            "declared_member_api": [{"signature": "private int callback(java.lang.Object entity)"}],
        })},
    ]


def test_parser_admits_one_statement_and_rejects_sibling_java_declarations():
    assert parse_one_java_statement("return 1;") == "return 1;"
    assert parse_one_java_statement("if (true) { System.out.println(1); }") == (
        "if (true) { System.out.println(1); }")
    with pytest.raises(ValueError):
        parse_one_java_statement("return 1;\nreturn 2;")
    with pytest.raises(ValueError):
        parse_one_java_statement(
            "private int handler() { return 1; }\n"
            "private void registerBehavior() { }"
        )
    with pytest.raises(ValueError):
        parse_one_java_statement("import java.util.List;")


def test_host_assembles_schema_constrained_statements_under_fixed_signature():
    router = FakeStatementRouter([
        "System.out.println(entity);", "return 1;", "",
    ])
    source = author_java_callback_statements(
        router, _messages(), data_name="handle_event",
        frame=_FRAME, require_effect=True,
    )
    assert source == (
        _FRAME["prefix"] +
        "System.out.println(entity);\nreturn 1;" + _FRAME["suffix"]
    )
    assert len(router.calls) == 3
    assert all("statement_" in action for action, _ in router.calls)
    assert "response_source_frame" not in router.calls[0][1][-2]["content"]


def test_production_whole_method_output_can_never_escape_host_frame():
    invalid = (
        "private int callback(java.lang.Object entity) {\n"
        "return 1;\n}\n"
        "private void registerBehavior() { System.out.println(1); }"
    )
    router = FakeStatementRouter([
        invalid,
        "System.out.println(entity);",
        "return 1;",
        "",
    ])
    source = author_java_callback_statements(
        router, _messages(), data_name="bad_handler",
        frame=_FRAME, require_effect=True,
    )
    assert "registerBehavior" not in source
    assert source.count("callback(") == 1
    assert len(router.calls) == 4
    assert "last_error" in router.calls[1][1][-1]["content"]


def test_undecodable_twice_fails_without_adding_method_declarations():
    invalid = "private void registerBehavior() { }"
    router = FakeStatementRouter([invalid, invalid])
    with pytest.raises(ValueError, match="JAVA_CALLBACK_ATOMIC_STATEMENT_INVALID"):
        author_java_callback_statements(
            router, _messages(), data_name="two_rejections",
            frame=_FRAME, require_effect=True,
        )


def test_return_only_handler_is_not_a_fake_completed_feature():
    router = FakeStatementRouter(["return 0;", ""])
    with pytest.raises(ValueError, match="JAVA_CALLBACK_EFFECT_REQUIRED"):
        author_java_callback_statements(
            router, _messages(), data_name="no_effect",
            frame=_FRAME, require_effect=True,
        )


def test_missing_typed_state_accessor_is_rejected_before_next_statement():
    frame = {**_FRAME, "host_state_access": [{
        "owner": "example.AuthoredStateModel",
        "methods": [{"name": "readState_credits",
                     "signature": "public static int readState_credits();"}],
    }]}
    invalid = "example.AuthoredStateModel.readState_missing();"
    router = FakeStatementRouter([invalid, invalid])
    with pytest.raises(ValueError, match="JAVA_CALLBACK_ATOMIC_STATEMENT_INVALID"):
        author_java_callback_statements(
            router, _messages(), data_name="unknown_state",
            frame=frame, require_effect=True,
        )
