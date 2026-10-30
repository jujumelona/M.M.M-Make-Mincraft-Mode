"""Regression: inspected source hooks are host work, not a 9B menu loop.

The production-log 13 failure performed 96 semantic nonprogress decisions,
including repeated event cursor/cancel navigation without any Java operation.
These tests do not hard-code a gameplay request or trigger vocabulary.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.incremental_java_events import select_original_occurrence
from minecraft_mod_ai.incremental_java_operations import _register
from minecraft_mod_ai.source_occurrence_error import SourceHookUnavailable


def _event():
    return {
        "event_id": "inspected.events.Observed#EVENT",
        "event_owner": "inspected.events.Observed",
        "event_field": "EVENT",
        "parameters": ["java.lang.Object"],
        "return_type": "void",
    }


def test_explicit_original_hook_is_bound_without_model_inference():
    event = _event()

    class NoModel:
        def generate_structured_data(self, *args, **kwargs):
            pytest.fail("explicit, checked ABI binding must not consult the AI")

    builder = SimpleNamespace(
        member={"behavior": {
            "behavior_id": "b-original", "event_binding": event["event_id"],
            "trigger": "a custom runtime occurrence",
        }},
        api=SimpleNamespace(callbacks=lambda: [event]),
        router=NoModel(),
    )
    assert select_original_occurrence(builder)["event_id"] == event["event_id"]


def test_absent_observation_reports_producer_without_model_loop():
    class NoModel:
        def generate_structured_data(self, *args, **kwargs):
            pytest.fail("missing ABI is an upstream producer task, not a model menu")

    builder = SimpleNamespace(
        member={"behavior": {"behavior_id": "b-missing"}},
        api=SimpleNamespace(callbacks=lambda: []),
        router=NoModel(),
    )
    with pytest.raises(SourceHookUnavailable, match="no_inspected_callback_requires_source_producer"):
        select_original_occurrence(builder)


def test_ambiguous_hook_is_one_bounded_source_decision_not_event_paging():
    event = _event()

    class DecliningModel:
        calls = 0

        def generate_structured_data(self, role, messages, **kwargs):
            self.calls += 1
            assert role == "planner"
            assert "SOURCE_REQUIRED" in kwargs["parameters"]["properties"]["event_id"]["enum"]
            return {"event_id": "SOURCE_REQUIRED"}

    router = DecliningModel()
    builder = SimpleNamespace(
        member={"behavior": {"behavior_id": "b-unknown", "trigger": "new runtime interaction"}},
        api=SimpleNamespace(callbacks=lambda: [event] * 13),
        router=router,
    )
    with pytest.raises(SourceHookUnavailable, match="custom_runtime_hook_needed"):
        select_original_occurrence(builder)
    assert router.calls == 1


def _builder_without_model():
    builder = object.__new__(IncrementalJavaBuilder)
    builder.member = None
    builder.context = {"host_registration_enabled": True}
    builder.scope = {}
    builder.path = []
    builder.block = []
    builder.current_operation = ""
    builder.pending = deque()
    builder.history = {}
    builder._progress_epoch = {}
    builder._nonprogress_choices = {}
    builder._semantic_operation_keys = {}
    return builder


def test_world_registrar_is_rendered_once_by_host_without_kind_or_event_menu(monkeypatch):
    from minecraft_mod_ai import incremental_java_author as author
    from minecraft_mod_ai import incremental_java_progress as progress
    from minecraft_mod_ai import incremental_java_work as work

    builder = _builder_without_model()
    behavior = {"behavior_id": "b-runtime", "runtime_state_writes": ["world.currency"]}
    member = {"id": "m1", "kind": "method", "registrar": True,
              "behavior": behavior, "body": []}
    builder.allocate = lambda prefix: "n1"
    builder.api = SimpleNamespace(bindings={})

    calls = []
    def materialize(current, node_owner, operation):
        assert current is builder
        assert node_owner is member
        assert operation == "register"
        calls.append(operation)
        return {"id": "n1", "op": "register", "event_owner": "inspected.Event",
                "event_field": "EVENT", "handler": "handler1"}

    def finish(current, pending_before):
        assert current is builder
        assert pending_before == 0

    monkeypatch.setattr(author, "_safe_author_operation", materialize)
    monkeypatch.setattr(progress, "finish_new_members", finish)
    monkeypatch.setattr(work, "_world_bound", lambda b, m: True)

    def forbidden(*args, **kwargs):
        pytest.fail("registrar construction should not consume a model choice")

    builder.ask = forbidden
    assert builder.author_block(member, {}, [], body=member["body"]) is member["body"]
    assert len(calls) == 1
    assert len(member["body"]) == 1
    assert builder.author_block(member, {}, [], body=member["body"]) is member["body"]
    assert len(calls) == 1


def test_registrar_rejects_duplicate_source_hooks_before_ai_turn(monkeypatch):
    from minecraft_mod_ai import incremental_java_work as work
    builder = _builder_without_model()
    member = {"id": "m2", "kind": "method", "registrar": True, "behavior": {},
              "body": [{"id": "n1", "op": "register"},
                       {"id": "n2", "op": "register"}]}
    monkeypatch.setattr(work, "_world_bound", lambda b, m: True)
    with pytest.raises(RuntimeError, match="JAVA_DUPLICATE_ORIGINAL_OCCURRENCE_REGISTRATION"):
        builder.author_block(member, {}, [], body=member["body"])


def test_incremental_registration_uses_host_binding_instead_of_menu(monkeypatch):
    from minecraft_mod_ai import incremental_java_events as events

    chosen = _event()
    requests = []
    monkeypatch.setattr(events, "select_original_occurrence", lambda builder: (
        requests.append(builder.member["behavior"]["behavior_id"]) or chosen))
    class Builder:
        context = {"host_registration_enabled": True}
        member = {"owner": "Owner", "behavior": {"behavior_id": "b-1"}}
        def new_method(self, **kwargs):
            return {"name": "listener", "metadata": kwargs["metadata"]}
    node = {"id": "n1", "op": "register"}
    _register(Builder(), node)
    assert requests == ["b-1"]
    assert node["event_owner"] == chosen["event_owner"]
    assert node["handler"] == "listener"
