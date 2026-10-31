"""Regression contracts for durable member-level Java generation."""
from __future__ import annotations

import pytest
import json

from minecraft_mod_ai.construction_journal import JournaledConstructionRouter


FRAME = {"kind": "method", "prefix": "public void ping() {\n",
         "suffix": "\n}"}
MESSAGES = [{"role": "user", "content": "Implement ping"}]
SPEC = {"language": "java", "target_path": "src/main/java/example/Ping.java"}
CONTEXT = {"target": {"loader": "fabric", "minecraft": "26.2"},
           "prerequisite_hashes": {}, "authoring_contract": {}}


class Backend:
    def __init__(self, *, error=None):
        self.calls = 0
        self.error = error

    def generate_java_source(self, role, messages, *, data_name, source_frame):
        self.calls += 1
        if self.error:
            raise self.error
        return source_frame["prefix"] + "    return;" + source_frame["suffix"]


def journal(tmp_path, backend):
    return JournaledConstructionRouter(backend, tmp_path, SPEC, CONTEXT)


def test_completed_member_replays_without_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "test-fingerprint")
    first_backend = Backend()
    output = journal(tmp_path, first_backend).generate_java_source(
        "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    second_backend = Backend(error=AssertionError("model must not run"))
    assert journal(tmp_path, second_backend).generate_java_source(
        "coder", MESSAGES, data_name="ping_body", source_frame=FRAME) == output
    assert first_backend.calls == 1
    assert second_backend.calls == 0


def test_member_replay_rejects_changed_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "test-fingerprint")
    journal(tmp_path, Backend()).generate_java_source(
        "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    with pytest.raises(ValueError, match="CONSTRUCTION_INFERENCE_REPLAY_DRIFT"):
        journal(tmp_path, Backend()).generate_java_source(
            "coder", [{"role": "user", "content": "Different behavior"}],
            data_name="ping_body", source_frame=FRAME)


def test_failed_member_is_not_checkpointed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "test-fingerprint")
    first = journal(tmp_path, Backend(error=RuntimeError("degenerate output")))
    with pytest.raises(RuntimeError, match="degenerate output"):
        first.generate_java_source(
            "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    second = Backend()
    assert "return;" in journal(tmp_path, second).generate_java_source(
        "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    assert second.calls == 1


def test_failed_member_can_retry_on_the_same_journal_without_a_cursor_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "test-fingerprint")
    backend = Backend(error=RuntimeError("invalid model response"))
    current = journal(tmp_path, backend)
    with pytest.raises(RuntimeError, match="invalid model response"):
        current.generate_java_source("coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    backend.error = None
    assert "return;" in current.generate_java_source(
        "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    assert backend.calls == 2


def test_metadata_key_order_after_reload_does_not_change_member_request_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "test-fingerprint")

    class MetadataBackend(Backend):
        def generate_structured_data(self, *args, **kwargs):
            return {"visibility": "public", "static": False, "return_type": "void"}

    parameters = {"type": "object"}
    first = journal(tmp_path, MetadataBackend())
    metadata = first.generate_structured_data(
        "coder", MESSAGES, data_name="metadata", parameters=parameters)
    expected = first.generate_java_source("coder", [{"role": "user", "content": json.dumps(metadata)}],
        data_name="ping_body", source_frame=FRAME)
    backend = MetadataBackend(error=AssertionError("completed member must replay"))
    second = journal(tmp_path, backend)
    reloaded = second.generate_structured_data(
        "coder", MESSAGES, data_name="metadata", parameters=parameters)
    assert list(metadata) != list(reloaded)  # Durable JSON is key-sorted.
    assert second.generate_java_source("coder", [{"role": "user", "content": json.dumps(reloaded)}],
        data_name="ping_body", source_frame=FRAME) == expected
    assert backend.calls == 0


def test_structured_journal_write_failure_restores_cursor_and_replays_after_retry(
    tmp_path, monkeypatch,
):
    from pathlib import Path
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "durable-structured-io-retry-test")
    class StructuredBackend(Backend):
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            self.calls += 1
            if self.error:
                raise self.error
            return {"value": "accepted"}
    backend = StructuredBackend()
    current = journal(tmp_path, backend)
    original_replace = Path.replace
    failed = []
    def fail_once(self, target):
        if str(target) == str(current.path) and not failed:
            failed.append(True)
            raise OSError("simulated journal disk write failure")
        return original_replace(self, target)
    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="simulated journal disk write failure"):
        current.generate_structured_data(
            "coder", MESSAGES, data_name="metadata",
            parameters={"type": "object"})
    assert current.position == 0
    assert current.rows == []
    assert not current.path.exists()
    assert current.generate_structured_data(
        "coder", MESSAGES, data_name="metadata",
        parameters={"type": "object"}) == {"value": "accepted"}
    assert backend.calls == 2
    assert current.position == 1
    assert len(current.rows) == 1
    fresh = journal(tmp_path, StructuredBackend(error=AssertionError("must replay")))
    assert fresh.generate_structured_data(
        "coder", MESSAGES, data_name="metadata",
        parameters={"type": "object"}) == {"value": "accepted"}


def test_java_journal_write_failure_does_not_leave_unsaved_member_in_memory(
    tmp_path, monkeypatch,
):
    from pathlib import Path
    monkeypatch.setattr(
        "minecraft_mod_ai.construction_implementation.construction_implementation_fingerprint",
        lambda: "durable-java-io-retry-test")
    backend = Backend()
    current = journal(tmp_path, backend)
    original_replace = Path.replace
    failed = []
    def fail_once(self, target):
        if str(target) == str(current.path) and not failed:
            failed.append(True)
            raise OSError("simulated member journal failure")
        return original_replace(self, target)
    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="simulated member journal failure"):
        current.generate_java_source(
            "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    assert current.position == 0
    assert current.rows == []
    assert not current.path.exists()
    assert "return;" in current.generate_java_source(
        "coder", MESSAGES, data_name="ping_body", source_frame=FRAME)
    assert backend.calls == 2
    assert current.position == 1
    assert len(current.rows) == 1
