"""Regression contracts for durable member-level Java generation."""
from __future__ import annotations

import pytest

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
