from __future__ import annotations

import json

import minecraft_mod_ai.root_cause_trace as trace


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_root_cause_trace_is_durable_and_preserves_first_failure(tmp_path, monkeypatch):
    trace_path = tmp_path / "run" / "root_cause.jsonl"
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(trace_path))

    with trace.trace_scope("durable-test", trace_id="trace-test"):
        trace.emit_root_cause("start", result="START")
        first = RuntimeError("primary")
        trace.emit_root_cause("first_failure", result="FAIL", exc=first)
        trace.emit_root_cause("secondary_failure", result="ERROR", reason="secondary")

    records = _records(trace_path)
    assert [item["event"] for item in records] == [
        "start",
        "first_failure",
        "secondary_failure",
    ]
    first_failure = records[1]
    secondary_failure = records[2]
    assert first_failure["schema_version"] == "mmm/root-cause-trace-v3"
    assert first_failure["is_first_failure"] is True
    assert secondary_failure["is_first_failure"] is False
    assert secondary_failure["first_failure_seq"] == first_failure["first_failure_seq"]
    assert first_failure["exception_chain"][0]["type"] == "RuntimeError"


def test_trace_serialization_failure_uses_emergency_record(tmp_path, monkeypatch):
    trace_path = tmp_path / "root_cause.jsonl"
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(trace_path))

    real_dumps = trace.json.dumps
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TypeError("synthetic serializer failure")
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(trace.json, "dumps", fail_once)

    # The diagnostic failure must never escape and replace the production failure.
    production_error = ValueError("original production failure")
    trace.emit_root_cause(
        "production_failure",
        result="FAIL",
        exc=production_error,
        details={"unsafe": object()},
    )

    records = _records(trace_path)
    assert records[-1]["event"] == "trace_emergency_fallback"
    assert records[-1]["original_event"] == "production_failure"
    assert records[-1]["original_exception_type"] == "ValueError"
    assert records[-1]["logger_exception_type"] == "TypeError"
