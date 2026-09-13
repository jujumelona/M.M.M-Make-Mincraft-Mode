from __future__ import annotations

import io
import json
from collections import deque
from types import SimpleNamespace

from minecraft_mod_ai.java_lsp import _JsonRpcProcess
from minecraft_mod_ai.root_cause_trace import emit_root_cause


class Notebook(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1


def test_jdt_stderr_is_emitted_as_it_is_read(monkeypatch, tmp_path):
    notebook = Notebook()
    monkeypatch.setattr("sys.stderr", notebook)
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    rpc = object.__new__(_JsonRpcProcess)
    rpc.process = SimpleNamespace(stderr=io.BytesIO(b"JDT importing Gradle project\n"), pid=123)
    rpc.stderr = deque(maxlen=30)
    rpc._read_stderr()
    events = [json.loads(line.split("ROOT CAUSE TRACE: ", 1)[1]) for line in notebook.getvalue().splitlines()]
    event = next(row for row in events if row["event"] == "jdt_stderr")
    assert event["details"]["line"] == "JDT importing Gradle project"
    assert list(rpc.stderr) == ["JDT importing Gradle project"]
    assert notebook.flushes > 0


def test_retry_event_flushes_notebook_output_immediately(monkeypatch, tmp_path):
    notebook = Notebook()
    monkeypatch.setattr("sys.stderr", notebook)
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    emit_root_cause("mcp_verifier_transport_retry", result="RETRY", details={"failure_count": 1})
    assert "mcp_verifier_transport_retry" in notebook.getvalue()
    assert notebook.flushes > 0


def test_jdt_stderr_redacts_credentials_in_cell_and_durable_trace(monkeypatch, tmp_path):
    notebook = Notebook()
    monkeypatch.setattr("sys.stderr", notebook)
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    rpc = object.__new__(_JsonRpcProcess)
    rpc.process = SimpleNamespace(stderr=io.BytesIO(b"authorization: Bearer example-private-value\n"), pid=123)
    rpc.stderr = deque(maxlen=30)
    rpc._read_stderr()
    assert "[REDACTED]" in notebook.getvalue()
    assert "example-private-value" not in notebook.getvalue()
    assert "example-private-value" not in "".join(rpc.stderr)
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert "example-private-value" not in artifact.read_text(encoding="utf-8")


def test_jdt_stderr_redacts_multiline_private_key(monkeypatch, tmp_path):
    notebook = Notebook()
    monkeypatch.setattr("sys.stderr", notebook)
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    rpc = object.__new__(_JsonRpcProcess)
    rpc.process = SimpleNamespace(
        stderr=io.BytesIO(b"-----BEGIN RSA PRIVATE KEY-----\nprivate-material\n-----END RSA PRIVATE KEY-----\nJDT ready\n"),
        pid=123,
    )
    rpc.stderr = deque(maxlen=30)
    rpc._read_stderr()
    assert "JDT ready" in notebook.getvalue()
    assert "private-material" not in notebook.getvalue()
    assert "private-material" not in "".join(rpc.stderr)
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert "private-material" not in artifact.read_text(encoding="utf-8")
