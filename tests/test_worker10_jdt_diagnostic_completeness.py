from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import java_lsp
from minecraft_mod_ai.java_lsp_process_safety_contract import install


class _FakeRpc:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages: queue.Queue[dict[str, object]] = queue.Queue()
        for message in messages:
            self.messages.put(message)
        self.process = SimpleNamespace(poll=lambda: None)
        self.stderr: list[str] = []
        self._mmm_reader_failure = None
        self.sent: list[dict[str, object]] = []

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)


def _published(uri: str, diagnostics: object) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": uri, "diagnostics": diagnostics},
    }


def _collector():
    install(java_lsp)
    return java_lsp._collect_diagnostics


def test_runtime_installs_complete_diagnostic_gate() -> None:
    collect = _collector()
    assert getattr(collect, "_mmm_jdt_complete_diagnostics", False)


def test_no_diagnostics_for_opened_file_fails_closed() -> None:
    collect = _collector()
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match="did not publish diagnostics for every opened Java file",
    ):
        collect(
            _FakeRpc([]),
            expected_uris={"file:///A.java"},
            timeout_seconds=0.01,
            quiet_seconds=0.0,
        )


def test_partial_diagnostics_for_opened_files_fail_closed() -> None:
    collect = _collector()
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match=r"observed=1, expected=2, missing=1",
    ):
        collect(
            _FakeRpc([_published("file:///A.java", [])]),
            expected_uris={"file:///A.java", "file:///B.java"},
            timeout_seconds=0.01,
            quiet_seconds=0.0,
        )


def test_complete_diagnostics_return_without_waiting_for_quiet_period() -> None:
    collect = _collector()
    error = {
        "severity": 1,
        "message": "cannot find symbol",
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
    }
    result = collect(
        _FakeRpc(
            [
                _published("file:///A.java", []),
                _published("file:///B.java", [error]),
            ]
        ),
        expected_uris={"file:///A.java", "file:///B.java"},
        timeout_seconds=10.0,
        quiet_seconds=10.0,
    )
    assert set(result) == {"file:///A.java", "file:///B.java"}
    assert result["file:///A.java"] == []
    assert result["file:///B.java"][0]["severity"] == 1


def test_malformed_expected_diagnostics_payload_fails_closed() -> None:
    collect = _collector()
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match="malformed diagnostics payload",
    ):
        collect(
            _FakeRpc([_published("file:///A.java", {"not": "a list"})]),
            expected_uris={"file:///A.java"},
            timeout_seconds=1.0,
            quiet_seconds=0.0,
        )
