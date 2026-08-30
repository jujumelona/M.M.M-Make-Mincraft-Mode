from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import java_lsp


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


def test_empty_expected_uri_set_is_trivially_complete() -> None:
    assert java_lsp._collect_diagnostics(
        _FakeRpc([]),
        expected_uris=set(),
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    ) == {}


def test_no_diagnostics_for_opened_file_fails_closed() -> None:
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match="did not publish diagnostics for every opened Java file",
    ):
        java_lsp._collect_diagnostics(
            _FakeRpc([]),
            expected_uris={"file:///A.java"},
            timeout_seconds=0.01,
            quiet_seconds=0.0,
        )


def test_partial_diagnostics_for_opened_files_fail_closed() -> None:
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match=r"observed=1, expected=2, missing=1",
    ):
        java_lsp._collect_diagnostics(
            _FakeRpc([_published("file:///A.java", [])]),
            expected_uris={"file:///A.java", "file:///B.java"},
            timeout_seconds=0.01,
            quiet_seconds=0.0,
        )


def test_complete_diagnostics_return_when_coverage_and_quiet_are_satisfied() -> None:
    error = {
        "severity": 1,
        "message": "cannot find symbol",
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
    }
    result = java_lsp._collect_diagnostics(
        _FakeRpc(
            [
                _published("file:///A.java", []),
                _published("file:///B.java", [error]),
            ]
        ),
        expected_uris={"file:///A.java", "file:///B.java"},
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )
    assert set(result) == {"file:///A.java", "file:///B.java"}
    assert result["file:///A.java"] == []
    assert result["file:///B.java"][0]["severity"] == 1


def test_republished_diagnostics_replace_initial_empty_result_before_settle() -> None:
    error = {"severity": 1, "message": "late compiler error"}
    result = java_lsp._collect_diagnostics(
        _FakeRpc(
            [
                _published("file:///A.java", []),
                _published("file:///A.java", [error]),
            ]
        ),
        expected_uris={"file:///A.java"},
        timeout_seconds=1.0,
        quiet_seconds=0.01,
    )
    assert result["file:///A.java"] == [error]


def test_malformed_expected_diagnostics_payload_fails_closed() -> None:
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match="malformed diagnostics payload",
    ):
        java_lsp._collect_diagnostics(
            _FakeRpc([_published("file:///A.java", {"not": "a list"})]),
            expected_uris={"file:///A.java"},
            timeout_seconds=1.0,
            quiet_seconds=0.0,
        )


def test_reader_failure_while_collecting_diagnostics_fails_immediately() -> None:
    rpc = _FakeRpc([])
    rpc._mmm_reader_failure = RuntimeError("reader died")
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match="stdout reader failed while collecting diagnostics",
    ):
        java_lsp._collect_diagnostics(
            rpc,
            expected_uris={"file:///A.java"},
            timeout_seconds=10.0,
            quiet_seconds=0.0,
        )
