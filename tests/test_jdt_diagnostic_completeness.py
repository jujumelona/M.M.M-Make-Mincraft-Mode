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


def test_malformed_diagnostic_item_fails_closed() -> None:
    with pytest.raises(
        java_lsp.JDTLanguageServerError,
        match="malformed diagnostics payload",
    ):
        java_lsp._collect_diagnostics(
            _FakeRpc([_published("file:///A.java", [{"severity": 1}, "bad-item"])]),
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


def test_explicit_java_file_cannot_escape_project_root(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "Outside.java"
    outside.write_text("class Outside {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical project-relative|escaped"):
        java_lsp._java_files(project, ("../Outside.java",))


def test_explicit_java_file_symlink_is_rejected(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    real = project / "Real.java"
    real.write_text("class Real {}\n", encoding="utf-8")
    alias = project / "Alias.java"
    try:
        alias.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        java_lsp._java_files(project, ("Alias.java",))


def test_explicit_java_file_symlinked_parent_is_rejected(tmp_path) -> None:
    project = tmp_path / "project"
    real_dir = project / "real"
    alias_dir = project / "alias"
    real_dir.mkdir(parents=True)
    (real_dir / "A.java").write_text("class A {}\n", encoding="utf-8")
    try:
        alias_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        java_lsp._java_files(project, ("alias/A.java",))
