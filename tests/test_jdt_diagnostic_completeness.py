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
        self.requests: list[tuple[str, dict[str, object], float]] = []

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    def request(self, method: str, params: dict[str, object], timeout: float):
        self.requests.append((method, params, timeout))
        return None


def _published(uri: str, diagnostics: object) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": uri, "diagnostics": diagnostics},
    }


def test_explicit_refresh_requests_each_open_document_with_lifecycle_fence() -> None:
    rpc = _FakeRpc([])
    deadline = java_lsp.time.monotonic() + 1.0

    java_lsp._refresh_open_document_diagnostics(
        rpc,
        expected_uris={"file:///B.java", "file:///A.java"},
        deadline=deadline,
    )

    assert [method for method, _params, _timeout in rpc.requests] == [
        "workspace/executeCommand",
        "workspace/executeCommand",
    ]
    assert [params for _method, params, _timeout in rpc.requests] == [
        {
            "command": "java.project.refreshDiagnostics",
            "arguments": ["file:///A.java", "thisFile", False, True],
        },
        {
            "command": "java.project.refreshDiagnostics",
            "arguments": ["file:///B.java", "thisFile", False, True],
        },
    ]
    assert all(timeout > 0 for _method, _params, timeout in rpc.requests)


def test_clean_file_diagnostics_are_forced_instead_of_inferred_from_silence(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)
    source = source_root / "Clean.java"
    source.write_text("final class Clean {}\n", encoding="utf-8")

    class RefreshingRpc(_FakeRpc):
        def __init__(self) -> None:
            super().__init__([])
            self.notifications: list[tuple[str, dict[str, object]]] = []

        def notify(self, method: str, params: dict[str, object]) -> None:
            self.notifications.append((method, params))

        def request(self, method: str, params: dict[str, object], timeout: float):
            self.requests.append((method, params, timeout))
            assert method == "workspace/executeCommand"
            arguments = params["arguments"]
            assert isinstance(arguments, list)
            uri = str(arguments[0])
            self.messages.put(_published(uri, []))
            return None

    service = java_lsp.JavaLanguageService(diagnostic_quiet_seconds=0.0)
    rpc = RefreshingRpc()
    monkeypatch.setattr(java_lsp, "assert_server_safe_source_sets", lambda _root: None)
    monkeypatch.setattr(
        service,
        "_ensure_rpc_locked",
        lambda _root, *, timeout_seconds, deadline=None: rpc,
    )

    result = service.diagnostics(tmp_path, timeout_seconds=1)

    uri = source.resolve().as_uri()
    assert result["diagnostics"] == {uri: []}
    assert result["files_opened"] == 1
    assert result["error_count"] == 0
    assert result["warning_count"] == 0
    assert rpc.requests[0][1] == {
        "command": "java.project.refreshDiagnostics",
        "arguments": [uri, "thisFile", False, True],
    }


def test_equivalent_file_uri_spellings_match_diagnostics() -> None:
    message = _published("file:/tmp/Project/A.java", [])
    matched = java_lsp._published_diagnostics(
        message,
        {"file:///tmp/Project/A.java"},
    )

    assert matched == ("file:///tmp/Project/A.java", [])


def test_percent_encoded_file_uri_matches_open_document() -> None:
    message = _published("file:///tmp/My%20Project/A.java", [])
    matched = java_lsp._published_diagnostics(
        message,
        {"file:///tmp/My%20Project/A.java"},
    )

    assert matched == ("file:///tmp/My%20Project/A.java", [])


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
