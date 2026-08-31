from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from minecraft_mod_ai import java_lsp
from minecraft_mod_ai.java_lsp import JDTLanguageServerError, JavaLanguageService


class _FakeJsonRpcProcess:
    instances: list[_FakeJsonRpcProcess] = []

    def __init__(self, command: list[str], cwd: Path) -> None:
        self.command = command
        self.cwd = cwd
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr = ["stable fake stderr"]
        self.request_timeouts: list[float] = []
        self.request_methods: list[str] = []
        self.opened_uris: list[str] = []
        self.workspace_folders: tuple[dict[str, str], ...] = ()
        self.closed = False
        self.__class__.instances.append(self)

    def request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        self.request_timeouts.append(timeout)
        self.request_methods.append(method)
        if method == "initialize":
            return {}
        if method == "workspace/symbol":
            return {"query": params.get("query", "")}
        raise AssertionError(f"unexpected fake JDT request: {method}")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        if method == "textDocument/didOpen":
            uri = str(params["textDocument"]["uri"])
            self.opened_uris.append(uri)
            number = int(Path(uri).stem.removeprefix("Generated"))
            severity = 1 if number % 2 == 0 else 2
            self.messages.put(
                {
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "diagnostics": [
                            {
                                "severity": severity,
                                "message": f"z-{number}",
                            },
                            {
                                "severity": severity,
                                "message": f"a-{number}",
                            },
                        ],
                    },
                }
            )

    def close(self) -> None:
        self.closed = True


def _write_java_files(root: Path, count: int) -> None:
    for number in range(count):
        (root / f"Generated{number:04d}.java").write_text(
            f"final class Generated{number:04d} {{}}\n",
            encoding="utf-8",
        )


def test_diagnostics_pages_every_java_file_without_a_total_count_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_java_files(tmp_path, 300)
    _FakeJsonRpcProcess.instances.clear()
    monkeypatch.setattr(java_lsp, "_JsonRpcProcess", _FakeJsonRpcProcess)
    service = JavaLanguageService(
        "fake-jdtls",
        diagnostic_page_max_files=64,
        diagnostic_page_max_source_bytes=16 * 1024,
        diagnostic_quiet_seconds=0.001,
    )

    result = service.diagnostics(tmp_path, timeout_seconds=3)

    assert result["schema_version"] == "mmm/java-diagnostics-v2"
    assert result["files_opened"] == 300
    assert result["page_count"] == 5
    assert [page["file_count"] for page in result["pages"]] == [
        64,
        64,
        64,
        64,
        44,
    ]
    assert len(result["diagnostics"]) == 300
    assert result["error_count"] == 300
    assert result["warning_count"] == 300
    assert list(result["diagnostics"]) == sorted(result["diagnostics"])
    first_values = next(iter(result["diagnostics"].values()))
    assert [item["message"] for item in first_values] == ["a-0", "z-0"]
    assert len(_FakeJsonRpcProcess.instances) == 1
    rpc = _FakeJsonRpcProcess.instances[0]
    assert len(rpc.opened_uris) == 300
    assert rpc.request_timeouts == [3]
    assert rpc.closed is False
    service.close()
    assert rpc.closed is True


def test_explicit_file_list_is_deduplicated_and_sorted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_java_files(tmp_path, 3)
    _FakeJsonRpcProcess.instances.clear()
    monkeypatch.setattr(java_lsp, "_JsonRpcProcess", _FakeJsonRpcProcess)
    service = JavaLanguageService(
        "fake-jdtls",
        diagnostic_page_max_files=2,
        diagnostic_quiet_seconds=0.001,
    )

    result = service.diagnostics(
        tmp_path,
        relative_files=[
            "Generated0002.java",
            "Generated0000.java",
            "Generated0002.java",
        ],
        timeout_seconds=3,
    )

    assert result["files_opened"] == 2
    assert result["page_count"] == 1
    assert result["pages"][0]["first_file"] == "Generated0000.java"
    assert result["pages"][0]["last_file"] == "Generated0002.java"


def test_source_byte_budget_creates_more_pages_instead_of_project_rejection(
    tmp_path: Path,
) -> None:
    paths = []
    for name, text in (
        ("A.java", "a" * 4),
        ("B.java", "b" * 5),
        ("C.java", "c" * 6),
    ):
        path = tmp_path / name
        path.write_bytes(text.encode("ascii"))
        paths.append(path)

    pages = java_lsp._diagnostic_pages(
        paths,
        max_files=20,
        max_source_bytes=9,
    )

    assert [[path.name for path in page] for page in pages] == [
        ["A.java", "B.java"],
        ["C.java"],
    ]


def test_one_source_larger_than_the_host_page_budget_fails_explicitly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Oversized.java"
    source.write_bytes(b"x" * 10)

    with pytest.raises(ValueError, match="per-page JDT LS source-byte limit"):
        java_lsp._diagnostic_pages(
            [source],
            max_files=128,
            max_source_bytes=9,
        )


def test_empty_project_does_not_start_a_language_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_process(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("empty diagnostics should not start JDT LS")

    monkeypatch.setattr(java_lsp, "_JsonRpcProcess", _unexpected_process)

    result = JavaLanguageService("fake-jdtls").diagnostics(tmp_path)

    assert result["files_opened"] == 0
    assert result["page_count"] == 0
    assert result["diagnostics"] == {}


def test_each_page_fails_closed_when_opened_uri_never_publishes() -> None:
    class _SilentRpc:
        def __init__(self) -> None:
            self.messages: queue.Queue[dict[str, Any]] = queue.Queue()

    with pytest.raises(JDTLanguageServerError, match="observed=0, expected=1, missing=1"):
        java_lsp._collect_diagnostics(
            _SilentRpc(),  # type: ignore[arg-type]
            expected_uris={"file:///Silent.java"},
            timeout_seconds=0.01,
            quiet_seconds=1.0,
        )


def test_workspace_symbol_calls_reuse_one_initialized_jdt_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeJsonRpcProcess.instances.clear()
    monkeypatch.setattr(java_lsp, "_JsonRpcProcess", _FakeJsonRpcProcess)
    service = JavaLanguageService("fake-jdtls")

    first = service.workspace_symbols(tmp_path, "Alpha", timeout_seconds=3)
    second = service.workspace_symbols(tmp_path, "Beta", timeout_seconds=3)

    assert first["symbols"] == {"query": "Alpha"}
    assert second["symbols"] == {"query": "Beta"}
    assert len(_FakeJsonRpcProcess.instances) == 1
    rpc = _FakeJsonRpcProcess.instances[0]
    assert rpc.request_methods == ["initialize", "workspace/symbol", "workspace/symbol"]
    assert rpc.closed is False
    service.close()
    assert rpc.closed is True


def test_switching_project_root_restarts_the_jdt_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _FakeJsonRpcProcess.instances.clear()
    monkeypatch.setattr(java_lsp, "_JsonRpcProcess", _FakeJsonRpcProcess)
    service = JavaLanguageService("fake-jdtls")

    service.workspace_symbols(first_root, "One", timeout_seconds=3)
    first_rpc = _FakeJsonRpcProcess.instances[0]
    service.workspace_symbols(second_root, "Two", timeout_seconds=3)

    assert len(_FakeJsonRpcProcess.instances) == 2
    assert first_rpc.closed is True
    assert _FakeJsonRpcProcess.instances[1].closed is False
    service.close()
