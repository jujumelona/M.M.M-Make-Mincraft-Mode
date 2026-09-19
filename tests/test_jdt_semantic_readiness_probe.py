from __future__ import annotations

import queue

import pytest

from minecraft_mod_ai import java_lsp


class _FakeRpc:
    def __init__(self, messages: list[dict] | None = None) -> None:
        self.messages: queue.Queue[dict] = queue.Queue()
        for message in messages or []:
            self.messages.put(message)
        self.configuration = {"java": {"autobuild": {"enabled": True}}}
        self.workspace_folders = []
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)


def _status(status_type: str, message: str = "") -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "language/status",
        "params": {"type": status_type, "message": message},
    }


def test_jdt_readiness_waits_for_service_ready_and_preserves_other_messages(
    tmp_path,
) -> None:
    started = _status("Started", "Ready")
    unrelated = {
        "jsonrpc": "2.0",
        "method": "window/logMessage",
        "params": {"type": 3, "message": "workspace initialized"},
    }
    rpc = _FakeRpc([started, unrelated, _status("ServiceReady", "ServiceReady")])

    java_lsp._await_java_core_ready(
        rpc,
        tmp_path,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    preserved = [rpc.messages.get_nowait(), rpc.messages.get_nowait()]
    assert preserved == [started, unrelated]


def test_jdt_service_ready_requires_exact_lifecycle_status() -> None:
    assert java_lsp._jdt_service_ready(
        _status("ServiceReady", "ServiceReady")
    )
    assert not java_lsp._jdt_service_ready(_status("Started", "Ready"))
    assert not java_lsp._jdt_service_ready(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"diagnostics": []},
        }
    )


def test_jdt_readiness_handles_server_requests_before_service_ready(tmp_path) -> None:
    rpc = _FakeRpc(
        [
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "workspace/configuration",
                "params": {"items": [{"section": "java"}]},
            },
            _status("ServiceReady", "ServiceReady"),
        ]
    )

    java_lsp._await_java_core_ready(
        rpc,
        tmp_path,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    assert rpc.sent == [
        {
            "jsonrpc": "2.0",
            "id": 7,
            "result": [{"autobuild": {"enabled": True}}],
        }
    ]


def test_jdt_readiness_fails_without_service_ready(tmp_path) -> None:
    started = _status("Started", "Ready")
    rpc = _FakeRpc([started])

    with pytest.raises(
        java_lsp.JDTWorkspaceBootstrapError,
        match="ServiceReady",
    ):
        java_lsp._await_java_core_ready(
            rpc,
            tmp_path,
            timeout_seconds=0.01,
            quiet_seconds=0.0,
        )

    assert rpc.messages.get_nowait() == started
