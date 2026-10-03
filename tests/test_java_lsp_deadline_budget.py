from __future__ import annotations

import pytest

from minecraft_mod_ai import java_lsp


class _FakeRpc:
    def __init__(self) -> None:
        self.stderr: list[str] = []
        self.notifications: list[tuple[str, dict[str, object]]] = []
        self.requests: list[tuple[str, dict[str, object], float]] = []

    def notify(self, method: str, params: dict[str, object]) -> None:
        self.notifications.append((method, params))

    def request(self, method: str, params: dict[str, object], timeout: float):
        self.requests.append((method, params, timeout))
        return []


def test_diagnostics_gets_fresh_budget_after_cold_start_and_shares_it_across_pages(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)
    (source_root / "A.java").write_text("final class A {}\n", encoding="utf-8")
    (source_root / "B.java").write_text("final class B {}\n", encoding="utf-8")

    service = java_lsp.JavaLanguageService(
        diagnostic_page_max_files=1,
        diagnostic_quiet_seconds=0.0,
    )
    rpc = _FakeRpc()
    now = [100.0]
    seen: list[tuple[str, float, float | None]] = []

    monkeypatch.setattr(java_lsp, "assert_server_safe_source_sets", lambda _root: None)
    monkeypatch.setattr(java_lsp.time, "monotonic", lambda: now[0])

    def fake_ensure(root, *, timeout_seconds, deadline=None):
        del root, timeout_seconds
        assert deadline is not None
        seen.append(("ensure", deadline, None))
        now[0] = 180.0
        return rpc

    def fake_collect(
        _rpc,
        *,
        expected_uris,
        timeout_seconds,
        quiet_seconds,
        deadline=None,
    ):
        del _rpc, quiet_seconds
        assert deadline is not None
        seen.append(("collect", deadline, timeout_seconds))
        return {uri: [] for uri in expected_uris}

    monkeypatch.setattr(service, "_ensure_rpc_locked", fake_ensure)
    monkeypatch.setattr(java_lsp, "_collect_diagnostics", fake_collect)

    result = service.diagnostics(tmp_path, timeout_seconds=90)

    assert result["page_count"] == 2
    assert [phase for phase, _deadline, _timeout in seen] == [
        "ensure",
        "collect",
        "collect",
    ]
    assert seen[0][1] == pytest.approx(190.0)
    assert seen[1][1] == pytest.approx(270.0)
    assert seen[2][1] == pytest.approx(270.0)
    assert seen[1][1] - seen[0][1] == pytest.approx(80.0)
    assert seen[1][2] == pytest.approx(90.0)
    assert seen[2][2] == pytest.approx(90.0)


def test_workspace_symbols_gets_fresh_operation_budget_after_cold_start(
    tmp_path, monkeypatch
):
    service = java_lsp.JavaLanguageService()
    rpc = _FakeRpc()
    now = [100.0]
    startup_deadlines: list[float] = []

    monkeypatch.setattr(java_lsp.time, "monotonic", lambda: now[0])

    def fake_ensure(root, *, timeout_seconds, deadline=None):
        del root, timeout_seconds
        assert deadline is not None
        startup_deadlines.append(deadline)
        now[0] = 180.0
        return rpc

    monkeypatch.setattr(service, "_ensure_rpc_locked", fake_ensure)

    result = service.workspace_symbols(tmp_path, "Example", timeout_seconds=90)

    assert result["symbols"] == []
    assert len(startup_deadlines) == 1
    assert startup_deadlines[0] == pytest.approx(190.0)
    assert len(rpc.requests) == 1
    method, params, request_timeout = rpc.requests[0]
    assert method == "workspace/symbol"
    assert params == {"query": "Example"}
    assert request_timeout == pytest.approx(90.0)


def test_expired_absolute_deadline_fails_closed(monkeypatch):
    monkeypatch.setattr(java_lsp.time, "monotonic", lambda: 10.0)
    with pytest.raises(java_lsp.JDTLanguageServerError, match="deadline exceeded"):
        java_lsp._remaining_jdt_deadline(9.0, operation="diagnostics")
