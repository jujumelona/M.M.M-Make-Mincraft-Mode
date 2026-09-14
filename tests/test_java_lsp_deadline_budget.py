from __future__ import annotations

import pytest

from minecraft_mod_ai import java_lsp


class _FakeRpc:
    def __init__(self) -> None:
        self.stderr: list[str] = []
        self.notifications: list[tuple[str, dict[str, object]]] = []

    def notify(self, method: str, params: dict[str, object]) -> None:
        self.notifications.append((method, params))


def test_diagnostics_reuses_one_absolute_deadline_across_pages(tmp_path, monkeypatch):
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)
    (source_root / "A.java").write_text("final class A {}\n", encoding="utf-8")
    (source_root / "B.java").write_text("final class B {}\n", encoding="utf-8")

    service = java_lsp.JavaLanguageService(
        diagnostic_page_max_files=1,
        diagnostic_quiet_seconds=0.0,
    )
    rpc = _FakeRpc()
    seen: list[tuple[str, float]] = []

    monkeypatch.setattr(java_lsp, "assert_server_safe_source_sets", lambda _root: None)

    def fake_ensure(root, *, timeout_seconds, deadline=None):
        del root, timeout_seconds
        assert deadline is not None
        seen.append(("ensure", deadline))
        return rpc

    def fake_collect(
        _rpc,
        *,
        expected_uris,
        timeout_seconds,
        quiet_seconds,
        deadline=None,
    ):
        del _rpc, timeout_seconds, quiet_seconds
        assert deadline is not None
        seen.append(("collect", deadline))
        return {uri: [] for uri in expected_uris}

    monkeypatch.setattr(service, "_ensure_rpc_locked", fake_ensure)
    monkeypatch.setattr(java_lsp, "_collect_diagnostics", fake_collect)

    result = service.diagnostics(tmp_path, timeout_seconds=90)

    assert result["page_count"] == 2
    assert [phase for phase, _deadline in seen] == ["ensure", "collect", "collect"]
    assert len({deadline for _phase, deadline in seen}) == 1


def test_expired_absolute_deadline_fails_closed(monkeypatch):
    monkeypatch.setattr(java_lsp.time, "monotonic", lambda: 10.0)
    with pytest.raises(java_lsp.JDTLanguageServerError, match="deadline exceeded"):
        java_lsp._remaining_jdt_deadline(9.0, operation="diagnostics")
