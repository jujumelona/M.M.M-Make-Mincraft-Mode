from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import java_lsp


class _FakeRpc:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, dict]] = []

    def notify(self, method: str, params: dict) -> None:
        self.notifications.append((method, params))


def test_jdt_readiness_probe_is_materialized_and_removed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)
    rpc = _FakeRpc()
    observed_probe: Path | None = None

    def fake_collect(
        _rpc,
        *,
        expected_uris,
        timeout_seconds,
        quiet_seconds,
        deadline=None,
    ):
        nonlocal observed_probe
        assert timeout_seconds > 0
        assert quiet_seconds >= 0
        probe = next(source_root.glob("__MmmJdtReadinessProbe_*.java"))
        observed_probe = probe
        assert probe.is_file()
        assert expected_uris == {probe.resolve().as_uri()}
        return {probe.resolve().as_uri(): []}

    monkeypatch.setattr(java_lsp, "_collect_diagnostics", fake_collect)

    java_lsp._await_java_core_ready(
        rpc,
        tmp_path,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    assert observed_probe is not None
    assert not observed_probe.exists()
    assert [method for method, _params in rpc.notifications] == [
        "textDocument/didOpen",
        "textDocument/didClose",
    ]


def test_jdt_readiness_ignores_non_core_probe_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)
    rpc = _FakeRpc()

    def fake_collect(
        _rpc,
        *,
        expected_uris,
        timeout_seconds,
        quiet_seconds,
        deadline=None,
    ):
        uri = next(iter(expected_uris))
        return {
            uri: [
                {
                    "severity": 1,
                    "message": "Synthetic unrelated project diagnostic",
                }
            ]
        }

    monkeypatch.setattr(java_lsp, "_collect_diagnostics", fake_collect)

    java_lsp._await_java_core_ready(
        rpc,
        tmp_path,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    assert not list(source_root.glob("__MmmJdtReadinessProbe_*.java"))
