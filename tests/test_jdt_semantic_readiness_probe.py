from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import java_lsp


class _FakeRpc:
    def __init__(self, *, include_probe: bool = True) -> None:
        self.notifications: list[tuple[str, dict]] = []
        self.requests: list[tuple[str, dict, float]] = []
        self.include_probe = include_probe

    def notify(self, method: str, params: dict) -> None:
        self.notifications.append((method, params))

    def request(self, method: str, params: dict, timeout: float):
        self.requests.append((method, params, timeout))
        assert method == "textDocument/documentSymbol"
        uri = params["textDocument"]["uri"]
        name = Path(uri.removeprefix("file://")).stem
        return [{"name": name, "kind": 5}] if self.include_probe else []


def test_jdt_readiness_probe_is_materialized_and_removed(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)
    rpc = _FakeRpc()

    java_lsp._await_java_core_ready(
        rpc,
        tmp_path,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    assert not list(source_root.glob("__MmmJdtReadinessProbe_*.java"))
    assert [method for method, _params in rpc.notifications] == [
        "textDocument/didOpen",
        "textDocument/didClose",
    ]
    assert rpc.requests
    assert rpc.requests[0][0] == "textDocument/documentSymbol"


def test_document_symbol_readiness_accepts_nested_document_symbols() -> None:
    assert java_lsp._document_symbols_contain_name(
        [
            {
                "name": "Outer",
                "children": [
                    {"name": "__MmmJdtReadinessProbe_1_2"},
                ],
            }
        ],
        "__MmmJdtReadinessProbe_1_2",
    )


def test_document_symbol_readiness_rejects_missing_probe() -> None:
    assert (
        java_lsp._document_symbols_contain_name(
            [{"name": "DifferentType"}],
            "__MmmJdtReadinessProbe_1_2",
        )
        is False
    )
