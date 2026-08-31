from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.java_lsp import JDTLanguageServerError, _collect_diagnostics


def _rpc_with_messages(*messages: dict[str, object]) -> SimpleNamespace:
    notifications: queue.Queue[dict[str, object]] = queue.Queue()
    for message in messages:
        notifications.put(message)
    return SimpleNamespace(messages=notifications)


def test_diagnostic_collection_fails_closed_when_one_opened_uri_never_publishes() -> None:
    rpc = _rpc_with_messages(
        {
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": "file:///workspace/Broken.java",
                "diagnostics": [{"severity": 1, "message": "broken"}],
            },
        }
    )

    started = time.monotonic()
    with pytest.raises(JDTLanguageServerError, match="missing=1"):
        _collect_diagnostics(
            rpc,
            expected_uris={
                "file:///workspace/Broken.java",
                "file:///workspace/Clean.java",
            },
            timeout_seconds=0.05,
            quiet_seconds=0.02,
        )
    assert time.monotonic() - started < 0.5


def test_diagnostic_collection_fails_closed_when_no_opened_uri_publishes() -> None:
    rpc = _rpc_with_messages()

    started = time.monotonic()
    with pytest.raises(JDTLanguageServerError, match="observed=0, expected=2, missing=2"):
        _collect_diagnostics(
            rpc,
            expected_uris={
                "file:///workspace/CleanOne.java",
                "file:///workspace/CleanTwo.java",
            },
            timeout_seconds=0.05,
            quiet_seconds=0.02,
        )
    assert time.monotonic() - started < 0.5
