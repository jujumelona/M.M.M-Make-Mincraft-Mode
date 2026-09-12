from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace

from minecraft_mod_ai.managed_llama_reuse_contract import _install_fast_path


class _AliveProcess:
    @staticmethod
    def poll():
        return None


def _fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_exact_managed_runtime_reuse_skips_wrapped_lifecycle(monkeypatch) -> None:
    calls = []
    config = object()
    selection = {"runtime": "exact"}
    url = "http://127.0.0.1:8910/v1"

    def slow_lifecycle(_config, _request):
        calls.append("called")
        raise AssertionError("canonical lifecycle must not run for exact live reuse")

    autotune = SimpleNamespace(
        _AUTOTUNE_LOCK=threading.RLock(),
        _MANAGED_PROCESS=_AliveProcess(),
        _MANAGED_URL=url,
        _MMM_LLAMA_RUNTIME_RECEIPT={
            "selection_inputs_sha256": _fingerprint(selection),
        },
        ensure_tuned_server=slow_lifecycle,
    )
    runtime_tuning = SimpleNamespace(
        _selection_inputs=lambda value: selection if value is config else {},
        _json_fingerprint=_fingerprint,
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", url)

    _install_fast_path(autotune, runtime_tuning)

    assert autotune.ensure_tuned_server(config, object()) == url
    assert calls == []
    assert getattr(autotune.ensure_tuned_server, "_mmm_managed_llama_exact_reuse_v1", False)


def test_stale_managed_runtime_receipt_falls_through_once(monkeypatch) -> None:
    calls = []
    url = "http://127.0.0.1:8910/v1"

    def lifecycle(config, request):
        calls.append((config, request))
        return "replacement"

    autotune = SimpleNamespace(
        _AUTOTUNE_LOCK=threading.RLock(),
        _MANAGED_PROCESS=_AliveProcess(),
        _MANAGED_URL=url,
        _MMM_LLAMA_RUNTIME_RECEIPT={
            "selection_inputs_sha256": _fingerprint({"runtime": "old"}),
        },
        ensure_tuned_server=lifecycle,
    )
    runtime_tuning = SimpleNamespace(
        _selection_inputs=lambda _config: {"runtime": "new"},
        _json_fingerprint=_fingerprint,
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", url)
    config = object()
    request = object()

    _install_fast_path(autotune, runtime_tuning)

    assert autotune.ensure_tuned_server(config, request) == "replacement"
    assert calls == [(config, request)]
