from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_exact(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


STREAM = "minecraft_mod_ai/llama_stream_efficiency_contract.py"

replace_exact(
    STREAM,
    '''def _native_tool_liveness_reporter(\n''',
    '''def _needs_native_tool_liveness_reporter(payload: Mapping[str, Any]) -> bool:\n    """Use slot polling only as a fallback when semantic SSE progress is unavailable."""\n\n    return bool(payload.get("tools")) and payload.get("return_progress") is not True\n\n\ndef _native_tool_liveness_reporter(\n''',
)
replace_exact(
    STREAM,
    '''        if has_tools:\n            reporter = threading.Thread(\n''',
    '''        if _needs_native_tool_liveness_reporter(streamed_payload):\n            reporter = threading.Thread(\n''',
)

# Streamed protocol shape errors are type violations, not runtime-state failures. Keeping
# that distinction explicit improves diagnostics and removes the old lint debt in this owner.
for old, new in (
    (
        'raise RuntimeError("llama server streamed tool_calls in a non-list shape")',
        'raise TypeError("llama server streamed tool_calls in a non-list shape")',
    ),
    (
        'raise RuntimeError("internal streamed tool-call accumulator is invalid")',
        'raise TypeError("internal streamed tool-call accumulator is invalid")',
    ),
    (
        'raise RuntimeError("llama server streamed an invalid tool-call delta")',
        'raise TypeError("llama server streamed an invalid tool-call delta")',
    ),
    (
        'raise RuntimeError("internal streamed tool-call entry is invalid")',
        'raise TypeError("internal streamed tool-call entry is invalid")',
    ),
    (
        'raise RuntimeError("llama server streamed a non-string tool-call id")',
        'raise TypeError("llama server streamed a non-string tool-call id")',
    ),
    (
        'raise RuntimeError("llama server streamed a non-string tool-call type")',
        'raise TypeError("llama server streamed a non-string tool-call type")',
    ),
    (
        'raise RuntimeError("llama server streamed invalid tool-call function metadata")',
        'raise TypeError("llama server streamed invalid tool-call function metadata")',
    ),
    (
        'raise RuntimeError("internal streamed tool-call function accumulator is invalid")',
        'raise TypeError("internal streamed tool-call function accumulator is invalid")',
    ),
    (
        '''raise RuntimeError(\n                    f"llama server streamed non-string tool-call function {key}"\n                )''',
        '''raise TypeError(\n                    f"llama server streamed non-string tool-call function {key}"\n                )''',
    ),
):
    replace_exact(STREAM, old, new)

# Native slot/health probes are optional observability fallbacks. Preserve the health
# fallback after a /slots failure while retaining the failure class for diagnostics instead
# of silently swallowing it. Client eviction close failures are likewise rare but observable.
replace_exact(
    STREAM,
    '''    try:\n        response = client.get(f"{origin}/slots", timeout=timeout)\n        if response.status_code == 200:\n            snapshot = _slot_progress_from_payload(response.json())\n            if snapshot is not None:\n                return {"state": "slots", **snapshot}\n    except Exception:\n        pass\n    try:\n        response = client.get(f"{origin}/health", timeout=timeout)\n        if response.status_code == 200:\n            return {"state": "healthy-unobservable"}\n        return {"state": f"health-http-{response.status_code}"}\n    except Exception:\n        return {"state": "probe-unavailable"}\n''',
    '''    slots_error = ""\n    try:\n        response = client.get(f"{origin}/slots", timeout=timeout)\n        if response.status_code == 200:\n            snapshot = _slot_progress_from_payload(response.json())\n            if snapshot is not None:\n                return {"state": "slots", **snapshot}\n    except Exception as exc:  # noqa: BLE001 - optional native observability boundary\n        slots_error = type(exc).__name__\n    try:\n        response = client.get(f"{origin}/health", timeout=timeout)\n        if response.status_code == 200:\n            result = {"state": "healthy-unobservable"}\n            if slots_error:\n                result["slots_error"] = slots_error\n            return result\n        result = {"state": f"health-http-{response.status_code}"}\n        if slots_error:\n            result["slots_error"] = slots_error\n        return result\n    except Exception as exc:  # noqa: BLE001 - optional native observability boundary\n        result = {"state": "probe-unavailable", "health_error": type(exc).__name__}\n        if slots_error:\n            result["slots_error"] = slots_error\n        return result\n''',
)
replace_exact(
    STREAM,
    '''            try:\n                stale.close()\n            except Exception:\n                pass\n''',
    '''            try:\n                stale.close()\n            except Exception as exc:  # noqa: BLE001 - best-effort stale-client cleanup\n                print(\n                    "llama server: stale client close failed",\n                    f" error={type(exc).__name__}",\n                    flush=True,\n                )\n''',
)

replace_exact(
    "minecraft_mod_ai/llama_completion_liveness_contract.py",
    '_REPORTER_MARKER = "_mmm_no_slot_poll_completion_liveness_v1"\n',
    "",
)
old_reporter = '''def _install_no_slot_poll_reporter(stream_module: Any) -> None:\n    current = stream_module._native_tool_liveness_reporter\n    if getattr(current, _REPORTER_MARKER, False):\n        return\n\n    @wraps(current)\n    def sse_owned_reporter(*_args: Any, **_kwargs: Any) -> None:\n        return None\n\n    setattr(sse_owned_reporter, _REPORTER_MARKER, True)\n    sse_owned_reporter.__wrapped__ = current  # type: ignore[attr-defined]\n    stream_module._native_tool_liveness_reporter = sse_owned_reporter\n\n\n'''
replace_exact(
    "minecraft_mod_ai/llama_completion_liveness_contract.py",
    old_reporter,
    "",
)
replace_exact(
    "minecraft_mod_ai/llama_completion_liveness_contract.py",
    "    _install_no_slot_poll_reporter(stream_module)\n",
    "",
)

# Extend the existing liveness contract tests so the fallback policy is executable and the
# liveness installer proves it no longer depends on a reporter symbol being present.
test_path = ROOT / "tests/test_llama_completion_liveness_contract.py"
test_text = test_path.read_text(encoding="utf-8")
anchor = '''def test_runtime_completion_transport_has_one_progress_aware_owner() -> None:\n'''
addition = '''def test_semantic_progress_disables_native_slot_reporter() -> None:\n    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract\n\n    assert (\n        stream_contract._needs_native_tool_liveness_reporter(\n            {"tools": [{"type": "function"}], "return_progress": True}\n        )\n        is False\n    )\n\n\ndef test_native_slot_reporter_remains_fallback_without_semantic_progress() -> None:\n    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract\n\n    assert (\n        stream_contract._needs_native_tool_liveness_reporter(\n            {"tools": [{"type": "function"}]}\n        )\n        is True\n    )\n    assert stream_contract._needs_native_tool_liveness_reporter({"messages": []}) is False\n\n\ndef test_liveness_install_has_no_reporter_monkeypatch_dependency() -> None:\n    calls: list[tuple[str, dict]] = []\n\n    class FakeClient:\n        def __init__(self, _client=None):\n            self._client = _client\n\n        def post(self, url: str, **kwargs):\n            calls.append((url, kwargs))\n            return "ok"\n\n        def stream(self, method: str, url: str, **kwargs):\n            return (method, url, kwargs)\n\n    stream_module = SimpleNamespace(\n        _StreamingCompletionClient=FakeClient,\n        _CLIENTS={},\n        _tool_idle_timeout_seconds=lambda: 12.0,\n        _stream_idle_timeout_seconds=lambda: 120.0,\n    )\n\n    contract.install(stream_module)\n\n    assert not hasattr(stream_module, "_native_tool_liveness_reporter")\n    assert stream_module._slot_progress_from_payload is contract._slot_progress_from_payload\n\n\n'''
if test_text.count(anchor) != 1:
    raise SystemExit(
        "tests/test_llama_completion_liveness_contract.py: runtime-owner anchor missing/ambiguous"
    )
test_path.write_text(test_text.replace(anchor, addition + anchor, 1), encoding="utf-8")

# One-shot staging must disappear from the production commit.
(ROOT / ".github/workflows/worker05-liveness-reporter-canonical-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
