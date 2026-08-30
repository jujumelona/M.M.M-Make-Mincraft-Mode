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


replace_exact(
    "minecraft_mod_ai/llama_stream_efficiency_contract.py",
    '''def _native_tool_liveness_reporter(\n''',
    '''def _needs_native_tool_liveness_reporter(payload: Mapping[str, Any]) -> bool:\n    """Use slot polling only as a fallback when semantic SSE progress is unavailable."""\n\n    return bool(payload.get("tools")) and payload.get("return_progress") is not True\n\n\ndef _native_tool_liveness_reporter(\n''',
)
replace_exact(
    "minecraft_mod_ai/llama_stream_efficiency_contract.py",
    '''        if has_tools:\n            reporter = threading.Thread(\n''',
    '''        if _needs_native_tool_liveness_reporter(streamed_payload):\n            reporter = threading.Thread(\n''',
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
