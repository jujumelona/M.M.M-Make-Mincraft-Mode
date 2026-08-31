from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    "import os\nimport threading\nimport time\n",
    "import os\n",
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    "values = tuple(sorted(set(int(value) for value in parallel_values)))",
    "values = tuple(sorted({int(value) for value in parallel_values}))",
)
replace_once(
    "minecraft_mod_ai/qwen35_mtp_hotpath_contract.py",
    '''        if value in {"--port", "-p"} and index + 1 < len(args):\n            if args[index + 1] == target:\n                return True\n''',
    '''        if (\n            value in {"--port", "-p"}\n            and index + 1 < len(args)\n            and args[index + 1] == target\n        ):\n            return True\n''',
)
replace_once(
    "tests/test_llama_semantic_progress_watchdog.py",
    '''from minecraft_mod_ai.llama_completion_liveness_contract import (\n    LlamaSemanticProgressTimeout,\n    _SemanticProgressWatchdog,\n    _semantic_progress_from_sse_line,\n)\n''',
    '''from minecraft_mod_ai.llama_completion_liveness_contract import (\n    LlamaSemanticProgressTimeout,\n    _semantic_progress_from_sse_line,\n    _SemanticProgressWatchdog,\n)\n''',
)
replace_once(
    "tests/test_llama_semantic_progress_watchdog.py",
    '''    assert getattr(\n        stream_contract._native_tool_liveness_reporter,\n        "_mmm_no_slot_poll_completion_liveness_v1",\n        False,\n    )\n''',
    '''    assert not hasattr(stream_contract, "_native_tool_liveness_reporter")\n    assert not hasattr(stream_contract, "_probe_native_tool_progress")\n    assert not hasattr(stream_contract, "_slot_progress_from_payload")\n''',
)

# The old completion-liveness tests asserted the deleted /slots polling implementation.
# Keep the transport timeout tests and turn the removed API into an explicit regression:
# liveness must be driven only by semantic SSE progress and must never reintroduce /slots.
(ROOT / "tests/test_llama_tool_completion_liveness.py").write_text(
    '''from __future__ import annotations\n\nfrom typing import Any\n\nimport httpx\n\nfrom minecraft_mod_ai import llama_stream_efficiency_contract as contract\n\n\nclass _ImmediateRawClient:\n    def __init__(self) -> None:\n        self.timeout: httpx.Timeout | None = None\n        self.response = object()\n\n    def post(self, _url: str, **kwargs: Any) -> object:\n        timeout = kwargs.get("timeout")\n        assert isinstance(timeout, httpx.Timeout)\n        self.timeout = timeout\n        return self.response\n\n\ndef _timeout(read: float) -> httpx.Timeout:\n    return httpx.Timeout(connect=30.0, read=read, write=30.0, pool=30.0)\n\n\ndef test_native_tool_turn_enforces_bounded_idle_read_timeout(monkeypatch) -> None:\n    raw = _ImmediateRawClient()\n    client = contract._StreamingCompletionClient(raw)\n    monkeypatch.delenv("MMM_LLAMA_TOOL_IDLE_TIMEOUT_SECONDS", raising=False)\n\n    response = client.post(\n        "http://127.0.0.1:8080/v1/chat/completions",\n        json={"tools": [{"type": "function"}], "max_tokens": 4096},\n        timeout=_timeout(600.0),\n    )\n\n    assert response is raw.response\n    assert raw.timeout is not None\n    assert raw.timeout.read == 120.0\n\n\ndef test_explicit_tool_completion_timeout_is_honored(monkeypatch) -> None:\n    raw = _ImmediateRawClient()\n    client = contract._StreamingCompletionClient(raw)\n    monkeypatch.setenv("MMM_LLAMA_TOOL_IDLE_TIMEOUT_SECONDS", "75")\n\n    client.post(\n        "http://127.0.0.1:8080/v1/chat/completions",\n        json={"tools": [{"type": "function"}], "max_tokens": 4096},\n        timeout=_timeout(600.0),\n    )\n\n    assert raw.timeout is not None\n    assert raw.timeout.read == 75.0\n\n\ndef test_slot_poll_liveness_api_stays_removed() -> None:\n    assert not hasattr(contract, "_native_tool_liveness_reporter")\n    assert not hasattr(contract, "_probe_native_tool_progress")\n    assert not hasattr(contract, "_slot_progress_from_payload")\n''',
    encoding="utf-8",
)

Path(__file__).unlink(missing_ok=True)
