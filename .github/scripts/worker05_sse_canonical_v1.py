from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVENESS = ROOT / "minecraft_mod_ai/llama_completion_liveness_contract.py"
STREAM = ROOT / "minecraft_mod_ai/llama_stream_efficiency_contract.py"
BOOTSTRAP = ROOT / "minecraft_mod_ai/runtime_bootstrap.py"
OLD_SHIM = ROOT / "minecraft_mod_ai/llama_sse_error_contract.py"
PROTOCOL = ROOT / "minecraft_mod_ai/llama_sse_protocol.py"
LIVENESS_TEST = ROOT / "tests/test_llama_completion_liveness_contract.py"
OLD_SSE_TEST = ROOT / "tests/test_llama_sse_error_contract.py"
SSE_TEST = ROOT / "tests/test_llama_sse_protocol.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path.relative_to(ROOT)}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


PROTOCOL.write_text(
    '''from __future__ import annotations\n\n"""Pure llama.cpp SSE protocol parsing shared by transport and liveness owners."""\n\nimport json\nfrom collections.abc import Mapping\nfrom typing import Any\n\n\nclass LlamaSseServerError(RuntimeError):\n    """An explicit server-side error delivered inside an SSE stream."""\n\n    def __init__(self, status_code: int, error: Mapping[str, Any]) -> None:\n        self.status_code = max(400, int(status_code))\n        self.error = dict(error)\n        super().__init__(str(self.error.get("message", "llama-server stream error")))\n\n\ndef _error_status(value: Any) -> int:\n    try:\n        status = int(value)\n    except (TypeError, ValueError):\n        return 500\n    return status if 400 <= status <= 599 else 500\n\n\ndef _normalize_error(value: Any) -> dict[str, Any] | None:\n    if isinstance(value, Mapping):\n        error = dict(value)\n        error.setdefault("message", "llama-server stream error")\n        return error\n    if isinstance(value, str) and value.strip():\n        return {"code": 500, "message": value.strip(), "type": "server_error"}\n    return None\n\n\ndef sse_error_from_line(raw_line: Any) -> tuple[int, dict[str, Any]] | None:\n    """Parse current ``data: {error: ...}`` and legacy ``error: ...`` records."""\n\n    if isinstance(raw_line, bytes):\n        line = raw_line.decode("utf-8", errors="replace").strip()\n    else:\n        line = str(raw_line or "").strip()\n    if not line:\n        return None\n\n    if line.startswith("data:"):\n        payload_text = line[5:].strip()\n        legacy = False\n        if not payload_text or payload_text == "[DONE]":\n            return None\n    elif line.startswith("error:"):\n        payload_text = line[6:].strip()\n        legacy = True\n    else:\n        return None\n\n    try:\n        decoded = json.loads(payload_text)\n    except (json.JSONDecodeError, TypeError, ValueError):\n        if legacy and payload_text:\n            return 500, {\n                "code": 500,\n                "message": payload_text,\n                "type": "server_error",\n            }\n        return None\n\n    if legacy:\n        error = _normalize_error(decoded)\n    elif isinstance(decoded, Mapping):\n        error = _normalize_error(decoded.get("error"))\n    else:\n        error = None\n    if error is None:\n        return None\n    status = _error_status(error.get("code"))\n    error["code"] = status\n    return status, error\n\n\n__all__ = ["LlamaSseServerError", "sse_error_from_line"]\n''',
    encoding="utf-8",
)

# Liveness detects explicit server errors before the semantic-stall watchdog can obscure them.
replace_once(
    LIVENESS,
    "from typing import Any\n\n_MARKER",
    "from typing import Any\n\nfrom .llama_sse_protocol import LlamaSseServerError, sse_error_from_line\n\n_MARKER",
)
replace_once(
    LIVENESS,
    '''    def iter_lines(self, *args: Any, **kwargs: Any):\n        watchdog = _SemanticProgressWatchdog(self._idle_seconds)\n        for raw_line in self._response.iter_lines(*args, **kwargs):\n            watchdog.observe(raw_line)\n            yield raw_line\n''',
    '''    def iter_lines(self, *args: Any, **kwargs: Any):\n        watchdog = _SemanticProgressWatchdog(self._idle_seconds)\n        for raw_line in self._response.iter_lines(*args, **kwargs):\n            parsed_error = sse_error_from_line(raw_line)\n            if parsed_error is not None:\n                status, error = parsed_error\n                raise LlamaSseServerError(status, error)\n            watchdog.observe(raw_line)\n            yield raw_line\n''',
)
replace_once(
    LIVENESS,
    "    stream_module._slot_progress_from_payload = _slot_progress_from_payload\n",
    "",
)
# Slot polling is no longer a liveness owner. Remove compatibility-only normalization code.
liveness_text = LIVENESS.read_text(encoding="utf-8")
liveness_text, count = re.subn(
    r"\ndef _slot_progress_from_payload\(payload: Any\) -> dict\[str, Any\] \| None:\n.*?(?=\ndef _ping_interval_seconds)",
    "\n",
    liveness_text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("liveness: slot normalization block not found exactly once")
liveness_text = liveness_text.replace('    "_slot_progress_from_payload",\n', "")
LIVENESS.write_text(liveness_text, encoding="utf-8")

# Stream transport owns SSE error-to-HTTP conversion directly. Remove the obsolete slot
# reporter fallback: semantic SSE progress is now the only completion liveness authority.
replace_once(
    STREAM,
    "from typing import Any\n\n_CLIENT_LOCK",
    "from typing import Any\n\nfrom .llama_sse_protocol import LlamaSseServerError, sse_error_from_line\n\n_CLIENT_LOCK",
)
for line in (
    "_DEFAULT_TOOL_LIVENESS_HEARTBEAT_SECONDS = 15.0\n",
    "_DEFAULT_TOOL_STALL_WARNING_SECONDS = 60.0\n",
    "_DEFAULT_TOOL_PROBE_TIMEOUT_SECONDS = 2.0\n",
):
    replace_once(STREAM, line, "")
stream_text = STREAM.read_text(encoding="utf-8")
stream_text, count = re.subn(
    r"\ndef _completion_origin\(url: str\) -> str:\n.*?(?=\nclass _StreamingCompletionClient:)",
    "\n",
    stream_text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("stream: obsolete native reporter block not found exactly once")
STREAM.write_text(stream_text, encoding="utf-8")
replace_once(
    STREAM,
    '''        saw_done = False\n        started = time.monotonic()\n        stop = threading.Event()\n        reporter: threading.Thread | None = None\n        if _needs_native_tool_liveness_reporter(streamed_payload):\n            reporter = threading.Thread(\n                target=_native_tool_liveness_reporter,\n                args=(self._client, url, stop, started),\n                name="mmm-llama-tool-liveness",\n                daemon=True,\n            )\n            reporter.start()\n\n        timeout_exc = getattr(httpx, "TimeoutException", None)\n''',
    '''        saw_done = False\n\n        timeout_exc = getattr(httpx, "TimeoutException", None)\n''',
)
replace_once(
    STREAM,
    '''                for raw_line in response.iter_lines():\n                    line = raw_line.strip()\n''',
    '''                for raw_line in response.iter_lines():\n                    parsed_error = sse_error_from_line(raw_line)\n                    if parsed_error is not None:\n                        status, error = parsed_error\n                        return httpx.Response(\n                            status,\n                            json={"error": error},\n                            request=request,\n                        )\n                    line = raw_line.strip()\n''',
)
replace_once(
    STREAM,
    '''        except timeout_exc as exc:\n            if has_tools:\n                raise LlamaToolLivenessTimeout(\n                    "native llama-server streamed tool completion produced no readable SSE "\n                    f"progress for {read_seconds:.0f}s; request aborted"\n                ) from exc\n            raise\n        finally:\n            stop.set()\n            if reporter is not None:\n                reporter.join(timeout=0.2)\n''',
    '''        except LlamaSseServerError as exc:\n            return httpx.Response(\n                exc.status_code,\n                json={"error": exc.error},\n                request=request,\n            )\n        except timeout_exc as exc:\n            if has_tools:\n                raise LlamaToolLivenessTimeout(\n                    "native llama-server streamed tool completion produced no readable SSE "\n                    f"progress for {read_seconds:.0f}s; request aborted"\n                ) from exc\n            raise\n''',
)
# The plain streaming fast path also recognizes explicit SSE errors without relying on
# bootstrap monkeypatch order.
needle = '''                for raw_line in response.iter_lines():\n                    line = raw_line.strip()\n'''
replacement = '''                for raw_line in response.iter_lines():\n                    parsed_error = sse_error_from_line(raw_line)\n                    if parsed_error is not None:\n                        status, error = parsed_error\n                        raise LlamaSseServerError(status, error)\n                    line = raw_line.strip()\n'''
stream_text = STREAM.read_text(encoding="utf-8")
# The aggregator occurrence was already replaced; exactly one fast-path occurrence remains.
if stream_text.count(needle) != 1:
    raise SystemExit(f"stream: expected one remaining fast-path loop, found {stream_text.count(needle)}")
STREAM.write_text(stream_text.replace(needle, replacement, 1), encoding="utf-8")

# Bootstrap no longer composes a runtime SSE monkeypatch.
replace_once(BOOTSTRAP, "        llama_completion_liveness_contract,\n", "")
replace_once(BOOTSTRAP, "    from .llama_sse_error_contract import install as install_sse_errors\n", "")
replace_once(
    BOOTSTRAP,
    '''    install_sse_errors(\n        llama_completion_liveness_contract,\n        llama_stream_efficiency_contract,\n    )\n''',
    "",
)

if not OLD_SHIM.exists():
    raise SystemExit("obsolete SSE monkeypatch shim is unexpectedly absent")
OLD_SHIM.unlink()

# Rewrite liveness tests around the canonical semantic owner; delete slot-polling expectations.
LIVENESS_TEST.write_text(
    '''from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nimport pytest\n\nfrom minecraft_mod_ai import llama_completion_liveness_contract as contract\nfrom minecraft_mod_ai.llama_sse_protocol import LlamaSseServerError\nfrom minecraft_mod_ai.model_adapters import llama_cpp_adapter\n\n\ndef test_progress_payload_requests_prompt_events_and_bounded_ping() -> None:\n    stream_module = SimpleNamespace(\n        _tool_idle_timeout_seconds=lambda: 120.0,\n        _stream_idle_timeout_seconds=lambda: 120.0,\n    )\n    original = {"model": "local", "messages": [], "tools": [{"type": "function"}]}\n\n    result = contract._progress_aware_payload(stream_module, original)\n\n    assert result is not original\n    assert "return_progress" not in original\n    assert result["return_progress"] is True\n    assert result["sse_ping_interval"] == 30\n\n\ndef test_semantic_progress_ignores_transport_ping_and_tracks_prompt_progress() -> None:\n    progressed, processed = contract._semantic_progress_from_sse_line(\n        ": ping", last_prompt_processed=None\n    )\n    assert progressed is False\n    assert processed is None\n\n    progressed, processed = contract._semantic_progress_from_sse_line(\n        'data: {"prompt_progress":{"processed":64}}',\n        last_prompt_processed=None,\n    )\n    assert progressed is True\n    assert processed == 64\n\n\ndef test_progress_response_raises_server_error_before_watchdog() -> None:\n    response = SimpleNamespace(\n        iter_lines=lambda: iter(\n            ['data: {"error":{"code":400,"message":"context overflow"}}']\n        )\n    )\n    wrapped = contract._ProgressCheckedResponse(response, 0.001)\n\n    with pytest.raises(LlamaSseServerError, match="context overflow"):\n        list(wrapped.iter_lines())\n\n\ndef test_install_wraps_nonstream_chat_completion_without_changing_timeout() -> None:\n    calls: list[tuple[str, dict]] = []\n\n    class FakeClient:\n        def post(self, url: str, **kwargs):\n            calls.append((url, kwargs))\n            return "ok"\n\n    stream_module = SimpleNamespace(\n        _StreamingCompletionClient=FakeClient,\n        _tool_idle_timeout_seconds=lambda: 12.0,\n        _stream_idle_timeout_seconds=lambda: 120.0,\n    )\n\n    contract.install(stream_module)\n    client = FakeClient()\n    timeout = object()\n    result = client.post(\n        "http://127.0.0.1:8080/v1/chat/completions",\n        json={"messages": [], "tools": [{"type": "function"}]},\n        timeout=timeout,\n    )\n\n    assert result == "ok"\n    assert calls[0][1]["timeout"] is timeout\n    assert calls[0][1]["json"]["return_progress"] is True\n    assert calls[0][1]["json"]["sse_ping_interval"] == 4\n\n\ndef test_liveness_install_has_no_reporter_or_slot_polling_dependency() -> None:\n    class FakeClient:\n        def __init__(self, _client=None):\n            self._client = _client\n\n        def post(self, _url: str, **_kwargs):\n            return "ok"\n\n        def stream(self, method: str, url: str, **kwargs):\n            return method, url, kwargs\n\n    stream_module = SimpleNamespace(\n        _StreamingCompletionClient=FakeClient,\n        _CLIENTS={},\n        _tool_idle_timeout_seconds=lambda: 12.0,\n        _stream_idle_timeout_seconds=lambda: 120.0,\n    )\n\n    contract.install(stream_module)\n\n    assert not hasattr(stream_module, "_native_tool_liveness_reporter")\n    assert not hasattr(stream_module, "_probe_native_tool_progress")\n\n\ndef test_runtime_completion_transport_has_one_progress_aware_owner() -> None:\n    assert getattr(\n        llama_cpp_adapter._post_completion,\n        "_mmm_single_progress_aware_completion_owner_v1",\n        False,\n    )\n''',
    encoding="utf-8",
)

SSE_TEST.write_text(
    '''from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nfrom minecraft_mod_ai import llama_stream_efficiency_contract as stream\nfrom minecraft_mod_ai.llama_sse_protocol import sse_error_from_line\n\n\ndef test_current_llama_sse_error_shape_is_normalized() -> None:\n    parsed = sse_error_from_line(\n        'data: {"error":{"code":400,"message":"context overflow","type":"invalid_request_error"}}'\n    )\n    assert parsed is not None\n    status, error = parsed\n    assert status == 400\n    assert error["code"] == 400\n    assert error["message"] == "context overflow"\n\n\ndef test_legacy_llama_sse_error_shape_is_preserved() -> None:\n    parsed = sse_error_from_line('error: {"code":503,"message":"server busy"}')\n    assert parsed == (503, {"code": 503, "message": "server busy"})\n\n\ndef test_legacy_text_error_is_normalized() -> None:\n    parsed = sse_error_from_line("error: temporary backend failure")\n    assert parsed == (\n        500,\n        {"code": 500, "message": "temporary backend failure", "type": "server_error"},\n    )\n\n\ndef test_normal_content_is_not_misclassified_as_error() -> None:\n    assert (\n        sse_error_from_line('data: {"choices":[{"delta":{"content":"error: harmless"}}]}')\n        is None\n    )\n\n\ndef test_stream_aggregator_returns_http_error_without_runtime_monkeypatch() -> None:\n    class Response:\n        status_code = 200\n        headers = {}\n\n        def __enter__(self):\n            return self\n\n        def __exit__(self, *_args):\n            return False\n\n        def iter_lines(self):\n            return iter(\n                ['data: {"error":{"code":400,"message":"context overflow"}}']\n            )\n\n    class Client:\n        def stream(self, *_args, **_kwargs):\n            return Response()\n\n        def post(self, *_args, **_kwargs):\n            raise AssertionError("non-stream fallback must not be used")\n\n    client = stream._StreamingCompletionClient(Client())\n    response = client.post(\n        "http://127.0.0.1:8080/v1/chat/completions",\n        json={"messages": [], "max_tokens": 16},\n    )\n\n    assert response.status_code == 400\n    assert response.json()["error"]["message"] == "context overflow"\n\n\ndef test_stream_module_has_no_native_slot_reporter_fallback() -> None:\n    assert not hasattr(stream, "_native_tool_liveness_reporter")\n    assert not hasattr(stream, "_probe_native_tool_progress")\n    assert not hasattr(stream, "_needs_native_tool_liveness_reporter")\n''',
    encoding="utf-8",
)
if OLD_SSE_TEST.exists():
    OLD_SSE_TEST.unlink()

# No stale runtime shim references may survive.
for base in (ROOT / "minecraft_mod_ai", ROOT / "tests"):
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "llama_sse_error_contract" in text:
            raise SystemExit(f"stale SSE shim reference: {path.relative_to(ROOT)}")

(ROOT / ".github/workflows/worker05-sse-canonical-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
