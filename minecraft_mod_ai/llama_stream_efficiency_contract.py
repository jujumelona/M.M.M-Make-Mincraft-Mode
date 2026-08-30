from __future__ import annotations

"""Persistent llama.cpp HTTP/SSE transport with bounded inactivity.

All chat completions, including tool turns, use SSE. Tool-call deltas are transport
fragments only: they are buffered until the server emits ``[DONE]`` and are never
exposed for execution while partial. The completed OpenAI-compatible message is then
returned to the model adapter, where MMM performs the canonical host allowlist/schema
validation before any tool action can run.
"""

import json
import math
import os
import threading
import time
from collections.abc import Mapping
from functools import wraps
from typing import Any

_CLIENT_LOCK = threading.RLock()
_CLIENTS: dict[str, Any] = {}
_CLIENT_LIMIT = 4
_REPORTED_URL_LOCK = threading.RLock()
_REPORTED_SERVER_URLS: set[str] = set()
_DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
_DEFAULT_TOOL_IDLE_TIMEOUT_SECONDS = 120.0
_DEFAULT_TOOL_LIVENESS_HEARTBEAT_SECONDS = 15.0
_DEFAULT_TOOL_STALL_WARNING_SECONDS = 60.0
_DEFAULT_TOOL_PROBE_TIMEOUT_SECONDS = 2.0


class LlamaToolLivenessTimeout(TimeoutError):
    """A streamed native tool response produced no readable transport progress in time."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _stream_idle_timeout_seconds() -> float:
    return _positive_env_float(
        "MMM_LLAMA_STREAM_IDLE_TIMEOUT_SECONDS",
        _DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS,
    )


def _tool_idle_timeout_seconds() -> float:
    return _positive_env_float(
        "MMM_LLAMA_TOOL_IDLE_TIMEOUT_SECONDS",
        _DEFAULT_TOOL_IDLE_TIMEOUT_SECONDS,
    )


def _bounded_timeout(timeout: Any, *, read_seconds: float) -> Any:
    """Preserve stricter caller settings while forbidding an infinite read timeout."""

    import httpx

    timeout_cls = getattr(httpx, "Timeout", None)
    if (isinstance(timeout_cls, type) and isinstance(timeout, timeout_cls)) or hasattr(timeout, "read"):
        caller_read = getattr(timeout, "read", None)
        read = (
            min(float(caller_read), read_seconds)
            if caller_read is not None and float(caller_read) > 0.0
            else read_seconds
        )
        connect = getattr(timeout, "connect", None)
        write = getattr(timeout, "write", None)
        pool = getattr(timeout, "pool", None)
        return httpx.Timeout(
            connect=connect if connect is not None else 30.0,
            read=read,
            write=write if write is not None else 30.0,
            pool=pool if pool is not None else 30.0,
        )
    if isinstance(timeout, (int, float)) and float(timeout) > 0.0:
        read_seconds = min(read_seconds, float(timeout))
    return httpx.Timeout(
        connect=30.0,
        read=read_seconds,
        write=30.0,
        pool=30.0,
    )


def _tool_call_index(value: Any) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("llama server streamed a tool call without a valid index") from exc
    if index < 0:
        raise RuntimeError("llama server streamed a negative tool-call index")
    return index


def _empty_tool_call() -> dict[str, Any]:
    return {
        "id": "",
        "type": "function",
        "function": {"name": "", "arguments": ""},
    }


def _append_tool_call_deltas(message: dict[str, Any], raw_calls: Any) -> int:
    if raw_calls is None:
        return 0
    if not isinstance(raw_calls, list):
        raise TypeError("llama server streamed tool_calls in a non-list shape")
    calls = message.setdefault("tool_calls", [])
    if not isinstance(calls, list):
        raise TypeError("internal streamed tool-call accumulator is invalid")

    progressed = 0
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise TypeError("llama server streamed an invalid tool-call delta")
        index = _tool_call_index(raw_call.get("index", 0))
        while len(calls) <= index:
            calls.append(_empty_tool_call())
        target = calls[index]
        if not isinstance(target, dict):
            raise TypeError("internal streamed tool-call entry is invalid")

        call_id = raw_call.get("id")
        if call_id is not None:
            if not isinstance(call_id, str):
                raise TypeError("llama server streamed a non-string tool-call id")
            if call_id:
                previous_id = target.get("id")
                target["id"] = (previous_id if isinstance(previous_id, str) else "") + call_id
                progressed += len(call_id)

        call_type = raw_call.get("type")
        if call_type is not None:
            if not isinstance(call_type, str):
                raise TypeError("llama server streamed a non-string tool-call type")
            target["type"] = call_type

        raw_function = raw_call.get("function")
        if raw_function is None:
            continue
        if not isinstance(raw_function, Mapping):
            raise TypeError("llama server streamed invalid tool-call function metadata")
        function = target.setdefault("function", {"name": "", "arguments": ""})
        if not isinstance(function, dict):
            raise TypeError("internal streamed tool-call function accumulator is invalid")
        for key in ("name", "arguments"):
            value = raw_function.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(
                    f"llama server streamed non-string tool-call function {key}"
                )
            previous = function.get(key)
            function[key] = (previous if isinstance(previous, str) else "") + value
            progressed += len(value)
    return progressed


def _append_message_delta(message: dict[str, Any], delta: Mapping[str, Any]) -> int:
    progressed = 0
    role = delta.get("role")
    if isinstance(role, str) and role:
        message["role"] = role
    for key in ("reasoning_content", "reasoning", "content"):
        value = delta.get(key)
        if not isinstance(value, str) or not value:
            continue
        previous = message.get(key)
        message[key] = (previous if isinstance(previous, str) else "") + value
        progressed += len(value)
    progressed += _append_tool_call_deltas(message, delta.get("tool_calls"))
    return progressed


def _completion_origin(url: str) -> str:
    value = url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _coerce_progress_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _slot_progress_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, list):
        return None
    processing = [
        slot
        for slot in payload
        if isinstance(slot, Mapping) and slot.get("is_processing") is True
    ]
    decoded_values = [
        value
        for slot in processing
        if (value := _coerce_progress_int(slot.get("n_decoded"))) is not None
    ]
    prompt_values: list[int] = []
    for slot in processing:
        value = _coerce_progress_int(slot.get("n_prompt_tokens_processed"))
        if value is None:
            value = _coerce_progress_int(slot.get("n_prompt_tokens"))
        if value is not None:
            prompt_values.append(value)
    return {
        "processing_slots": len(processing),
        "decoded": sum(decoded_values) if decoded_values else None,
        "prompt_processed": sum(prompt_values) if prompt_values else None,
    }


def _probe_native_tool_progress(client: Any, completion_url: str) -> dict[str, Any]:
    origin = _completion_origin(completion_url)
    timeout = _positive_env_float(
        "MMM_LLAMA_TOOL_PROBE_TIMEOUT_SECONDS",
        _DEFAULT_TOOL_PROBE_TIMEOUT_SECONDS,
    )
    slots_error = ""
    try:
        response = client.get(f"{origin}/slots", timeout=timeout)
        if response.status_code == 200:
            snapshot = _slot_progress_from_payload(response.json())
            if snapshot is not None:
                return {"state": "slots", **snapshot}
    except Exception as exc:  # noqa: BLE001 - optional native observability boundary
        slots_error = type(exc).__name__
    try:
        response = client.get(f"{origin}/health", timeout=timeout)
        if response.status_code == 200:
            result = {"state": "healthy-unobservable"}
            if slots_error:
                result["slots_error"] = slots_error
            return result
        result = {"state": f"health-http-{response.status_code}"}
        if slots_error:
            result["slots_error"] = slots_error
        return result
    except Exception as exc:  # noqa: BLE001 - optional native observability boundary
        result = {"state": "probe-unavailable", "health_error": type(exc).__name__}
        if slots_error:
            result["slots_error"] = slots_error
        return result


def _needs_native_tool_liveness_reporter(payload: Mapping[str, Any]) -> bool:
    """Use slot polling only as a fallback when semantic SSE progress is unavailable."""

    return bool(payload.get("tools")) and payload.get("return_progress") is not True


def _native_tool_liveness_reporter(
    client: Any,
    completion_url: str,
    stop: threading.Event,
    started: float,
) -> None:
    heartbeat = _positive_env_float(
        "MMM_LLAMA_TOOL_LIVENESS_HEARTBEAT_SECONDS",
        _DEFAULT_TOOL_LIVENESS_HEARTBEAT_SECONDS,
    )
    warning_after = min(
        _positive_env_float(
            "MMM_LLAMA_TOOL_STALL_WARNING_SECONDS",
            _DEFAULT_TOOL_STALL_WARNING_SECONDS,
        ),
        _tool_idle_timeout_seconds(),
    )
    last_progress_at = started
    last_decoded: int | None = None
    last_prompt: int | None = None
    warned = False

    while not stop.wait(heartbeat):
        now = time.monotonic()
        snapshot = _probe_native_tool_progress(client, completion_url)
        state = str(snapshot.get("state", "probe-unavailable"))
        if state != "slots":
            print(
                "llama server: tool completion liveness",
                f" state={state}",
                " progress=unobservable",
                f" elapsed={now - started:.1f}s",
                f" read_deadline={_tool_idle_timeout_seconds():.0f}s",
                sep="",
                flush=True,
            )
            continue

        processing = int(snapshot.get("processing_slots", 0) or 0)
        decoded = snapshot.get("decoded")
        prompt = snapshot.get("prompt_processed")
        progressed = bool(
            (isinstance(decoded, int) and (last_decoded is None or decoded > last_decoded))
            or (isinstance(prompt, int) and (last_prompt is None or prompt > last_prompt))
        )
        if progressed:
            last_progress_at = now
            warned = False
        if isinstance(decoded, int):
            last_decoded = decoded
        if isinstance(prompt, int):
            last_prompt = prompt

        idle_for = max(0.0, now - last_progress_at)
        if processing <= 0:
            phase = "no-processing-slot"
        elif isinstance(decoded, int) and decoded > 0:
            phase = "generating" if progressed else "processing-no-new-decode"
        elif isinstance(prompt, int):
            phase = "prompting" if progressed else "processing-no-new-prompt"
        else:
            phase = "processing-progress-counter-unavailable"
        print(
            "llama server: tool completion liveness",
            f" state={phase}",
            f" slots={processing}",
            f" prompt_processed={prompt if isinstance(prompt, int) else 'unknown'}",
            f" decoded={decoded if isinstance(decoded, int) else 'unknown'}",
            f" no_counter_progress={idle_for:.1f}s",
            f" elapsed={now - started:.1f}s",
            f" read_deadline={_tool_idle_timeout_seconds():.0f}s",
            sep="",
            flush=True,
        )
        if processing > 0 and idle_for >= warning_after and not warned:
            print(
                "llama server: WARNING tool completion has no observed slot-counter progress",
                f" for={idle_for:.1f}s",
                " streamed_transport=true",
                flush=True,
            )
            warned = True


class _StreamingCompletionClient:
    """Reuse one HTTP client and aggregate every chat completion through SSE."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def close(self) -> Any:
        return self._client.close()

    def stream(self, method: str, url: str, **kwargs: Any) -> Any:
        stream_kwargs = dict(kwargs)
        stream_kwargs["timeout"] = _bounded_timeout(
            stream_kwargs.get("timeout"),
            read_seconds=_stream_idle_timeout_seconds(),
        )
        return self._client.stream(method, url, **stream_kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        payload = kwargs.get("json")
        if (
            not isinstance(payload, Mapping)
            or not url.rstrip("/").endswith("/chat/completions")
            or payload.get("stream") is True
        ):
            return self._client.post(url, **kwargs)

        has_tools = bool(payload.get("tools"))
        if has_tools and not hasattr(self._client, "stream"):
            # Compatibility for minimal test/dummy clients. Production httpx.Client
            # always provides stream(), so native tool turns use the SSE path below.
            import httpx

            native_kwargs = dict(kwargs)
            deadline = _tool_idle_timeout_seconds()
            native_kwargs["timeout"] = _bounded_timeout(
                native_kwargs.get("timeout"),
                read_seconds=deadline,
            )
            timeout_exc = getattr(httpx, "TimeoutException", None)
            if not (isinstance(timeout_exc, type) and issubclass(timeout_exc, BaseException)):
                timeout_exc = ()
            try:
                return self._client.post(url, **native_kwargs)
            except timeout_exc as exc:
                raise LlamaToolLivenessTimeout(
                    "native llama-server tool completion produced no readable transport "
                    f"progress for {deadline:.0f}s; request aborted"
                ) from exc

        import httpx

        streamed_payload = dict(payload)
        streamed_payload["stream"] = True
        streamed_payload["stream_options"] = {"include_usage": True}
        stream_kwargs = dict(kwargs)
        stream_kwargs["json"] = streamed_payload
        read_seconds = (
            _tool_idle_timeout_seconds() if has_tools else _stream_idle_timeout_seconds()
        )
        stream_kwargs["timeout"] = _bounded_timeout(
            stream_kwargs.get("timeout"),
            read_seconds=read_seconds,
        )
        request = httpx.Request("POST", url)
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        finish_reason: Any = None
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        saw_done = False
        started = time.monotonic()
        stop = threading.Event()
        reporter: threading.Thread | None = None
        if _needs_native_tool_liveness_reporter(streamed_payload):
            reporter = threading.Thread(
                target=_native_tool_liveness_reporter,
                args=(self._client, url, stop, started),
                name="mmm-llama-tool-liveness",
                daemon=True,
            )
            reporter.start()

        timeout_exc = getattr(httpx, "TimeoutException", None)
        if not (isinstance(timeout_exc, type) and issubclass(timeout_exc, BaseException)):
            timeout_exc = ()
        try:
            with self._client.stream("POST", url, **stream_kwargs) as response:
                if response.status_code >= 400:
                    body = response.read()
                    return httpx.Response(
                        response.status_code,
                        headers=response.headers,
                        content=body,
                        request=request,
                    )
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("llama server returned malformed SSE JSON") from exc
                    if not isinstance(chunk, dict):
                        continue
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = chunk_usage
                    chunk_timings = chunk.get("timings")
                    if isinstance(chunk_timings, dict):
                        timings = chunk_timings
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, Mapping):
                        continue
                    current_finish = choice.get("finish_reason")
                    if current_finish is not None:
                        finish_reason = current_finish
                    delta = choice.get("delta")
                    if not isinstance(delta, Mapping):
                        candidate = choice.get("message")
                        delta = candidate if isinstance(candidate, Mapping) else None
                    if delta is not None:
                        _append_message_delta(message, delta)
        except timeout_exc as exc:
            if has_tools:
                raise LlamaToolLivenessTimeout(
                    "native llama-server streamed tool completion produced no readable SSE "
                    f"progress for {read_seconds:.0f}s; request aborted"
                ) from exc
            raise
        finally:
            stop.set()
            if reporter is not None:
                reporter.join(timeout=0.2)

        if not saw_done:
            raise RuntimeError("llama server stream ended before the [DONE] marker")
        result: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
        }
        if usage is not None:
            result["usage"] = usage
        if timings is not None:
            result["timings"] = timings
        return httpx.Response(200, json=result, request=request)


def _client(server_url: str) -> Any:
    import httpx

    origin = server_url.rstrip("/")
    with _CLIENT_LOCK:
        client = _CLIENTS.get(origin)
        if client is not None:
            return client
        timeout = httpx.Timeout(
            connect=30.0,
            read=_stream_idle_timeout_seconds(),
            write=30.0,
            pool=30.0,
        )
        raw_client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=8,
                keepalive_expiry=60.0,
            ),
        )
        client = _StreamingCompletionClient(raw_client)
        _CLIENTS[origin] = client
        while len(_CLIENTS) > _CLIENT_LIMIT:
            key = next(iter(_CLIENTS))
            stale = _CLIENTS.pop(key)
            try:
                stale.close()
            except Exception as exc:  # noqa: BLE001 - best-effort stale-client cleanup
                print(
                    "llama server: stale client close failed",
                    f" error={type(exc).__name__}",
                    flush=True,
                )
        return client


def _report_server_connection(server_url: str) -> None:
    with _REPORTED_URL_LOCK:
        if server_url in _REPORTED_SERVER_URLS:
            return
        _REPORTED_SERVER_URLS.add(server_url)
    print("llama server: connected", server_url, flush=True)


def _native_timing_summary(timings: Any) -> dict[str, float | int] | None:
    if not isinstance(timings, dict):
        return None
    result: dict[str, float | int] = {}
    try:
        predicted_per_second = float(timings.get("predicted_per_second", 0.0) or 0.0)
    except (TypeError, ValueError):
        predicted_per_second = 0.0
    if predicted_per_second > 0.0:
        result["predicted_per_second"] = predicted_per_second
    try:
        draft_n = max(0, int(timings.get("draft_n", 0) or 0))
        accepted = max(0, int(timings.get("draft_n_accepted", 0) or 0))
    except (TypeError, ValueError):
        draft_n = 0
        accepted = 0
    if draft_n > 0:
        accepted = min(accepted, draft_n)
        result["draft_n"] = draft_n
        result["draft_n_accepted"] = accepted
        result["draft_acceptance_pct"] = 100.0 * accepted / draft_n
    return result or None


def _active_decode_profile() -> tuple[str, str, str]:
    spec = os.environ.get("MMM_LLAMA_ACTIVE_SPEC_TYPE", "none").strip() or "none"
    width = os.environ.get("MMM_LLAMA_ACTIVE_DRAFT_N_MAX", "0").strip() or "0"
    p_min = os.environ.get("MMM_LLAMA_ACTIVE_MTP_P_MIN", "0").strip() or "0"
    return spec, width, p_min


def _commit_usage(
    hardware_module: Any,
    usage: dict[str, Any],
    elapsed: float,
) -> dict[str, float] | None:
    try:
        prompt = max(0, int(usage.get("prompt_tokens", 0) or 0))
        output = max(0, int(usage.get("completion_tokens", 0) or 0))
    except (TypeError, ValueError):
        return None
    if prompt == 0 and output == 0:
        return None
    with hardware_module._TELEMETRY_LOCK:
        totals = hardware_module._TELEMETRY_TOTALS
        totals["prompt_tokens"] += prompt
        totals["output_tokens"] += output
        totals["generation_seconds"] += max(0.0, float(elapsed))
        totals["requests"] += 1
        cumulative = dict(totals)
    return {
        "prompt_tokens": prompt,
        "output_tokens": output,
        "generation_seconds": max(0.0, float(elapsed)),
        "cumulative_prompt_tokens": int(cumulative["prompt_tokens"]),
        "cumulative_output_tokens": int(cumulative["output_tokens"]),
        "cumulative_generation_seconds": float(cumulative["generation_seconds"]),
        "cumulative_requests": int(cumulative["requests"]),
    }


def install(hardware_module: Any) -> None:
    """Install the bounded SSE fast path for plain text generation."""

    current = hardware_module._strict_server_generate
    if getattr(current, "_mmm_sse_usage_fast_path", False):
        return

    @wraps(current)
    def fast_stream_generate(adapter: Any, request: Any, server_url: str) -> str:
        if _env_bool("MMM_LLAMA_DETAILED_TELEMETRY", False):
            return current(adapter, request, server_url)

        from .model_adapters import ModelBackendError

        try:
            payload = hardware_module._server_payload(adapter, request)
            if payload.get("tools"):
                return current(adapter, request, server_url)
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            endpoint = f"{server_url.rstrip('/')}/chat/completions"
            client = _client(server_url)
            pieces: list[str] = []
            reasoning_chars = 0
            request_started = time.monotonic()
            first_output_time: float | None = None
            saw_done = False
            final_usage: dict[str, Any] | None = None
            final_timings: dict[str, Any] | None = None

            with client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    body = response.text.strip().replace("\n", " ")
                    if len(body) > 1200:
                        body = body[:1200] + "..."
                    raise RuntimeError(
                        f"llama server returned HTTP {response.status_code}"
                        + (f": {body}" if body else "")
                    )
                _report_server_connection(server_url)
                print(
                    "llama server: request accepted; streaming",
                    f" input_chars={hardware_module._request_content_chars(payload)}",
                    f" max_tokens={payload['max_tokens']}",
                    f" idle_timeout={_stream_idle_timeout_seconds():.0f}s",
                    sep="",
                    flush=True,
                )
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("llama server returned malformed SSE JSON") from exc
                    if not isinstance(chunk, dict):
                        continue
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        final_usage = usage
                    timings = chunk.get("timings")
                    if isinstance(timings, dict):
                        final_timings = timings
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    reasoning, content = hardware_module._stream_delta_parts(choice)
                    if reasoning:
                        reasoning_chars += len(reasoning)
                    if content:
                        pieces.append(content)
                    if (reasoning or content) and first_output_time is None:
                        first_output_time = time.monotonic()

            if not saw_done:
                raise RuntimeError("llama server stream ended before the [DONE] marker")
            content = "".join(pieces).strip()
            if not content:
                if reasoning_chars:
                    raise RuntimeError(
                        "llama server produced reasoning deltas but no visible content"
                    )
                raise RuntimeError("llama server stream produced no text content")

            ended = time.monotonic()
            elapsed = ended - request_started
            generation_elapsed = (
                ended - first_output_time if first_output_time is not None else elapsed
            )
            committed = (
                _commit_usage(hardware_module, final_usage, generation_elapsed)
                if final_usage is not None
                else None
            )
            native = _native_timing_summary(final_timings)
            if committed is not None:
                native_tps = (
                    float(native["predicted_per_second"])
                    if native is not None and "predicted_per_second" in native
                    else float(committed["output_tokens"]) / max(1e-9, generation_elapsed)
                )
                print(
                    "llama server: generation complete",
                    f" prompt_tokens={int(committed['prompt_tokens'])}",
                    f" output_tokens={int(committed['output_tokens'])}",
                    f" tok_s={native_tps:.2f}",
                    f" elapsed={elapsed:.1f}s",
                    sep="",
                    flush=True,
                )
            else:
                print(
                    "llama server: generation complete",
                    f" content_chars={len(content)}",
                    f" elapsed={elapsed:.1f}s",
                    " token_telemetry=unavailable",
                    sep="",
                    flush=True,
                )
            return content
        except Exception as exc:
            if isinstance(exc, ModelBackendError):
                raise
            raise ModelBackendError(
                role=adapter.config.role,
                model_id=adapter.config.model_id,
                cause=exc,
            ) from exc

    fast_stream_generate._mmm_sse_usage_fast_path = True
    hardware_module._strict_server_generate = fast_stream_generate


__all__ = [
    "LlamaToolLivenessTimeout",
    "_active_decode_profile",
    "_append_message_delta",
    "_bounded_timeout",
    "_client",
    "_native_timing_summary",
    "_probe_native_tool_progress",
    "_report_server_connection",
    "_slot_progress_from_payload",
    "_stream_idle_timeout_seconds",
    "_tool_idle_timeout_seconds",
    "install",
]
