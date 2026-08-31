from __future__ import annotations

"""Make llama.cpp completion liveness depend on semantic model progress.

The canonical completion client aggregates OpenAI-compatible SSE into one host-validated
message. Long prompt/tool turns request prompt-progress events and bounded SSE comment
pings, but transport pings are deliberately *not* model progress: a healthy TCP/server
connection must not keep a stalled decode alive forever. A host watchdog therefore
resets only on prompt processing, visible/reasoning/token/tool deltas, or terminal
completion events. Current ``/slots`` decoding remains available for compatibility, but
active completion liveness no longer polls ``/slots`` on the decode hot path.
"""

import json
import time
from collections.abc import Callable, Mapping
from functools import wraps
from types import TracebackType
from typing import Any

from .llama_sse_protocol import LlamaSseServerError, sse_error_from_line

_MARKER = "_mmm_progress_aware_completion_transport_v1"
_STREAM_MARKER = "_mmm_progress_aware_completion_stream_v1"
_ADAPTER_MARKER = "_mmm_single_progress_aware_completion_owner_v1"
_CLIENT_INIT_MARKER = "_mmm_semantic_progress_client_v1"


class LlamaSemanticProgressTimeout(TimeoutError):
    """A live SSE connection produced no prompt/decode/tool progress in time."""


def _coerce_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)



def _ping_interval_seconds(stream_module: Any, payload: Mapping[str, Any]) -> int:
    if payload.get("tools"):
        idle = float(stream_module._tool_idle_timeout_seconds())
    else:
        idle = float(stream_module._stream_idle_timeout_seconds())
    return max(1, min(30, int(idle / 3.0) or 1))


def _progress_aware_payload(
    stream_module: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    result["return_progress"] = True
    result["sse_ping_interval"] = _ping_interval_seconds(stream_module, result)
    return result


def _tool_call_delta_has_progress(raw_calls: Any) -> bool:
    if not isinstance(raw_calls, list):
        return False
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        call_id = raw_call.get("id")
        if isinstance(call_id, str) and call_id:
            return True
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            continue
        for key in ("name", "arguments"):
            value = function.get(key)
            if isinstance(value, str) and value:
                return True
    return False


def _message_delta_has_progress(delta: Any) -> bool:
    if not isinstance(delta, Mapping):
        return False
    for key in ("content", "reasoning_content", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return True
    return _tool_call_delta_has_progress(delta.get("tool_calls"))


def _semantic_progress_from_sse_line(
    raw_line: Any,
    *,
    last_prompt_processed: int | None,
) -> tuple[bool, int | None]:
    """Return whether one SSE line proves model progress, excluding keepalive pings."""

    if isinstance(raw_line, bytes):
        line = raw_line.decode("utf-8", errors="replace").strip()
    else:
        line = str(raw_line or "").strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return False, last_prompt_processed
    data = line[5:].strip()
    if not data:
        return False, last_prompt_processed
    if data == "[DONE]":
        return True, last_prompt_processed
    try:
        chunk = json.loads(data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, last_prompt_processed
    if not isinstance(chunk, Mapping):
        return False, last_prompt_processed

    progressed = False
    prompt_progress = chunk.get("prompt_progress")
    if isinstance(prompt_progress, Mapping):
        processed = _coerce_nonnegative_int(prompt_progress.get("processed"))
        if processed is None:
            processed = _coerce_nonnegative_int(prompt_progress.get("n_processed"))
        if processed is not None and (
            last_prompt_processed is None or processed > last_prompt_processed
        ):
            last_prompt_processed = processed
            progressed = True

    choices = chunk.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            if choice.get("finish_reason") is not None:
                progressed = True
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                candidate = choice.get("message")
                delta = candidate if isinstance(candidate, Mapping) else None
            if _message_delta_has_progress(delta):
                progressed = True
    return progressed, last_prompt_processed


class _SemanticProgressWatchdog:
    """Track semantic inactivity independently from HTTP/SSE transport activity."""

    def __init__(
        self,
        idle_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle_seconds = max(0.001, float(idle_seconds))
        self._clock = clock
        self._last_progress_at = clock()
        self._last_prompt_processed: int | None = None

    def observe(self, raw_line: Any) -> None:
        now = self._clock()
        progressed, prompt_processed = _semantic_progress_from_sse_line(
            raw_line,
            last_prompt_processed=self._last_prompt_processed,
        )
        self._last_prompt_processed = prompt_processed
        if progressed:
            self._last_progress_at = now
            return
        idle_for = max(0.0, now - self._last_progress_at)
        if idle_for >= self.idle_seconds:
            raise LlamaSemanticProgressTimeout(
                "native llama-server SSE connection remained alive but produced no semantic "
                "prompt/decode/tool progress for "
                f"{self.idle_seconds:.0f}s; request aborted"
            )


class _ProgressCheckedResponse:
    def __init__(self, response: Any, idle_seconds: float) -> None:
        self._response = response
        self._idle_seconds = idle_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def iter_lines(self, *args: Any, **kwargs: Any):
        watchdog = _SemanticProgressWatchdog(self._idle_seconds)
        for raw_line in self._response.iter_lines(*args, **kwargs):
            parsed_error = sse_error_from_line(raw_line)
            if parsed_error is not None:
                status, error = parsed_error
                raise LlamaSseServerError(status, error)
            watchdog.observe(raw_line)
            yield raw_line


class _ProgressCheckedStream:
    def __init__(self, stream: Any, idle_seconds: float) -> None:
        self._stream = stream
        self._idle_seconds = idle_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self) -> Any:
        response = self._stream.__enter__()
        return _ProgressCheckedResponse(response, self._idle_seconds)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any:
        return self._stream.__exit__(exc_type, exc, tb)


class _SemanticProgressClient:
    """Proxy an HTTP client and enforce semantic inactivity on progress-aware SSE."""

    _mmm_semantic_progress_client_v1 = True

    def __init__(self, client: Any, stream_module: Any) -> None:
        self._client = client
        self._stream_module = stream_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def close(self) -> Any:
        return self._client.close()

    def stream(self, method: str, url: str, **kwargs: Any) -> Any:
        stream = self._client.stream(method, url, **kwargs)
        payload = kwargs.get("json")
        if (
            not isinstance(payload, Mapping)
            or not url.rstrip("/").endswith("/chat/completions")
            or payload.get("return_progress") is not True
        ):
            return stream
        idle_seconds = (
            float(self._stream_module._tool_idle_timeout_seconds())
            if payload.get("tools")
            else float(self._stream_module._stream_idle_timeout_seconds())
        )
        return _ProgressCheckedStream(stream, idle_seconds)


def _wrap_raw_client(client: Any, stream_module: Any) -> Any:
    if getattr(client, "_mmm_semantic_progress_client_v1", False):
        return client
    if not callable(getattr(client, "stream", None)):
        return client
    return _SemanticProgressClient(client, stream_module)


def _install_raw_client_watchdog(stream_module: Any) -> None:
    client_type = stream_module._StreamingCompletionClient
    current_init = client_type.__init__
    if not getattr(current_init, _CLIENT_INIT_MARKER, False):

        @wraps(current_init)
        def progress_checked_init(self: Any, *args: Any, **kwargs: Any) -> None:
            if args:
                updated_args = list(args)
                updated_args[0] = _wrap_raw_client(updated_args[0], stream_module)
                current_init(self, *updated_args, **kwargs)
                return
            if "client" in kwargs:
                updated_kwargs = dict(kwargs)
                updated_kwargs["client"] = _wrap_raw_client(
                    updated_kwargs["client"], stream_module
                )
                current_init(self, **updated_kwargs)
                return
            current_init(self, *args, **kwargs)

        setattr(progress_checked_init, _CLIENT_INIT_MARKER, True)
        progress_checked_init.__wrapped__ = current_init  # type: ignore[attr-defined]
        client_type.__init__ = progress_checked_init

    for client in tuple(getattr(stream_module, "_CLIENTS", {}).values()):
        raw = getattr(client, "_client", None)
        if raw is not None:
            client._client = _wrap_raw_client(raw, stream_module)


def _install_stream_progress_payload(stream_module: Any) -> None:
    client_type = stream_module._StreamingCompletionClient
    current = getattr(client_type, "stream", None)
    if not callable(current) or getattr(current, _STREAM_MARKER, False):
        return

    @wraps(current)
    def progress_aware_stream(self: Any, method: str, url: str, **kwargs: Any) -> Any:
        payload = kwargs.get("json")
        if (
            isinstance(payload, Mapping)
            and url.rstrip("/").endswith("/chat/completions")
            and payload.get("stream") is True
        ):
            updated = dict(kwargs)
            updated["json"] = _progress_aware_payload(stream_module, payload)
            kwargs = updated
        return current(self, method, url, **kwargs)

    setattr(progress_aware_stream, _STREAM_MARKER, True)
    progress_aware_stream.__wrapped__ = current  # type: ignore[attr-defined]
    client_type.stream = progress_aware_stream


def _install_adapter_completion_transport(stream_module: Any, adapter_module: Any) -> None:
    """Replace the blind heartbeat wrapper with the canonical SSE liveness owner."""

    current = adapter_module._post_completion
    if getattr(current, _ADAPTER_MARKER, False):
        return

    @wraps(current)
    def progress_owned_completion(server_url: str, payload: Mapping[str, Any]) -> Any:
        endpoint = f"{server_url}/chat/completions"
        read_timeout = adapter_module._positive_env_float(
            "MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS",
            adapter_module._DEFAULT_COMPLETION_TIMEOUT_SECONDS,
        )
        input_chars = adapter_module._payload_content_chars(payload)
        max_tokens = payload.get("max_tokens", "?")
        tool_count = len(payload.get("tools", ()) or ())
        print(
            "llama server: completion request",
            f" input_chars={input_chars}",
            f" max_tokens={max_tokens}",
            f" tools={tool_count}",
            f" read_timeout={read_timeout:.0f}s",
            " transport=sse-semantic-progress",
            sep="",
            flush=True,
        )
        timeout = adapter_module.httpx.Timeout(
            connect=30.0,
            read=read_timeout,
            write=30.0,
            pool=30.0,
        )
        try:
            if adapter_module.httpx.post is not adapter_module._DEFAULT_HTTPX_POST:
                return adapter_module.httpx.post(endpoint, json=payload, timeout=timeout)
            return stream_module._client(server_url).post(
                endpoint,
                json=payload,
                timeout=timeout,
            )
        except adapter_module.httpx.TimeoutException as exc:
            raise RuntimeError(
                "native llama-server completion made no readable transport progress for "
                f"{read_timeout:.0f}s"
            ) from exc

    setattr(progress_owned_completion, _ADAPTER_MARKER, True)
    progress_owned_completion.__wrapped__ = current  # type: ignore[attr-defined]
    adapter_module._post_completion = progress_owned_completion


def install(stream_module: Any, adapter_module: Any | None = None) -> None:
    """Install one SSE semantic-progress owner for plain and tool completions."""

    _install_raw_client_watchdog(stream_module)
    _install_stream_progress_payload(stream_module)

    client_type = stream_module._StreamingCompletionClient
    current = client_type.post
    if not getattr(current, _MARKER, False):

        @wraps(current)
        def progress_aware_post(self: Any, url: str, **kwargs: Any) -> Any:
            payload = kwargs.get("json")
            if (
                isinstance(payload, Mapping)
                and url.rstrip("/").endswith("/chat/completions")
                and payload.get("stream") is not True
            ):
                updated = dict(kwargs)
                updated["json"] = _progress_aware_payload(stream_module, payload)
                kwargs = updated
            return current(self, url, **kwargs)

        setattr(progress_aware_post, _MARKER, True)
        progress_aware_post.__wrapped__ = current  # type: ignore[attr-defined]
        client_type.post = progress_aware_post

    if adapter_module is not None:
        _install_adapter_completion_transport(stream_module, adapter_module)


__all__ = [
    "LlamaSemanticProgressTimeout",
    "_SemanticProgressWatchdog",
    "_progress_aware_payload",
    "_semantic_progress_from_sse_line",
    "install",
]
