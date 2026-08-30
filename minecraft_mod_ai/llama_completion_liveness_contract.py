from __future__ import annotations

"""Make llama.cpp completion liveness depend on observable transport progress.

The canonical completion client already aggregates OpenAI-compatible SSE into one
host-validated message. This contract makes long prompt/tool turns observable without
inflating the read timeout: request prompt-progress events and bounded SSE comment pings,
read decode progress from the current ``/slots`` schema, and remove the older generic
wall-clock heartbeat that could not distinguish work from a stall.
"""

from collections.abc import Mapping
from functools import wraps
from typing import Any

_MARKER = "_mmm_progress_aware_completion_transport_v1"
_ADAPTER_MARKER = "_mmm_single_progress_aware_completion_owner_v1"


def _coerce_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _slot_progress_from_payload(payload: Any) -> dict[str, Any] | None:
    """Normalize current and older llama.cpp slot progress shapes."""

    if not isinstance(payload, list):
        return None
    processing = [
        slot
        for slot in payload
        if isinstance(slot, Mapping) and slot.get("is_processing") is True
    ]
    decoded_values: list[int] = []
    prompt_values: list[int] = []
    for slot in processing:
        next_token = slot.get("next_token")
        decoded = (
            _coerce_nonnegative_int(next_token.get("n_decoded"))
            if isinstance(next_token, Mapping)
            else None
        )
        if decoded is None:
            decoded = _coerce_nonnegative_int(slot.get("n_decoded"))
        if decoded is not None:
            decoded_values.append(decoded)

        prompt = _coerce_nonnegative_int(slot.get("n_prompt_tokens_processed"))
        if prompt is None:
            prompt = _coerce_nonnegative_int(slot.get("n_prompt_tokens"))
        if prompt is not None:
            prompt_values.append(prompt)

    return {
        "processing_slots": len(processing),
        "decoded": sum(decoded_values) if decoded_values else None,
        "prompt_processed": sum(prompt_values) if prompt_values else None,
    }


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
            " transport=sse-progress",
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
                "native llama-server completion made no readable progress for "
                f"{read_timeout:.0f}s"
            ) from exc

    setattr(progress_owned_completion, _ADAPTER_MARKER, True)
    progress_owned_completion.__wrapped__ = current  # type: ignore[attr-defined]
    adapter_module._post_completion = progress_owned_completion


def install(stream_module: Any, adapter_module: Any | None = None) -> None:
    """Install progress-aware SSE requests and current ``/slots`` decoding."""

    stream_module._slot_progress_from_payload = _slot_progress_from_payload

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
    "_progress_aware_payload",
    "_slot_progress_from_payload",
    "install",
]
