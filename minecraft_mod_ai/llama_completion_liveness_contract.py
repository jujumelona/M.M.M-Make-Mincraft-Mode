from __future__ import annotations

"""Make llama.cpp completion liveness depend on observable transport progress.

The canonical completion client already aggregates OpenAI-compatible SSE into one
host-validated message. This contract makes long prompt/tool turns observable without
inflating the read timeout: request prompt-progress events and bounded SSE comment pings,
and read decode progress from the current ``/slots`` schema.
"""

from collections.abc import Mapping
from functools import wraps
from typing import Any

_MARKER = "_mmm_progress_aware_completion_transport_v1"


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


def install(stream_module: Any) -> None:
    """Install progress-aware SSE requests and current ``/slots`` decoding."""

    stream_module._slot_progress_from_payload = _slot_progress_from_payload

    client_type = stream_module._StreamingCompletionClient
    current = client_type.post
    if getattr(current, _MARKER, False):
        return

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


__all__ = [
    "_progress_aware_payload",
    "_slot_progress_from_payload",
    "install",
]
