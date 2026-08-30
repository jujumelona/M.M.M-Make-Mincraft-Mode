from __future__ import annotations

"""Preserve llama.cpp SSE server errors across the streaming aggregation boundary.

Recent llama.cpp emits OpenAI-compatible ``data: {\"error\": ...}`` records while older
builds emitted ``error: {...}``. The generic completion aggregator historically ignored
both shapes and could turn an explicit context overflow into an empty successful HTTP
response. This contract converts either stream error into the same HTTP-like response
shape consumed by the canonical finish-reason/context-pressure classifier.
"""

import json
from collections.abc import Mapping
from functools import wraps
from typing import Any

_ITER_MARKER = "_mmm_sse_server_error_detection_v1"
_POST_MARKER = "_mmm_sse_server_error_response_v1"


class LlamaSseServerError(RuntimeError):
    def __init__(self, status_code: int, error: Mapping[str, Any]) -> None:
        self.status_code = max(400, int(status_code))
        self.error = dict(error)
        super().__init__(str(self.error.get("message", "llama-server stream error")))


def _error_status(value: Any) -> int:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return 500
    return status if 400 <= status <= 599 else 500


def _normalize_error(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        error = dict(value)
        if "message" not in error:
            error["message"] = "llama-server stream error"
        return error
    if isinstance(value, str) and value.strip():
        return {"code": 500, "message": value.strip(), "type": "server_error"}
    return None


def _sse_error_from_line(raw_line: Any) -> tuple[int, dict[str, Any]] | None:
    if isinstance(raw_line, bytes):
        line = raw_line.decode("utf-8", errors="replace").strip()
    else:
        line = str(raw_line or "").strip()
    if not line:
        return None

    payload_text = ""
    legacy = False
    if line.startswith("data:"):
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            return None
    elif line.startswith("error:"):
        payload_text = line[6:].strip()
        legacy = True
    else:
        return None

    try:
        decoded = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        if legacy and payload_text:
            error = {"code": 500, "message": payload_text, "type": "server_error"}
            return 500, error
        return None

    if legacy:
        error = _normalize_error(decoded)
    elif isinstance(decoded, Mapping):
        error = _normalize_error(decoded.get("error"))
    else:
        error = None
    if error is None:
        return None
    status = _error_status(error.get("code"))
    error["code"] = status
    return status, error


def install(liveness_module: Any, stream_module: Any) -> None:
    response_type = liveness_module._ProgressCheckedResponse
    current_iter = response_type.iter_lines
    if not getattr(current_iter, _ITER_MARKER, False):

        @wraps(current_iter)
        def error_checked_iter_lines(self: Any, *args: Any, **kwargs: Any):
            watchdog = liveness_module._SemanticProgressWatchdog(self._idle_seconds)
            for raw_line in self._response.iter_lines(*args, **kwargs):
                parsed = _sse_error_from_line(raw_line)
                if parsed is not None:
                    status, error = parsed
                    raise LlamaSseServerError(status, error)
                watchdog.observe(raw_line)
                yield raw_line

        setattr(error_checked_iter_lines, _ITER_MARKER, True)
        error_checked_iter_lines.__wrapped__ = current_iter  # type: ignore[attr-defined]
        response_type.iter_lines = error_checked_iter_lines

    client_type = stream_module._StreamingCompletionClient
    current_post = client_type.post
    if getattr(current_post, _POST_MARKER, False):
        return

    @wraps(current_post)
    def error_aware_post(self: Any, url: str, **kwargs: Any) -> Any:
        try:
            return current_post(self, url, **kwargs)
        except LlamaSseServerError as exc:
            import httpx

            request = httpx.Request("POST", url)
            return httpx.Response(
                exc.status_code,
                json={"error": exc.error},
                request=request,
            )

    setattr(error_aware_post, _POST_MARKER, True)
    error_aware_post.__wrapped__ = current_post  # type: ignore[attr-defined]
    client_type.post = error_aware_post


__all__ = ["LlamaSseServerError", "_sse_error_from_line", "install"]
