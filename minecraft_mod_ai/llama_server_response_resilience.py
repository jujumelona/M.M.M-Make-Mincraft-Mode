from __future__ import annotations

"""One bounded retry for transient native llama-server completion failures.

The retry wraps inference only. No model turn has been returned to ModelRouter yet, so
no tool or filesystem action can have executed between attempts. Permanent request,
context, schema and tool-protocol failures are deliberately not retried.
"""

import os
import re
import sys
import time
from functools import wraps
from typing import Any, Mapping

import httpx

_MARKER = "_mmm_transient_llama_response_retry_v1"
_HTTP_STATUS = re.compile(r"llama server returned HTTP\s+(\d{3})", re.IGNORECASE)
_ALWAYS_TRANSIENT_STATUS = frozenset({429, 502, 503, 504})
_PERMANENT_500_MARKERS = (
    "context length",
    "context window",
    "context size",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "request too large",
    "invalid request",
    "invalid argument",
    "invalid json",
    "json schema",
    "tool schema",
    "tool_choice",
)
_TRANSIENT_500_MARKERS = (
    "busy",
    "no available slot",
    "slot unavailable",
    "failed to acquire slot",
    "temporarily unavailable",
    "resource temporarily unavailable",
    "loading model",
    "model is loading",
)
_TRANSIENT_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


def _retry_delay_seconds() -> float:
    raw = os.environ.get("MMM_LLAMA_TRANSIENT_RETRY_DELAY_SECONDS", "").strip()
    if not raw:
        return 0.5
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "MMM_LLAMA_TRANSIENT_RETRY_DELAY_SECONDS must be a non-negative number"
        ) from exc
    return max(0.0, min(10.0, value))


def _http_status(exc: BaseException) -> int | None:
    match = _HTTP_STATUS.search(str(exc))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _transient_completion_failure(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_TRANSPORT_ERRORS):
        return True

    status = _http_status(exc)
    if status in _ALWAYS_TRANSIENT_STATUS:
        return True
    if status != 500:
        return False

    text = str(exc).casefold()
    if any(marker in text for marker in _PERMANENT_500_MARKERS):
        return False
    if any(marker in text for marker in _TRANSIENT_500_MARKERS):
        return True

    # llama.cpp uses HTTP 500 for some server-side runtime failures without a stable
    # machine-readable error code. One inference-only retry is safe; a second failure
    # is returned unchanged so a permanent unknown error cannot loop indefinitely.
    return True


def install(llama_cpp_module: Any) -> None:
    current = llama_cpp_module._completion_message
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def completion_with_transient_retry(
        server_url: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            return current(server_url, payload)
        except Exception as exc:
            if not _transient_completion_failure(exc):
                raise
            delay = _retry_delay_seconds()
            print(
                "llama server recovery: retrying transient completion once",
                f" reason={type(exc).__name__}: {str(exc)[:320]}",
                f" delay={delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            if delay:
                time.sleep(delay)
            return current(server_url, payload)

    setattr(completion_with_transient_retry, _MARKER, True)
    llama_cpp_module._completion_message = completion_with_transient_retry


__all__ = ["_transient_completion_failure", "install"]
