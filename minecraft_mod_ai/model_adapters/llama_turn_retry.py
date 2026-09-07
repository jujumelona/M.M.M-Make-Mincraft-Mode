from __future__ import annotations

"""Bounded transactional retry for one native llama.cpp semantic turn.

The SSE transport buffers assistant text and tool-call fragments until a complete
``[DONE]`` frame is observed. This layer retries only a failed semantic model turn,
so a transient transport failure cannot force the planner/tool pipeline to replay and
no partial assistant/tool action escapes the failed attempt.
"""

import os
import time
from functools import wraps
from typing import Any, Callable

import httpx

from .base import ModelBackendError

_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
_MAX_RETRY_ATTEMPTS = 5


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0.0 else default


def _retry_attempts() -> int:
    return _bounded_env_int(
        "MMM_LLAMA_TURN_RETRY_ATTEMPTS",
        _DEFAULT_RETRY_ATTEMPTS,
        minimum=1,
        maximum=_MAX_RETRY_ATTEMPTS,
    )


def _retry_backoff_seconds() -> float:
    return _positive_env_float(
        "MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS",
        _DEFAULT_RETRY_BACKOFF_SECONDS,
    )


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, ModelBackendError) and isinstance(current.cause, BaseException):
            current = current.cause
            continue
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_retriable_turn_failure(exc: BaseException) -> bool:
    """Retry transport/liveness failures, never semantic/schema/request failures."""

    for item in _exception_chain(exc):
        if isinstance(item, (httpx.TransportError, TimeoutError)):
            return True
        message = str(item).strip().lower()
        if "stream ended before the [done] marker" in message:
            return True
        if "llama server returned http 5" in message:
            return True
    return False


def _discarded_partial_hint(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        message = str(item).lower()
        if "stream" in message or "sse" in message or "transport" in message:
            return True
    return False


def _install_marker(owner: type[Any]) -> str:
    return f"{owner.__module__}.{owner.__qualname__}"


def install_llama_turn_retry(adapter_type: type[Any]) -> None:
    """Install one idempotent retry owner on ``adapter_type.generate_turn``."""

    current = adapter_type.generate_turn
    if getattr(current, "_mmm_transactional_turn_retry", False):
        return

    original: Callable[..., Any] = current

    @wraps(original)
    def generate_turn_with_retry(self: Any, request: Any) -> Any:
        attempts = _retry_attempts()
        backoff = _retry_backoff_seconds()
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                return original(self, request)
            except Exception as exc:
                retriable = _is_retriable_turn_failure(exc)
                will_retry = retriable and attempt < attempts
                elapsed = time.monotonic() - started
                print(
                    "llama server: turn failure",
                    f" attempt={attempt}/{attempts}",
                    f" error={type(exc).__name__}",
                    f" elapsed={elapsed:.2f}s",
                    f" partial_discarded={str(_discarded_partial_hint(exc)).lower()}",
                    f" retry={str(will_retry).lower()}",
                    sep="",
                    flush=True,
                )
                if not will_retry:
                    raise
                if backoff:
                    time.sleep(backoff * attempt)
        raise AssertionError("unreachable llama turn retry state")

    generate_turn_with_retry._mmm_transactional_turn_retry = True  # type: ignore[attr-defined]
    generate_turn_with_retry._mmm_retry_owner = _install_marker(adapter_type)  # type: ignore[attr-defined]
    adapter_type.generate_turn = generate_turn_with_retry


__all__ = [
    "_is_retriable_turn_failure",
    "install_llama_turn_retry",
]
