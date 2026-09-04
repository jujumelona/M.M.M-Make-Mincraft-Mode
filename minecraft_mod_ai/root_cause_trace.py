from __future__ import annotations

"""Bounded, secret-safe structured tracing for host-owned execution boundaries.

The trace is deliberately independent of model output.  It records what the host
actually attempted, what gate/result was observed, and the original exception chain
before callers wrap or aggregate the failure.
"""

import inspect
import itertools
import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

_TRACE_PREFIX = "ROOT CAUSE TRACE: "
_TRACE_SEQUENCE = itertools.count(1)
_STRING_LIMIT = 512
_COLLECTION_LIMIT = 64
_DEPTH_LIMIT = 5
_SECRET_KEY_PARTS = (
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
)
_FAILURE_STATUSES = frozenset({"FAIL", "FAILED", "ERROR", "INVALID", "UNAVAILABLE", "TIMEOUT", "TIMED_OUT", "UNHEALTHY"})
_SKIP_STATUSES = frozenset({"SKIP", "SKIPPED", "NOT_RUN"})
F = TypeVar("F", bound=Callable[..., Any])


def _secret_key(value: Any) -> bool:
    key = str(value or "").casefold().replace("-", "_")
    return any(part in key for part in _SECRET_KEY_PARTS)


def bounded_safe(value: Any, *, depth: int = 0, key: str = "") -> Any:
    """Return deterministic bounded trace data while redacting credential-like fields."""
    if key and _secret_key(key):
        return "<redacted>"
    if depth >= _DEPTH_LIMIT:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _STRING_LIMIT else value[:_STRING_LIMIT] + "…"
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (raw_key, child) in enumerate(value.items()):
            if index >= _COLLECTION_LIMIT:
                result["<truncated>"] = max(0, len(value) - _COLLECTION_LIMIT)
                break
            child_key = str(raw_key)
            result[child_key] = bounded_safe(child, depth=depth + 1, key=child_key)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [bounded_safe(item, depth=depth + 1) for item in items[:_COLLECTION_LIMIT]]
        if len(items) > _COLLECTION_LIMIT:
            result.append(f"<truncated:{len(items) - _COLLECTION_LIMIT}>")
        return result
    return bounded_safe(str(value), depth=depth + 1)


def exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """Preserve the first causal exception instead of only the final wrapper message."""
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 16:
        seen.add(id(current))
        frames = traceback.extract_tb(current.__traceback__)[-20:] if current.__traceback__ else []
        chain.append(
            {
                "type": type(current).__name__,
                "message": bounded_safe(str(current)),
                "frames": [
                    {
                        "file": frame.filename,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in frames
                ],
            }
        )
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return chain


def _semantic_outcome(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "PASS"
    status = str(value.get("status") or value.get("state") or value.get("outcome") or "").strip().upper()
    if status in _FAILURE_STATUSES:
        return "FAIL"
    if status in _SKIP_STATUSES:
        return "SKIP"
    if value.get("ok") is False or value.get("success") is False:
        return "FAIL"
    return "PASS"


def emit_root_cause(
    event: str,
    *,
    stage: str = "",
    operation: str = "",
    gate: str = "",
    result: str = "",
    reason: str = "",
    details: Mapping[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "mmm/root-cause-trace-v1",
        "trace_seq": next(_TRACE_SEQUENCE),
        "event": str(event),
    }
    if stage:
        payload["stage"] = stage
    if operation:
        payload["operation"] = operation
    if gate:
        payload["gate"] = gate
    if result:
        payload["result"] = result
    if reason:
        payload["reason"] = bounded_safe(reason)
    if details:
        payload["details"] = bounded_safe(details)
    if exc is not None:
        payload["exception_chain"] = exception_chain(exc)
    print(
        _TRACE_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
        flush=True,
    )


def traced_callable(function: F, *, stage: str, operation: str | None = None) -> F:
    """Trace every invocation at a shared host boundary without changing its contract."""
    operation_name = operation or function.__name__
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        try:
            bound = signature.bind_partial(*args, **kwargs)
            safe_arguments = {
                name: bounded_safe(value, key=name)
                for name, value in bound.arguments.items()
            }
        except Exception as bind_exc:
            safe_arguments = {"argument_binding": f"{type(bind_exc).__name__}: {bind_exc}"}
        emit_root_cause(
            "operation_start",
            stage=stage,
            operation=operation_name,
            gate="host_boundary",
            result="START",
            details={"arguments": safe_arguments},
        )
        try:
            value = function(*args, **kwargs)
        except BaseException as exc:
            emit_root_cause(
                "operation_failure",
                stage=stage,
                operation=operation_name,
                gate="host_boundary",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={"elapsed_ms": round((time.monotonic() - started) * 1000.0, 3)},
                exc=exc,
            )
            raise
        outcome = _semantic_outcome(value)
        emit_root_cause(
            "operation_result",
            stage=stage,
            operation=operation_name,
            gate="host_boundary",
            result=outcome,
            reason="host operation returned",
            details={
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "result_summary": bounded_safe(value),
            },
        )
        return value

    return cast(F, wrapped)


__all__ = ["bounded_safe", "emit_root_cause", "exception_chain", "traced_callable"]
