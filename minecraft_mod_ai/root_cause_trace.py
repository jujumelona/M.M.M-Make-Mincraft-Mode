from __future__ import annotations

"""Bounded, secret-safe structured tracing for host-owned execution boundaries.

The trace is deliberately independent of model output. It records what the host
actually attempted, what gate/result was observed, and the original exception chain
before callers wrap or aggregate the failure. Traces always go to stderr so an MCP
stdio transport can reserve stdout exclusively for JSON-RPC protocol frames.
"""

import heapq
import inspect
import itertools
import json
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar, cast

_TRACE_PREFIX = "ROOT CAUSE TRACE: "
_TRACE_SEQUENCE = itertools.count(1)
_TRACE_ID: ContextVar[str] = ContextVar("mmm_root_trace_id", default="")
_SPAN_ID: ContextVar[str] = ContextVar("mmm_root_span_id", default="")
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


def current_trace_id() -> str:
    """Return the active correlation ID, creating it at the first boundary."""

    value = _TRACE_ID.get()
    if value:
        return value
    value = uuid.uuid4().hex
    _TRACE_ID.set(value)
    return value


@contextmanager
def trace_scope(operation: str, *, trace_id: str = ""):
    """Correlate nested planner and execution events without changing public APIs."""

    parent_span = _SPAN_ID.get()
    trace_token = _TRACE_ID.set(trace_id or _TRACE_ID.get() or uuid.uuid4().hex)
    span = f"{operation}:{next(_TRACE_SEQUENCE)}"
    span_token = _SPAN_ID.set(span)
    try:
        yield {
            "trace_id": _TRACE_ID.get(),
            "span_id": span,
            "parent_span_id": parent_span,
        }
    finally:
        _SPAN_ID.reset(span_token)
        _TRACE_ID.reset(trace_token)


def _secret_key(value: Any) -> bool:
    key = str(value or "").casefold().replace("-", "_")
    return any(part in key for part in _SECRET_KEY_PARTS)


def _bounded_collection(items: Sequence[Any], total: int, *, depth: int) -> list[Any]:
    result = [bounded_safe(item, depth=depth + 1) for item in items]
    if total > _COLLECTION_LIMIT:
        result.append(f"<truncated:{total - _COLLECTION_LIMIT}>")
    return result


def _bounded_sequence(value: Sequence[Any], *, depth: int) -> list[Any]:
    """Bound a sequence without copying or traversing its unreported tail."""
    total = len(value)
    if isinstance(value, (list, tuple)):
        items = value[:_COLLECTION_LIMIT]
    else:
        items = tuple(itertools.islice(value, _COLLECTION_LIMIT))
    return _bounded_collection(items, total, depth=depth)


def _bounded_set(value: set[Any] | frozenset[Any], *, depth: int) -> list[Any]:
    """Keep deterministic set traces with O(limit) auxiliary memory.

    Sorting the full set made every trace O(N log N) and allocated O(N) temporary
    storage even though only 64 values are emitted. nsmallest keeps the same repr-key
    ordering contract for the reported prefix while bounding auxiliary memory and the
    sort factor to the trace limit.
    """
    total = len(value)
    if total <= _COLLECTION_LIMIT:
        items = sorted(value, key=repr)
    else:
        items = heapq.nsmallest(_COLLECTION_LIMIT, value, key=repr)
    return _bounded_collection(items, total, depth=depth)


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
    if isinstance(value, (set, frozenset)):
        return _bounded_set(value, depth=depth)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _bounded_sequence(value, depth=depth)
    return bounded_safe(str(value), depth=depth + 1)


def exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """Preserve the causal exception chain instead of only the final wrapper message."""
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
                    {"file": frame.filename, "line": frame.lineno, "function": frame.name}
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
        "schema_version": "mmm/root-cause-trace-v2",
        "trace_seq": next(_TRACE_SEQUENCE),
        "trace_id": current_trace_id(),
        "event": str(event),
    }
    span_id = _SPAN_ID.get()
    if span_id:
        payload["span_id"] = span_id
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
        file=sys.stderr,
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


__all__ = [
    "bounded_safe",
    "current_trace_id",
    "emit_root_cause",
    "exception_chain",
    "trace_scope",
    "traced_callable",
]
