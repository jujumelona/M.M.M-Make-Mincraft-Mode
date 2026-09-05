from __future__ import annotations

"""Bounded, secret-safe, durable structured tracing for host-owned execution boundaries.

The trace is independent of model output. It records what the host actually attempted,
what gate/result was observed, and the original exception chain before callers wrap or
aggregate the failure.

Every event is mirrored to stderr for interactive visibility and appended to a durable
JSONL trace journal so process/UI truncation cannot erase the critical tail. The
durable writer deliberately uses only primitive os-level append/write/fsync operations
and has a minimal emergency fallback so a diagnostic serialization failure cannot
replace the first production failure.
"""

import heapq
import inspect
import itertools
import json
import os
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

_TRACE_PREFIX = "ROOT CAUSE TRACE: "
_TRACE_SEQUENCE = itertools.count(1)
_TRACE_ID: ContextVar[str] = ContextVar("mmm_root_trace_id", default="")
_SPAN_ID: ContextVar[str] = ContextVar("mmm_root_span_id", default="")
_FIRST_FAILURE_SEQ: ContextVar[int] = ContextVar("mmm_root_first_failure_seq", default=0)
_STRING_LIMIT = 512
_COLLECTION_LIMIT = 64
_DEPTH_LIMIT = 5
_TRACE_PATH_ENV = "MMM_ROOT_CAUSE_TRACE_PATH"
_RUN_DIR_ENV = "MMM_RUN_DIR"
_DEFAULT_TRACE_RELATIVE = Path(".mmm") / "traces" / "root_cause.jsonl"
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
_FAILURE_STATUSES = frozenset(
    {
        "FAIL",
        "FAILED",
        "ERROR",
        "INVALID",
        "UNAVAILABLE",
        "TIMEOUT",
        "TIMED_OUT",
        "UNHEALTHY",
    }
)
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


def durable_trace_path() -> Path:
    """Resolve the durable append-only trace path without model-owned input."""

    explicit = os.environ.get(_TRACE_PATH_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    run_dir = os.environ.get(_RUN_DIR_ENV, "").strip()
    if run_dir:
        return Path(run_dir).expanduser() / "root_cause.jsonl"
    return Path.cwd() / _DEFAULT_TRACE_RELATIVE


@contextmanager
def trace_scope(operation: str, *, trace_id: str = ""):
    """Correlate nested planner and execution events without changing public APIs."""

    parent_span = _SPAN_ID.get()
    is_root_scope = not parent_span
    trace_token = _TRACE_ID.set(trace_id or _TRACE_ID.get() or uuid.uuid4().hex)
    first_failure_token = _FIRST_FAILURE_SEQ.set(0) if is_root_scope else None
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
        if first_failure_token is not None:
            _FIRST_FAILURE_SEQ.reset(first_failure_token)
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
    """Keep deterministic set traces with O(limit) auxiliary memory."""

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
    try:
        rendered = str(value)
    except BaseException:
        rendered = f"<unprintable:{type(value).__name__}>"
    return bounded_safe(rendered, depth=depth + 1)


def exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """Preserve the causal exception chain instead of only the final wrapper message."""

    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 16:
        seen.add(id(current))
        try:
            frames = (
                traceback.extract_tb(current.__traceback__)[-20:]
                if current.__traceback__
                else []
            )
        except BaseException:
            frames = []
        try:
            message = bounded_safe(str(current))
        except BaseException:
            message = f"<unprintable:{type(current).__name__}>"
        chain.append(
            {
                "type": type(current).__name__,
                "message": message,
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
    status = str(
        value.get("status") or value.get("state") or value.get("outcome") or ""
    ).strip().upper()
    if status in _FAILURE_STATUSES:
        return "FAIL"
    if status in _SKIP_STATUSES:
        return "SKIP"
    if value.get("ok") is False or value.get("success") is False:
        return "FAIL"
    return "PASS"


def _is_failure(result: str, exc: BaseException | None) -> bool:
    if exc is not None:
        return True
    return str(result or "").strip().upper() in _FAILURE_STATUSES


def _append_durable_line(line: bytes) -> None:
    """Append one already-serialized JSONL record using primitive os operations."""

    path = durable_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(os.fspath(path), flags, 0o600)
    try:
        offset = 0
        while offset < len(line):
            written = os.write(fd, line[offset:])
            if written <= 0:
                raise OSError("durable trace write returned zero bytes")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def _stderr_line(line: str) -> None:
    try:
        sys.stderr.write(_TRACE_PREFIX + line + "\n")
        sys.stderr.flush()
    except BaseException:
        pass


def _emergency_trace(
    *,
    event: Any,
    trace_id: str,
    trace_seq: int,
    original_exc: BaseException | None,
    logger_exc: BaseException,
) -> None:
    """Best-effort fallback that cannot mask the caller's original failure."""

    record = {
        "schema_version": "mmm/root-cause-trace-emergency-v1",
        "trace_seq": trace_seq,
        "trace_id": trace_id,
        "event": "trace_emergency_fallback",
        "original_event": str(event),
        "original_exception_type": type(original_exc).__name__ if original_exc is not None else "",
        "logger_exception_type": type(logger_exc).__name__,
    }
    try:
        encoded = (
            json.dumps(
                record,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8", "backslashreplace")
            + b"\n"
        )
    except BaseException:
        encoded = (
            b'{"schema_version":"mmm/root-cause-trace-emergency-v1",'
            b'"event":"trace_emergency_fallback"}\n'
        )
    try:
        _append_durable_line(encoded)
    except BaseException:
        pass
    try:
        _stderr_line(encoded.decode("utf-8", "replace").rstrip("\n"))
    except BaseException:
        pass


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
    """Emit one durable event without allowing diagnostics to replace first cause."""

    trace_seq = next(_TRACE_SEQUENCE)
    trace_id = current_trace_id()
    try:
        payload: dict[str, Any] = {
            "schema_version": "mmm/root-cause-trace-v3",
            "trace_seq": trace_seq,
            "trace_id": trace_id,
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

        if _is_failure(result, exc):
            first_failure_seq = _FIRST_FAILURE_SEQ.get()
            if first_failure_seq <= 0:
                first_failure_seq = trace_seq
                _FIRST_FAILURE_SEQ.set(trace_seq)
            payload["first_failure_seq"] = first_failure_seq
            payload["is_first_failure"] = first_failure_seq == trace_seq
        elif _FIRST_FAILURE_SEQ.get() > 0:
            payload["first_failure_seq"] = _FIRST_FAILURE_SEQ.get()

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        _append_durable_line((serialized + "\n").encode("utf-8", "backslashreplace"))
        _stderr_line(serialized)
    except BaseException as logger_exc:
        _emergency_trace(
            event=event,
            trace_id=trace_id,
            trace_seq=trace_seq,
            original_exc=exc,
            logger_exc=logger_exc,
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
            safe_arguments = {
                "argument_binding": f"{type(bind_exc).__name__}: {bind_exc}"
            }
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
    "durable_trace_path",
    "emit_root_cause",
    "exception_chain",
    "trace_scope",
    "traced_callable",
]
