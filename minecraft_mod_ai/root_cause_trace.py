from __future__ import annotations

"""Source-first structured tracing for host-owned execution boundaries.

The journal is append-only and independent of model output. Boundary records contain
boundary identifiers and complete explicit diagnostic payloads; failure records point at
the deepest causal source location so a failing definition can be found from the log
without repository-wide searching.
"""

import heapq
import inspect
import itertools
import json
import os
import sys
import threading
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
_DIAGNOSTIC_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "mmm_root_diagnostic_context", default={}
)
_TRACE_WRITE_LOCK = threading.Lock()
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
_IDENTIFIER_KEYS = (
    "run_id",
    "trace_id",
    "stage",
    "operation",
    "template",
    "template_id",
    "template_path",
    "logical_path",
    "artifact",
    "artifact_id",
    "source",
    "source_path",
    "path",
    "invariant",
    "blocked_stage",
    "status",
    "name",
    "id",
)
_DIAGNOSTIC_KEYS = (
    "run_id",
    "stage",
    "operation",
    "template",
    "template_id",
    "template_path",
    "logical_path",
    "artifact",
    "artifact_id",
    "source",
    "source_path",
    "source_symbol",
    "source_line",
    "invariant",
    "expected",
    "actual",
    "offending_fields",
    "blocked_stage",
)
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


@contextmanager
def diagnostic_context(**fields: Any):
    """Attach small source/contract identifiers to every nested trace event."""

    parent = dict(_DIAGNOSTIC_CONTEXT.get())
    merged = dict(parent)
    for key, value in fields.items():
        if key in _DIAGNOSTIC_KEYS and value not in (None, "", [], {}, ()):
            merged[key] = _trace_safe(value, key=key)
    token = _DIAGNOSTIC_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _DIAGNOSTIC_CONTEXT.reset(token)


def _secret_key(value: Any) -> bool:
    key = str(value or "").casefold().replace("-", "_")
    return any(part in key for part in _SECRET_KEY_PARTS)


def _bounded_collection(items: Sequence[Any], total: int, *, depth: int) -> list[Any]:
    result = [bounded_safe(item, depth=depth + 1) for item in items]
    if total > _COLLECTION_LIMIT:
        result.append(f"<truncated:{total - _COLLECTION_LIMIT}>")
    return result


def _bounded_sequence(value: Sequence[Any], *, depth: int) -> list[Any]:
    total = len(value)
    if isinstance(value, (list, tuple)):
        items = value[:_COLLECTION_LIMIT]
    else:
        items = tuple(itertools.islice(value, _COLLECTION_LIMIT))
    return _bounded_collection(items, total, depth=depth)


def _bounded_set(value: set[Any] | frozenset[Any], *, depth: int) -> list[Any]:
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


def full_trace_enabled() -> bool:
    """Full console/journal diagnostics are the default; compact is explicit."""
    return os.environ.get("MMM_ROOT_CAUSE_TRACE_DETAIL", "full").strip().lower() != "compact"


def _trace_safe(value: Any, *, key: str = "") -> Any:
    if not full_trace_enabled():
        return bounded_safe(value, key=key)
    if key and _secret_key(key):
        return "<redacted>"
    from .planner_trace_artifacts import _redacted

    return _redacted(value, set())


def _scalar_identifier(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return bounded_safe(value)
    return None


def _value_summary(value: Any) -> dict[str, Any]:
    """Summarize a boundary value without serializing the domain object itself."""

    summary: dict[str, Any] = {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float, str)):
        summary["value"] = bounded_safe(value)
        return summary
    if isinstance(value, Mapping):
        summary["size"] = len(value)
        identifiers: dict[str, Any] = {}
        for key in _IDENTIFIER_KEYS:
            if key in value:
                scalar = _scalar_identifier(value.get(key))
                if scalar is not None:
                    identifiers[key] = scalar
        if identifiers:
            summary["identifiers"] = identifiers
        return summary
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        summary["size"] = len(value)
        return summary
    identifiers = {}
    for key in _IDENTIFIER_KEYS:
        try:
            scalar = _scalar_identifier(getattr(value, key))
        except (AttributeError, TypeError, ValueError):
            continue
        except BaseException:
            continue
        if scalar is not None:
            identifiers[key] = scalar
    if identifiers:
        summary["identifiers"] = identifiers
    return summary


def _argument_summary(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        bound = signature.bind_partial(*args, **dict(kwargs))
    except Exception as bind_exc:
        return {"binding_error": f"{type(bind_exc).__name__}: {bind_exc}"}
    return {name: _value_summary(value) for name, value in bound.arguments.items()}


def _deepest_exception(exc: BaseException) -> BaseException:
    current = exc
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        child = current.__cause__ if current.__cause__ is not None else current.__context__
        if child is None:
            break
        current = child
    return current


def _exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    root = _deepest_exception(exc)
    diagnostics: dict[str, Any] = dict(_DIAGNOSTIC_CONTEXT.get())
    for key in _DIAGNOSTIC_KEYS:
        if key in diagnostics:
            continue
        try:
            value = getattr(root, key)
        except (AttributeError, TypeError, ValueError):
            continue
        except BaseException:
            continue
        if value not in (None, "", [], {}, ()):
            diagnostics[key] = _trace_safe(value, key=key)
    try:
        frames = traceback.extract_tb(root.__traceback__) if root.__traceback__ else []
    except BaseException:
        frames = []
    if frames:
        leaf = frames[-1]
        diagnostics["source"] = {
            "file": leaf.filename,
            "line": leaf.lineno,
            "function": leaf.name,
        }
    diagnostics["cause_type"] = type(root).__name__
    try:
        diagnostics["cause"] = _trace_safe(str(root))
    except BaseException:
        diagnostics["cause"] = f"<unprintable:{type(root).__name__}>"
    return diagnostics


def exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """Preserve the causal exception chain exactly once for the trace's first failure."""

    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending and (full_trace_enabled() or len(chain) < 16):
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        try:
            frames = traceback.extract_tb(current.__traceback__) if current.__traceback__ else []
            if not full_trace_enabled():
                frames = frames[-20:]
        except BaseException:
            frames = []
        try:
            message = _trace_safe(str(current))
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
        # Supports both Python 3.11 groups and the Python 3.10 backport used by
        # AnyIO. Keep traversal bounded and cycle-safe like ordinary causes.
        children = getattr(current, "exceptions", ())
        if isinstance(children, tuple):
            pending.extend(
                child for child in reversed(children) if isinstance(child, BaseException)
            )
        cause = current.__cause__ if current.__cause__ is not None else current.__context__
        if cause is not None:
            pending.append(cause)
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


def _is_failure(result: str, exc: BaseException | None) -> bool:
    if exc is not None:
        return True
    return str(result or "").strip().upper() in _FAILURE_STATUSES


def _append_durable_line(line: bytes, *, sync: bool) -> None:
    path = durable_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    with _TRACE_WRITE_LOCK:
        fd = os.open(os.fspath(path), flags, 0o600)
        try:
            offset = 0
            while offset < len(line):
                written = os.write(fd, line[offset:])
                if written <= 0:
                    raise OSError("durable trace write returned zero bytes")
                offset += written
            if sync:
                os.fsync(fd)
        finally:
            os.close(fd)


def _stderr_line(line: str, *, flush: bool) -> None:
    try:
        sys.stderr.write(_TRACE_PREFIX + line + "\n")
        if flush:
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
        encoded = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8", "backslashreplace"
        ) + b"\n"
    except BaseException:
        encoded = b'{"schema_version":"mmm/root-cause-trace-emergency-v1","event":"trace_emergency_fallback"}\n'
    try:
        _append_durable_line(encoded, sync=True)
    except BaseException:
        pass
    try:
        _stderr_line(encoded.decode("utf-8", "replace").rstrip("\n"), flush=True)
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
    """Emit one append-only event; only the first failure carries the full exception chain."""

    trace_seq = next(_TRACE_SEQUENCE)
    trace_id = current_trace_id()
    failure = _is_failure(result, exc)
    failure_sync = failure and event not in {
        "detailed_planning_timeout",
        "planning_research_timeout",
    }
    try:
        payload: dict[str, Any] = {
            "schema_version": "mmm/root-cause-trace-v4",
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
            payload["reason"] = _trace_safe(reason)

        context = dict(_DIAGNOSTIC_CONTEXT.get())
        if context:
            payload["diagnostic_context"] = _trace_safe(context)
        if details:
            safe_details = _trace_safe(details)
            payload["details"] = safe_details
            from .planner_trace_artifacts import save_trace_artifact

            try:
                payload["details_artifact"] = save_trace_artifact(
                    details,
                    durable_trace_path().parent / "artifacts",
                    sync=failure_sync,
                )
            except Exception as artifact_error:
                payload["details_artifact_error"] = type(artifact_error).__name__

        first_failure_seq = _FIRST_FAILURE_SEQ.get()
        is_first_failure = False
        if failure:
            if first_failure_seq <= 0:
                first_failure_seq = trace_seq
                _FIRST_FAILURE_SEQ.set(trace_seq)
                is_first_failure = True
            payload["first_failure_seq"] = first_failure_seq
            payload["is_first_failure"] = is_first_failure
        elif first_failure_seq > 0:
            payload["first_failure_seq"] = first_failure_seq

        if exc is not None:
            payload["failure"] = _exception_diagnostics(exc)
            if is_first_failure or full_trace_enabled():
                payload["exception_chain"] = exception_chain(exc)
            else:
                payload["exception_chain_ref"] = first_failure_seq

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        _append_durable_line(
            (serialized + "\n").encode("utf-8", "backslashreplace"),
            sync=failure_sync,
        )
        _stderr_line(
            serialized,
            flush=failure
            or event
            in {
                "mcp_verifier_transport_retry",
                "mcp_verifier_transport_recovered",
                "jdt_stderr",
                "jdt_server_progress",
                "gradle_command_start",
                "gradle_command_output",
                "gradle_command_result",
                "gradle_distribution_start",
                "gradle_cache_lock_wait",
                "gradle_cache_lock_acquired",
                "planning_semantic_research_summary",
                "planner_requirement_page",
                "planner_requirement_page_received",
                "planner_requirement_coverage",
            },
        )
    except BaseException as logger_exc:
        _emergency_trace(
            event=event,
            trace_id=trace_id,
            trace_seq=trace_seq,
            original_exc=exc,
            logger_exc=logger_exc,
        )


def traced_callable(function: F, *, stage: str, operation: str | None = None) -> F:
    """Trace a host boundary using identifiers/types rather than full argument objects."""

    operation_name = operation or function.__name__
    signature = inspect.signature(function)

    def invoke(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        started = time.monotonic()
        emit_root_cause(
            "operation_start",
            stage=stage,
            operation=operation_name,
            gate="host_boundary",
            result="START",
            details={"arguments": _argument_summary(signature, args, kwargs)},
        )
        try:
            value = function(*args, **dict(kwargs))
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
                "result_summary": _value_summary(value),
            },
        )
        return value

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if _SPAN_ID.get():
            return invoke(args, kwargs)
        # A standalone host boundary is a distinct trace. Force a fresh correlation ID
        # instead of inheriting a ContextVar value left by a previous independent call.
        with trace_scope(operation_name, trace_id=uuid.uuid4().hex):
            return invoke(args, kwargs)

    return cast(F, wrapped)


__all__ = [
    "bounded_safe",
    "current_trace_id",
    "diagnostic_context",
    "durable_trace_path",
    "emit_root_cause",
    "exception_chain",
    "trace_scope",
    "traced_callable",
]
