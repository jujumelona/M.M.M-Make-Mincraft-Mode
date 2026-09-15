from __future__ import annotations

"""Per-incident diagnostic bundles for terminal CLI/UI failures.

The durable root-cause journal remains the canonical append-only trace. This module
adds a bounded, secret-safe snapshot that is easy to download and attach to a bug
report without weakening the original failure semantics.
"""

import json
import linecache
import os
import platform
import sys
import tempfile
import traceback
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .root_cause_trace import (
    bounded_safe,
    current_trace_id,
    durable_trace_path,
    emit_root_cause,
    exception_chain,
)

_DIAGNOSTIC_DIR_ENV = "MMM_DIAGNOSTIC_DIR"
_REPORT_SCHEMA = "mmm/failure-diagnostic-v1"
_MAX_CHAIN = 16
_MAX_FRAMES_PER_EXCEPTION = 20
_MAX_LOCALS_PER_FRAME = 64
_MAX_TRACE_TAIL_BYTES = 256 * 1024
_MAX_TRACE_TAIL_LINES = 200
_MAX_FORMATTED_TRACEBACK = 64 * 1024


def _safe_local(name: str, value: Any) -> Any:
    """Serialize useful local values without invoking arbitrary object reprs."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        try:
            return bounded_safe(value, key=name)
        except BaseException:
            return f"<unavailable:{type(value).__name__}>"
    if isinstance(value, Mapping) or isinstance(value, (list, tuple, set, frozenset)):
        try:
            return bounded_safe(value, key=name)
        except BaseException:
            return f"<unavailable:{type(value).__name__}>"
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _frame_snapshots(exc: BaseException) -> list[dict[str, Any]]:
    """Capture the failing frames and bounded locals needed to explain mismatches."""

    snapshots: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    chain_index = 0
    while current is not None and id(current) not in seen and chain_index < _MAX_CHAIN:
        seen.add(id(current))
        frames: list[tuple[Any, int]] = []
        tb = current.__traceback__
        while tb is not None:
            frames.append((tb.tb_frame, tb.tb_lineno))
            tb = tb.tb_next
        for frame, lineno in frames[-_MAX_FRAMES_PER_EXCEPTION:]:
            locals_snapshot: dict[str, Any] = {}
            try:
                local_items = tuple(frame.f_locals.items())[:_MAX_LOCALS_PER_FRAME]
            except BaseException:
                local_items = ()
            for raw_name, value in local_items:
                name = str(raw_name)
                if name in {"self", "cls"}:
                    continue
                locals_snapshot[name] = _safe_local(name, value)
            try:
                source_line = linecache.getline(frame.f_code.co_filename, lineno).strip()
            except BaseException:
                source_line = ""
            snapshots.append(
                {
                    "chain_index": chain_index,
                    "exception_type": type(current).__name__,
                    "file": frame.f_code.co_filename,
                    "line": lineno,
                    "function": frame.f_code.co_name,
                    "source": bounded_safe(source_line),
                    "locals": locals_snapshot,
                }
            )
        current = current.__cause__ if current.__cause__ is not None else current.__context__
        chain_index += 1
    return snapshots


def _formatted_traceback(exc: BaseException) -> str:
    try:
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except BaseException:
        return f"<traceback-unavailable:{type(exc).__name__}>"
    if len(rendered) > _MAX_FORMATTED_TRACEBACK:
        return rendered[-_MAX_FORMATTED_TRACEBACK:]
    return rendered


def _trace_tail(path: Path) -> list[str]:
    """Read only a bounded tail of the durable JSONL trace."""

    try:
        if not path.is_file():
            return []
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _MAX_TRACE_TAIL_BYTES))
            data = stream.read(_MAX_TRACE_TAIL_BYTES)
        text = data.decode("utf-8", "replace")
        lines = text.splitlines()
        if size > _MAX_TRACE_TAIL_BYTES and lines:
            lines = lines[1:]
        return lines[-_MAX_TRACE_TAIL_LINES:]
    except BaseException:
        return []


def _candidate_directories() -> tuple[Path, ...]:
    explicit = os.environ.get(_DIAGNOSTIC_DIR_ENV, "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(durable_trace_path().parent / "incidents")
    candidates.append(Path(tempfile.gettempdir()) / "mmm-diagnostics")
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.fspath(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _write_report_bytes(directory: Path, filename: str, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(os.fspath(target), flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("diagnostic report write returned zero bytes")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return target.resolve()


def emit_contract_mismatch(
    *,
    contract: str,
    field: str,
    expected: Any,
    actual: Any,
    stage: str,
    operation: str = "",
    reason: str = "contract values differ",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit expected/actual mismatch without changing caller behavior."""

    mismatch = {
        "contract": contract,
        "field": field,
        "expected": bounded_safe(expected, key="expected"),
        "actual": bounded_safe(actual, key="actual"),
    }
    merged = {"mismatch": mismatch}
    if details:
        merged["context"] = bounded_safe(details)
    emit_root_cause(
        "contract_mismatch",
        stage=stage,
        operation=operation,
        gate=contract,
        result="FAIL",
        reason=reason,
        details=merged,
    )
    return mismatch


def write_failure_report(
    exc: BaseException,
    *,
    stage: str,
    operation: str = "",
    context: Mapping[str, Any] | None = None,
    mismatch: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist one self-contained failure report and return its downloadable path.

    Diagnostics are strictly best-effort: a logging failure never replaces the original
    exception. Two locations are attempted so a read-only working directory still has
    a chance to produce a report in the system temporary directory.
    """

    trace_id = current_trace_id()
    incident_id = uuid.uuid4().hex
    safe_context = bounded_safe(context or {})
    safe_mismatch = bounded_safe(mismatch or {})
    emit_root_cause(
        "terminal_failure_capture",
        stage=stage,
        operation=operation,
        gate="failure_diagnostic",
        result="FAIL",
        reason=f"{type(exc).__name__}: {exc}",
        details={"incident_id": incident_id, "context": safe_context, "mismatch": safe_mismatch},
        exc=exc,
    )

    root_trace = durable_trace_path().resolve()
    report = {
        "schema_version": _REPORT_SCHEMA,
        "incident_id": incident_id,
        "trace_id": trace_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "operation": operation,
        "exception": {
            "type": type(exc).__name__,
            "message": bounded_safe(str(exc)),
            "chain": exception_chain(exc),
            "formatted_traceback": _formatted_traceback(exc),
        },
        "frame_snapshots": _frame_snapshots(exc),
        "mismatch": safe_mismatch,
        "context": safe_context,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "cwd": os.fspath(Path.cwd()),
        },
        "root_cause_trace": {
            "path": os.fspath(root_trace),
            "tail": _trace_tail(root_trace),
        },
    }
    try:
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            + "\n"
        ).encode("utf-8", "backslashreplace")
    except BaseException as serialization_exc:
        emit_root_cause(
            "failure_report_serialization_failure",
            stage=stage,
            operation=operation,
            gate="failure_diagnostic",
            result="FAIL",
            reason=f"{type(serialization_exc).__name__}: {serialization_exc}",
            exc=serialization_exc,
        )
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    filename = f"mmm-diagnostic-{stamp}-{trace_id[:12]}-{incident_id[:12]}.json"
    write_errors: list[str] = []
    for directory in _candidate_directories():
        try:
            path = _write_report_bytes(directory, filename, payload)
        except BaseException as write_exc:
            write_errors.append(f"{directory}: {type(write_exc).__name__}: {write_exc}")
            continue
        emit_root_cause(
            "failure_report_written",
            stage=stage,
            operation=operation,
            gate="failure_diagnostic",
            result="PASS",
            details={"incident_id": incident_id, "path": os.fspath(path)},
        )
        return path

    emit_root_cause(
        "failure_report_write_failure",
        stage=stage,
        operation=operation,
        gate="failure_diagnostic",
        result="FAIL",
        reason="No diagnostic directory was writable.",
        details={"incident_id": incident_id, "write_errors": write_errors},
    )
    return None


__all__ = ["emit_contract_mismatch", "write_failure_report"]
