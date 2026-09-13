from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .java_lsp import JDTLanguageServerError
from .root_cause_trace import emit_root_cause, exception_chain

_DIAGNOSTIC_SUCCESS_STATUSES = {"PASS", "OK", "AVAILABLE"}
_JDT_AVAILABILITY_ERRORS = (OSError, TimeoutError, JDTLanguageServerError)
_DIAGNOSTIC_ENVELOPE_KEYS = (
    "structured_content",
    "structuredContent",
    "parsed_text",
)
_MAX_ENVELOPE_DEPTH = 8
_CORE_RUNTIME_UNRESOLVED_PATTERNS = (
    "java.lang.object cannot be resolved",
    "the type java.lang.object cannot be resolved",
    "the type object cannot be resolved. it is indirectly referenced from required .class files",
    "implicit super constructor object() is undefined for default constructor",
)


def _mapping_keys(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    return sorted(str(key) for key in value.keys())


def _diagnostics_shape_error(raw: Any) -> str:
    if isinstance(raw, Mapping):
        for uri, group in raw.items():
            if not isinstance(group, list):
                return f"diagnostics group for {uri!r} is not a list"
            if any(not isinstance(item, Mapping) for item in group):
                return f"diagnostics group for {uri!r} contains a non-mapping item"
        return ""
    if isinstance(raw, list):
        if any(not isinstance(item, Mapping) for item in raw):
            return "diagnostics list contains a non-mapping item"
        return ""
    return "diagnostics payload is missing or malformed"


def unwrap_diagnostic_receipt(
    receipt: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Unwrap only reviewed MCP transport envelopes around a JDT receipt.

    AgentToolRuntime intentionally preserves MCP transport metadata around
    ``structured_content``. Verifier semantics must inspect the inner JDT receipt,
    never mistake the transport envelope itself for the receipt. Every unwrap step is
    traced so one execution log proves exactly which shape was received and selected.
    """

    if not isinstance(receipt, Mapping):
        emit_root_cause(
            "diagnostic_receipt_unwrap",
            stage="verify",
            operation="java_diagnostics",
            gate="receipt_shape",
            result="FAIL",
            reason="diagnostic receipt is not a mapping",
            details={"received_type": type(receipt).__name__},
        )
        return {}, ()

    current: Mapping[str, Any] = receipt
    path: list[str] = []
    seen: set[int] = set()
    for depth in range(_MAX_ENVELOPE_DEPTH + 1):
        if id(current) in seen:
            emit_root_cause(
                "diagnostic_receipt_unwrap",
                stage="verify",
                operation="java_diagnostics",
                gate="receipt_shape",
                result="FAIL",
                reason="cycle detected while unwrapping diagnostic receipt",
                details={"depth": depth, "path": path, "keys": _mapping_keys(current)},
            )
            return current, tuple(path)
        seen.add(id(current))

        keys = _mapping_keys(current)
        if "diagnostics" in current:
            emit_root_cause(
                "diagnostic_receipt_unwrap",
                stage="verify",
                operation="java_diagnostics",
                gate="receipt_shape",
                result="PASS",
                reason="resolved terminal JDT diagnostic receipt",
                details={
                    "depth": depth,
                    "path": path,
                    "keys": keys,
                    "status": current.get("status"),
                    "has_error": bool(str(current.get("error") or "").strip()),
                    "diagnostics_type": type(current.get("diagnostics")).__name__,
                },
            )
            return current, tuple(path)

        selected_key = ""
        selected: Mapping[str, Any] | None = None
        for key in _DIAGNOSTIC_ENVELOPE_KEYS:
            candidate = current.get(key)
            if isinstance(candidate, Mapping):
                selected_key = key
                selected = candidate
                break

        if selected is None:
            emit_root_cause(
                "diagnostic_receipt_unwrap",
                stage="verify",
                operation="java_diagnostics",
                gate="receipt_shape",
                result="FAIL",
                reason="no diagnostics field or reviewed structured-content envelope was present",
                details={"depth": depth, "path": path, "keys": keys},
            )
            return current, tuple(path)

        emit_root_cause(
            "diagnostic_receipt_layer",
            stage="verify",
            operation="java_diagnostics",
            gate="transport_envelope",
            result="PASS",
            reason="unwrapped reviewed MCP diagnostic envelope",
            details={
                "depth": depth,
                "selected_key": selected_key,
                "outer_keys": keys,
                "inner_keys": _mapping_keys(selected),
            },
        )
        path.append(selected_key)
        current = selected

    emit_root_cause(
        "diagnostic_receipt_unwrap",
        stage="verify",
        operation="java_diagnostics",
        gate="receipt_shape",
        result="FAIL",
        reason="diagnostic envelope exceeded maximum reviewed nesting depth",
        details={"path": path, "keys": _mapping_keys(current)},
    )
    return current, tuple(path)


def _availability_error(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a blocking diagnostic when a JDT receipt is not trustworthy."""

    status = str(receipt.get("status") or "").strip().upper()
    error = str(receipt.get("error") or "").strip()
    malformed = (
        "diagnostics payload is missing or malformed"
        if "diagnostics" not in receipt
        else _diagnostics_shape_error(receipt.get("diagnostics"))
    )
    failed_status = bool(status and status not in _DIAGNOSTIC_SUCCESS_STATUSES)
    if not (failed_status or error or malformed):
        return None

    details: list[str] = []
    if status:
        details.append(f"status={status}")
    if error:
        details.append(error)
    if malformed:
        details.append(malformed)
    return {
        "severity": 1,
        "source": "jdtls",
        "code": "JDT_DIAGNOSTICS_UNAVAILABLE",
        "message": "JDT diagnostics are unavailable: " + "; ".join(details),
    }


def _diagnostic_items_from_receipt(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("diagnostics", {})
    if isinstance(raw, Mapping):
        values: list[dict[str, Any]] = []
        for uri, group in sorted(raw.items(), key=lambda pair: str(pair[0])):
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, Mapping):
                    continue
                normalized = dict(item)
                normalized.setdefault("uri", str(uri))
                values.append(normalized)
        return values
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def diagnostic_items(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize JDT URI mappings and legacy lists while preserving source URIs."""

    normalized, _path = unwrap_diagnostic_receipt(receipt)
    return _diagnostic_items_from_receipt(normalized)


def _is_error(item: Mapping[str, Any]) -> bool:
    try:
        return int(item.get("severity", 1)) == 1
    except (TypeError, ValueError, OverflowError):
        return True


def _core_runtime_readiness_error(
    items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Detect diagnostics proving that JDT's Java runtime/classpath is not ready.

    These messages are not actionable source defects. Returning one infrastructure
    diagnostic suppresses the ordinary source diagnostics so repair cannot mutate user
    code in response to a broken verifier workspace.
    """

    matches: list[str] = []
    for item in items:
        if not _is_error(item):
            continue
        message = str(item.get("message") or "").strip()
        normalized = " ".join(message.casefold().split())
        if any(pattern in normalized for pattern in _CORE_RUNTIME_UNRESOLVED_PATTERNS):
            matches.append(message)
    if not matches:
        return None
    return {
        "severity": 1,
        "source": "jdtls",
        "code": "JDT_WORKSPACE_NOT_READY",
        "message": (
            "JDT workspace is not ready: Java core runtime symbols are unresolved; "
            "source repair is suppressed until verifier readiness is restored. "
            f"Evidence: {matches[0]}"
        ),
    }


def diagnostic_errors(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return source errors, or one fail-closed verifier/readiness diagnostic."""

    normalized, path = unwrap_diagnostic_receipt(receipt)
    items = _diagnostic_items_from_receipt(normalized)
    source_errors = [item for item in items if _is_error(item)]
    unavailable = _availability_error(normalized)
    readiness = None if unavailable is not None else _core_runtime_readiness_error(items)

    if unavailable is not None:
        errors = [unavailable]
    elif readiness is not None:
        errors = [readiness]
    else:
        errors = source_errors

    status = str(normalized.get("status") or "").strip().upper()
    infrastructure_error = unavailable or readiness
    emit_root_cause(
        "diagnostic_receipt_classified",
        stage="verify",
        operation="java_diagnostics",
        gate="verifier_semantics",
        result=("FAIL" if infrastructure_error is not None or errors else "PASS"),
        reason=(
            "JDT receipt is unavailable or malformed"
            if unavailable is not None
            else (
                "JDT core runtime symbols are unresolved; verifier workspace is not ready"
                if readiness is not None
                else ("JDT published severity-1 diagnostics" if errors else "JDT receipt is healthy")
            )
        ),
        details={
            "envelope_path": list(path),
            "status": status,
            "receipt_keys": _mapping_keys(normalized),
            "diagnostic_item_count": len(items),
            "severity_1_count": len(source_errors),
            "availability_error": unavailable,
            "readiness_error": readiness,
            "source_repair_suppressed": infrastructure_error is not None,
        },
    )
    return errors


def _accepts_keyword(callback: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == name
            and parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        )
        for parameter in parameters
    )


def unavailable_receipt(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "error": f"{type(exc).__name__}: {exc}",
        "error_type": type(exc).__name__,
        "diagnostics": {},
    }


def run_diagnostics(
    diagnostics_factory: Any,
    project_root: str | Path,
    *,
    relative_files: Any = None,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """Run JDT under one compatibility and exception-classification policy.

    Only process/transport availability failures become UNAVAILABLE receipts.
    Programming errors propagate. Legacy doubles without ``relative_files`` are
    detected by signature rather than by catching TypeError from their body.
    """

    started = time.monotonic()
    root_text = str(project_root)
    requested_files = list(relative_files) if relative_files is not None else None
    emit_root_cause(
        "diagnostic_run_start",
        stage="verify",
        operation="java_diagnostics",
        gate="diagnostic_service",
        result="START",
        details={
            "project_root": root_text,
            "relative_files": requested_files,
            "timeout_seconds": timeout_seconds,
            "factory": getattr(diagnostics_factory, "__qualname__", repr(diagnostics_factory)),
        },
    )

    try:
        service = diagnostics_factory()
        callback = service.diagnostics
        kwargs: dict[str, Any] = {"timeout_seconds": timeout_seconds}
        accepts_relative_files = _accepts_keyword(callback, "relative_files")
        if relative_files is not None and accepts_relative_files:
            kwargs["relative_files"] = relative_files
        emit_root_cause(
            "diagnostic_service_ready",
            stage="verify",
            operation="java_diagnostics",
            gate="diagnostic_service",
            result="PASS",
            details={
                "service_type": type(service).__name__,
                "callback": getattr(callback, "__qualname__", repr(callback)),
                "accepts_relative_files": accepts_relative_files,
                "invocation_kwargs": kwargs,
            },
        )
        receipt = callback(project_root, **kwargs)
    except _JDT_AVAILABILITY_ERRORS as exc:
        unavailable = unavailable_receipt(exc)
        emit_root_cause(
            "diagnostic_run_unavailable",
            stage="verify",
            operation="java_diagnostics",
            gate="diagnostic_service",
            result="UNAVAILABLE",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "exception_chain": exception_chain(exc),
            },
            exc=exc,
        )
        return unavailable
    except BaseException as exc:
        emit_root_cause(
            "diagnostic_run_programming_failure",
            stage="verify",
            operation="java_diagnostics",
            gate="diagnostic_service",
            result="FAIL",
            reason=f"{type(exc).__name__}: {exc}",
            details={"elapsed_ms": round((time.monotonic() - started) * 1000.0, 3)},
            exc=exc,
        )
        raise

    if not isinstance(receipt, Mapping):
        unavailable = {
            "status": "UNAVAILABLE",
            "error": "JDT diagnostics returned a non-mapping receipt.",
            "error_type": type(receipt).__name__,
            "diagnostics": {},
        }
        emit_root_cause(
            "diagnostic_run_invalid_receipt",
            stage="verify",
            operation="java_diagnostics",
            gate="receipt_shape",
            result="UNAVAILABLE",
            reason="JDT diagnostics returned a non-mapping receipt",
            details={
                "received_type": type(receipt).__name__,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            },
        )
        return unavailable

    result = dict(receipt)
    normalized, path = unwrap_diagnostic_receipt(result)
    items = _diagnostic_items_from_receipt(normalized)
    unavailable = _availability_error(normalized)
    readiness = None if unavailable is not None else _core_runtime_readiness_error(items)
    infrastructure_error = unavailable or readiness
    emit_root_cause(
        "diagnostic_run_result",
        stage="verify",
        operation="java_diagnostics",
        gate="diagnostic_service",
        result="UNAVAILABLE" if infrastructure_error is not None else "PASS",
        reason=(
            "diagnostic service returned an unready verifier receipt"
            if readiness is not None
            else "diagnostic service returned a receipt"
        ),
        details={
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            "outer_keys": _mapping_keys(result),
            "envelope_path": list(path),
            "receipt_keys": _mapping_keys(normalized),
            "status": normalized.get("status"),
            "diagnostic_item_count": len(items),
            "availability_error": unavailable,
            "readiness_error": readiness,
        },
    )
    return result


__all__ = [
    "diagnostic_errors",
    "diagnostic_items",
    "run_diagnostics",
    "unavailable_receipt",
    "unwrap_diagnostic_receipt",
]
