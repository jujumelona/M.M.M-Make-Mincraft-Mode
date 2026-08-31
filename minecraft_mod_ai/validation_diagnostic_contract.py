from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .java_lsp import JDTLanguageServerError

_DIAGNOSTIC_SUCCESS_STATUSES = {"PASS", "OK", "AVAILABLE"}
_JDT_AVAILABILITY_ERRORS = (OSError, TimeoutError, JDTLanguageServerError)


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


def diagnostic_items(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize JDT URI mappings and legacy lists while preserving source URIs."""

    if not isinstance(receipt, Mapping):
        return []
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


def _is_error(item: Mapping[str, Any]) -> bool:
    try:
        return int(item.get("severity", 1)) == 1
    except (TypeError, ValueError, OverflowError):
        return True


def diagnostic_errors(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return severity-1 diagnostics plus fail-closed availability evidence."""

    errors = [item for item in diagnostic_items(receipt) if _is_error(item)]
    unavailable = _availability_error(receipt if isinstance(receipt, Mapping) else {})
    if unavailable is not None:
        errors.append(unavailable)
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

    try:
        service = diagnostics_factory()
        callback = service.diagnostics
        kwargs: dict[str, Any] = {"timeout_seconds": timeout_seconds}
        if relative_files is not None and _accepts_keyword(callback, "relative_files"):
            kwargs["relative_files"] = relative_files
        receipt = callback(project_root, **kwargs)
    except _JDT_AVAILABILITY_ERRORS as exc:
        return unavailable_receipt(exc)

    if not isinstance(receipt, Mapping):
        return {
            "status": "UNAVAILABLE",
            "error": "JDT diagnostics returned a non-mapping receipt.",
            "diagnostics": {},
        }
    return dict(receipt)


__all__ = [
    "diagnostic_errors",
    "diagnostic_items",
    "run_diagnostics",
    "unavailable_receipt",
]
