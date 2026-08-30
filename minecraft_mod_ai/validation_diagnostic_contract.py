from __future__ import annotations

from typing import Any


_DIAGNOSTIC_SUCCESS_STATUSES = {"PASS", "OK", "AVAILABLE"}


def _availability_error(receipt: dict[str, Any]) -> dict[str, Any] | None:
    """Return a blocking diagnostic when the JDT receipt is not trustworthy.

    Native ``JavaLanguageService.diagnostics`` receipts do not currently carry a
    top-level status, so an absent status is valid when the diagnostics payload is
    present and well-formed. Explicit failure/unavailable states, an error field,
    or a malformed/missing diagnostics payload must fail closed.
    """

    status = str(receipt.get("status") or "").strip().upper()
    error = str(receipt.get("error") or "").strip()
    raw = receipt.get("diagnostics")
    malformed = "diagnostics" not in receipt or not isinstance(raw, (dict, list))
    failed_status = bool(status and status not in _DIAGNOSTIC_SUCCESS_STATUSES)
    if not (failed_status or error or malformed):
        return None

    details: list[str] = []
    if status:
        details.append(f"status={status}")
    if error:
        details.append(error)
    if malformed:
        details.append("diagnostics payload is missing or malformed")
    return {
        "severity": 1,
        "source": "jdtls",
        "code": "JDT_DIAGNOSTICS_UNAVAILABLE",
        "message": "JDT diagnostics are unavailable: " + "; ".join(details),
    }


def diagnostic_errors(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("diagnostics", {})
    if isinstance(raw, dict):
        values = [
            item
            for group in raw.values()
            if isinstance(group, list)
            for item in group
            if isinstance(item, dict)
        ]
    elif isinstance(raw, list):
        values = [item for item in raw if isinstance(item, dict)]
    else:
        values = []

    # JDT-LS/LSP severity 1 is Error and 2 is Warning. Warnings remain in the
    # diagnostic receipt but must not trigger an expensive repair/build deferral.
    errors = [item for item in values if int(item.get("severity", 1)) == 1]
    unavailable = _availability_error(receipt)
    if unavailable is not None:
        errors.append(unavailable)
    return errors


def install(validation_module: Any) -> None:
    """Make progressive repair consume a fail-closed JDT-LS diagnostic receipt."""

    diagnostic_errors._mmm_flattened_jdt_mapping = True
    diagnostic_errors._mmm_fail_closed_jdt_receipt = True
    validation_module._diagnostic_errors = diagnostic_errors
