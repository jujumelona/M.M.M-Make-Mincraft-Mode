from __future__ import annotations

"""Canonical machine-readable trace registry.

Static ledger metadata remains owned by :mod:`ledger_traceability`; executable
pytest routing remains owned by :mod:`ledger_regression_execution`. This module
joins the two without duplicating either source of truth.
"""

from dataclasses import replace

from .ledger_regression_execution import REGRESSION_EXECUTION_ROUTES
from .ledger_traceability import (
    ACCEPTANCE_MANIFEST,
    FAMILY_OWNERS,
    DecisionReceipt,
    LedgerTraceAudit,
    RegressionRoute,
    audit_ledger_text,
    validate_decision_receipt,
    validate_manifest_snapshot,
)
from .ledger_traceability import REGRESSION_MANIFEST as _LEDGER_REGRESSION_METADATA


def _validate_regression_registry_parity() -> None:
    """Reject missing/orphan routes without coupling validity to a magic row count."""

    metadata_ids = set(_LEDGER_REGRESSION_METADATA)
    execution_ids = set(REGRESSION_EXECUTION_ROUTES)
    if metadata_ids == execution_ids:
        return
    missing_execution = sorted(metadata_ids - execution_ids)
    orphan_execution = sorted(execution_ids - metadata_ids)
    raise RuntimeError(
        "ledger regression/execution registry drift: "
        f"missing_execution={missing_execution}, orphan_execution={orphan_execution}"
    )


_validate_regression_registry_parity()


REGRESSION_MANIFEST: dict[str, RegressionRoute] = {
    regression_id: replace(
        metadata,
        test_case=REGRESSION_EXECUTION_ROUTES[regression_id].pytest_target,
        execution_status="executable",
    )
    for regression_id, metadata in _LEDGER_REGRESSION_METADATA.items()
}


def validate_executable_manifest_snapshot() -> None:
    """Validate semantic parity and executable projection without fixed ID/count shape."""

    validate_manifest_snapshot()
    _validate_regression_registry_parity()
    if not REGRESSION_MANIFEST:
        raise ValueError("REGRESSION_EXECUTABLE_EMPTY")
    for regression_id, route in REGRESSION_MANIFEST.items():
        if route.execution_status != "executable":
            raise ValueError(f"REGRESSION_NOT_EXECUTABLE:{regression_id}")
        expected = REGRESSION_EXECUTION_ROUTES.get(regression_id)
        if expected is None or route.test_case != expected.pytest_target:
            raise ValueError(f"REGRESSION_ROUTE_DRIFT:{regression_id}")


__all__ = [
    "ACCEPTANCE_MANIFEST",
    "FAMILY_OWNERS",
    "REGRESSION_MANIFEST",
    "DecisionReceipt",
    "LedgerTraceAudit",
    "RegressionRoute",
    "audit_ledger_text",
    "validate_decision_receipt",
    "validate_executable_manifest_snapshot",
    "validate_manifest_snapshot",
]
