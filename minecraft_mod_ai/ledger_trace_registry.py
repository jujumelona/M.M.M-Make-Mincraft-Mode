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


if set(_LEDGER_REGRESSION_METADATA) != set(REGRESSION_EXECUTION_ROUTES):
    missing_execution = sorted(
        set(_LEDGER_REGRESSION_METADATA) - set(REGRESSION_EXECUTION_ROUTES)
    )
    orphan_execution = sorted(
        set(REGRESSION_EXECUTION_ROUTES) - set(_LEDGER_REGRESSION_METADATA)
    )
    raise RuntimeError(
        "ledger regression/execution registry drift: "
        f"missing_execution={missing_execution}, orphan_execution={orphan_execution}"
    )


REGRESSION_MANIFEST: dict[str, RegressionRoute] = {
    regression_id: replace(
        metadata,
        test_case=REGRESSION_EXECUTION_ROUTES[regression_id].pytest_target,
        execution_status="executable",
    )
    for regression_id, metadata in _LEDGER_REGRESSION_METADATA.items()
}


def validate_executable_manifest_snapshot() -> None:
    """Validate ID parity and executable-state projection without claiming ACC PASS."""

    validate_manifest_snapshot()
    if len(REGRESSION_MANIFEST) != 39:
        raise ValueError("REGRESSION_EXECUTABLE_COUNT")
    for regression_id, route in REGRESSION_MANIFEST.items():
        if route.execution_status != "executable":
            raise ValueError(f"REGRESSION_NOT_EXECUTABLE:{regression_id}")
        if route.test_case != REGRESSION_EXECUTION_ROUTES[regression_id].pytest_target:
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
