from __future__ import annotations

"""Prevent stale validation receipts from triggering an unrelated execution replay."""

import json
import sys
from collections.abc import Mapping
from typing import Any


def _checkpoint_for_exception(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    message = str(exc).casefold()
    if "failed deterministic validation" in message:
        return "validate-source"
    if "jdt reported errors" in message:
        return "validate-jdt"
    if "gradle/gametest failed after the repair loop" in message:
        return "gradle-build"
    # Artifact/runtime/visual/publication failures must never resurrect an older
    # source/JDT/build failure merely because its receipt remains in the ledger.
    return None


def _active_exception() -> BaseException | None:
    # sys.exception() is Python 3.11+. MMM still supports Python 3.10, where the
    # exception currently handled by the calling thread is available via exc_info().
    getter = getattr(sys, "exception", None)
    if callable(getter):
        return getter()
    return sys.exc_info()[1]


def install(feedback_module: Any) -> None:
    current = feedback_module._latest_failed_feedback
    if getattr(current, "_mmm_current_exception_scoped", False):
        return

    def latest_failed_feedback(ledger: Any) -> dict[str, Any] | None:
        checkpoint_id = _checkpoint_for_exception(_active_exception())
        if checkpoint_id is None:
            return None
        with ledger._connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_id, receipt_json, state, updated_at
                FROM checkpoints
                WHERE checkpoint_id = ? AND receipt_json IS NOT NULL
                LIMIT 1
                """,
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            return None
        raw_id, receipt_json, state, updated_at = row
        try:
            receipt = json.loads(receipt_json)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(receipt, Mapping):
            return None
        if not feedback_module._validation_failed(str(raw_id), receipt):
            return None
        diagnostics = feedback_module._diagnostics_from_value(receipt)
        return {
            "schema_version": "mmm/execution-validation-feedback-v1",
            "checkpoint_id": str(raw_id),
            "checkpoint_state": str(state),
            "checkpoint_updated_at": float(updated_at),
            "diagnostics": diagnostics,
            "diagnostic_fingerprint": feedback_module._sha(diagnostics),
            "failure_scope": "current_exception",
        }

    latest_failed_feedback._mmm_current_exception_scoped = True
    latest_failed_feedback.__wrapped__ = current
    feedback_module._latest_failed_feedback = latest_failed_feedback


__all__ = ["_active_exception", "_checkpoint_for_exception", "install"]
