from __future__ import annotations

"""Post-build validation gates for complete production."""

from collections.abc import Callable
from typing import Any

from .complete_orchestrator_support import CompleteProductionError
from .validation_diagnostic_contract import (
    diagnostic_errors,
    unwrap_diagnostic_receipt,
)

_JDT_INFRASTRUCTURE_CODES = frozenset({
    "JDT_DIAGNOSTICS_UNAVAILABLE",
    "JDT_WORKSPACE_NOT_READY",
})


def unchanged_postbuild_validation(
    source_report: dict[str, Any],
    jdt_receipt: dict[str, Any] | None,
    validate_jdt: Callable[[], dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    needs_postbuild_jdt = jdt_receipt is None
    if jdt_receipt is not None:
        normalized, _path = unwrap_diagnostic_receipt(jdt_receipt)
        needs_postbuild_jdt = (
            str(normalized.get("status") or "").strip().upper()
            == "DEFERRED_TO_POST_BUILD"
        )
    if validate_jdt is not None and needs_postbuild_jdt:
        return source_report, validate_jdt(), True
    return source_report, jdt_receipt, False


def blocking_jdt_errors(
    receipt: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return source diagnostics actionable after build verification."""

    return [
        item
        for item in diagnostic_errors(receipt)
        if str(item.get("code") or "").strip().upper()
        not in _JDT_INFRASTRUCTURE_CODES
    ]


def jdt_release_evidence_passed(receipt: dict[str, Any] | None) -> bool:
    """Require one real, clean JDT receipt for release authority."""

    if receipt is None:
        return False
    normalized, _path = unwrap_diagnostic_receipt(receipt)
    if not normalized:
        return False
    if diagnostic_errors(receipt):
        return False
    status = str(normalized.get("status") or "").strip().upper()
    if status == "DEFERRED_TO_POST_BUILD":
        return False
    try:
        error_count = int(normalized.get("error_count", -1))
        files_opened = int(normalized.get("files_opened", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    return error_count == 0 and files_opened > 0


def requested_verification_failures(
    *,
    run_jdt: bool,
    jdt_receipt: dict[str, Any] | None,
) -> list[str]:
    """Keep explicitly requested verifier work blocking when unavailable."""

    if run_jdt and not jdt_release_evidence_passed(jdt_receipt):
        return ["execution-gate:jdt:missing-jdt"]
    return []


def final_validation_failure(
    *,
    source_report: dict[str, Any],
    jdt_receipt: dict[str, Any] | None,
    run_jdt: bool,
) -> str | None:
    if source_report.get("status") != "PASS":
        return "Final project failed deterministic source validation after build/repair."
    if run_jdt and jdt_receipt is not None and blocking_jdt_errors(jdt_receipt):
        return "Final JDT validation still reports source errors after build/repair."
    return None


def refresh_validation_after_build(
    *,
    prebuild_manifest: str,
    final_manifest: str,
    source_report: dict[str, Any],
    jdt_receipt: dict[str, Any] | None,
    validate_source: Callable[[], dict[str, Any]],
    validate_jdt: Callable[[], dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Run JDT after compile; refresh source validation only when the tree changed."""

    if final_manifest == prebuild_manifest:
        return unchanged_postbuild_validation(
            source_report,
            jdt_receipt,
            validate_jdt,
        )

    from .deadline_executor import iter_completed_with_deadlines

    validation_jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("source", validate_source),
    ]
    if validate_jdt is not None:
        validation_jobs.append(("jdt", validate_jdt))

    validation_results: dict[str, dict[str, Any]] = {}
    for job, result in iter_completed_with_deadlines(
        validation_jobs,
        lambda item: item[1](),
        max_workers=len(validation_jobs),
        stage="complete_post_build_validation",
        sort_key=lambda item: item[0],
    ):
        validation_results[job[0]] = result

    refreshed_source = validation_results["source"]
    if refreshed_source.get("status") != "PASS":
        raise CompleteProductionError(
            "Final repaired project failed deterministic validation."
        )
    refreshed_jdt = validation_results.get("jdt")
    return refreshed_source, refreshed_jdt, True


__all__ = [
    "blocking_jdt_errors",
    "final_validation_failure",
    "jdt_release_evidence_passed",
    "refresh_validation_after_build",
    "requested_verification_failures",
    "unchanged_postbuild_validation",
]
