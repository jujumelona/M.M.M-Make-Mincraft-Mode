from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from minecraft_mod_ai.diagnostics import (
    DiagnosticCollector,
    FailureCategory,
    FailureEvent,
    FailureStatus,
    render_failure_summary,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
REPORT_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.json"
RUNNER_LOG_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.runner.log"
_VALID_CHECK_STATUSES = frozenset({"PASS", "WARN", "FAIL", "SKIP"})


def normalize_report_semantics(report: dict[str, Any]) -> dict[str, Any]:
    """Make per-check success semantics explicit in the persisted Debug artifact."""

    checks = report.get("checks")
    if not isinstance(checks, list):
        raise ValueError("audit report checks must be a list")
    normalized = dict(report)
    normalized_checks: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError(f"audit check {index} must be an object")
        item = dict(check)
        status = str(item.get("status", "")).upper()
        if status not in _VALID_CHECK_STATUSES:
            raise ValueError(
                f"audit check {index} has unsupported status: {status or 'missing'}"
            )
        item["status"] = status
        item["passed"] = status == "PASS"
        item["blocking_failure"] = status == "FAIL"
        item["non_blocking"] = status in {"WARN", "SKIP"}
        normalized_checks.append(item)
    normalized["checks"] = normalized_checks
    normalized["status_semantics"] = {
        "PASS": "passed=true; successful check",
        "WARN": "passed=false; non_blocking=true",
        "SKIP": "passed=false; non_blocking=true",
        "FAIL": "passed=false; blocking_failure=true",
    }
    return normalized


def failure_groups_from_report(report: dict[str, Any]):
    collector = DiagnosticCollector()
    checks = report.get("checks")
    if not isinstance(checks, list):
        checks = []
    for check in checks:
        if not isinstance(check, dict) or str(check.get("status", "")).upper() != "FAIL":
            continue
        collector.record(
            FailureEvent(
                stage=f"full-debug:{str(check.get('category') or 'unknown')}",
                operation=str(check.get("name") or "unknown-check"),
                category=FailureCategory.VALIDATION,
                cause_type="AuditCheckFailure",
                cause=str(check.get("detail") or "check failed"),
                retryable=False,
                final_status=FailureStatus.FAILED,
            )
        )
    return collector.groups()


def render_report_summary(report: dict[str, Any]) -> str:
    groups = failure_groups_from_report(report)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    warned = int(summary.get("warned") or 0)
    skipped = int(summary.get("skipped") or 0)
    passed = int(summary.get("passed") or 0)
    failed = int(summary.get("failed") or 0)
    counts = f"CHECKS pass={passed} warn={warned} fail={failed} skip={skipped}"
    if groups:
        return render_failure_summary(groups) + "\n" + counts
    if warned:
        return "FINAL STATUS\nWARN\n" + counts
    return "FINAL STATUS\nPASS\n" + counts


def _render_internal_report_error(exc: BaseException, operation: str) -> str:
    collector = DiagnosticCollector()
    collector.record_exception(
        exc,
        stage="full-debug:audit-runner",
        operation=operation,
        category=FailureCategory.INTERNAL,
        retryable=False,
        final_status=FailureStatus.FAILED,
        fallback=f"raw runner output preserved at {RUNNER_LOG_PATH.relative_to(ROOT)}",
    )
    return render_failure_summary(collector.groups())


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [sys.executable, str(ROOT / "tools/full_project_audit.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    RUNNER_LOG_PATH.write_text(process.stdout or "", encoding="utf-8")
    if not REPORT_PATH.is_file():
        collector = DiagnosticCollector()
        collector.record(
            FailureEvent(
                stage="full-debug:audit-runner",
                operation="load audit report",
                category=FailureCategory.INTERNAL,
                cause_type="MissingAuditReport",
                cause=f"{REPORT_PATH.relative_to(ROOT)} was not produced",
                retryable=False,
                final_status=FailureStatus.FAILED,
                fallback=f"raw runner output preserved at {RUNNER_LOG_PATH.relative_to(ROOT)}",
            )
        )
        print(render_failure_summary(collector.groups()))
        return process.returncode or 1
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(_render_internal_report_error(exc, "parse audit report"))
        return process.returncode or 1
    if not isinstance(report, dict):
        print(
            "ROOT FAILURE 1\n"
            "full-debug:audit-runner / parse audit report [INTERNAL]\n"
            "CAUSE\nInvalidAuditReport: top-level report is not an object\n"
            "ATTEMPTS\n1\nFALLBACK\nraw runner output preserved\n"
            "FINAL STATUS\nFAILED"
        )
        return process.returncode or 1
    try:
        report = normalize_report_semantics(report)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(_render_internal_report_error(exc, "normalize audit report"))
        return process.returncode or 1
    print(render_report_summary(report))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
