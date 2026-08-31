from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from minecraft_mod_ai.diagnostics import (
    DiagnosticCollector,
    FailureCategory,
    FailureEvent,
    FailureStatus,
    render_failure_summary,
)

if __package__:
    from .audit_stream_redactor import StreamingRedactor
else:
    from audit_stream_redactor import StreamingRedactor

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
REPORT_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.json"
RUNNER_LOG_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.runner.log"
_VALID_CHECK_STATUSES = frozenset({"PASS", "WARN", "FAIL", "SKIP"})
_INTERNAL_CHECK_CATEGORY = "audit-internal"
_OUTPUT_CHUNK_CHARS = 64 * 1024
_DEFAULT_AUDIT_TIMEOUT_SECONDS = 45 * 60


def _environment_secret_values() -> tuple[str, ...]:
    values: list[str] = []
    for name, secret in os.environ.items():
        upper = name.upper()
        if secret and any(
            marker in upper
            for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "COOKIE")
        ):
            values.append(secret)
    return tuple(dict.fromkeys(values))


def _redact_stream(source: Path, destination: Path, *, secrets: Iterable[str] | None = None) -> None:
    secret_values = _environment_secret_values() if secrets is None else tuple(secrets)
    redactor = StreamingRedactor(secret_values)
    with source.open("r", encoding="utf-8", errors="replace") as source_handle, destination.open(
        "w", encoding="utf-8", errors="replace"
    ) as destination_handle:
        while True:
            chunk = source_handle.read(_OUTPUT_CHUNK_CHARS)
            if not chunk:
                break
            safe = redactor.feed(chunk)
            if safe:
                destination_handle.write(safe)
        final = redactor.finish()
        if final:
            destination_handle.write(final)


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


def _summary_count(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"audit summary {key} must be a non-negative integer")
    return value


def validate_report_consistency(report: dict[str, Any]) -> None:
    """Fail closed when summary/index fields disagree with the actual checks."""

    checks = report.get("checks")
    summary = report.get("summary")
    if not isinstance(checks, list):
        raise ValueError("audit report checks must be a list")
    if not isinstance(summary, dict):
        raise ValueError("audit report summary must be an object")

    statuses = [str(check.get("status", "")) for check in checks if isinstance(check, dict)]
    if len(statuses) != len(checks):
        raise ValueError("audit report contains a non-object check")
    expected = {
        "total": len(checks),
        "passed": statuses.count("PASS"),
        "warned": statuses.count("WARN"),
        "failed": statuses.count("FAIL"),
        "skipped": statuses.count("SKIP"),
    }
    observed = {key: _summary_count(summary, key) for key in expected}
    if observed != expected:
        raise ValueError(
            "audit summary/check mismatch: "
            f"expected={expected!r} observed={observed!r}"
        )

    expected_lists = {
        "failed_checks": [
            str(check.get("name") or "unknown-check")
            for check in checks
            if check.get("status") == "FAIL"
        ],
        "warning_checks": [
            str(check.get("name") or "unknown-check")
            for check in checks
            if check.get("status") == "WARN"
        ],
        "skipped_checks": [
            str(check.get("name") or "unknown-check")
            for check in checks
            if check.get("status") == "SKIP"
        ],
    }
    for key, expected_names in expected_lists.items():
        value = report.get(key)
        if not isinstance(value, list) or [str(item) for item in value] != expected_names:
            raise ValueError(f"audit {key} does not match check statuses")

    expected_overall = (
        "failed"
        if expected["failed"]
        else "warning"
        if expected["warned"]
        else "passed"
    )
    if report.get("overall_status") != expected_overall:
        raise ValueError(
            "audit overall_status/check mismatch: "
            f"expected={expected_overall!r} observed={report.get('overall_status')!r}"
        )


def failure_groups_from_report(report: dict[str, Any]):
    collector = DiagnosticCollector()
    checks = report.get("checks")
    if not isinstance(checks, list):
        checks = []
    for check in checks:
        if not isinstance(check, dict) or str(check.get("status", "")).upper() != "FAIL":
            continue
        category = (
            FailureCategory.INTERNAL
            if str(check.get("category") or "") == _INTERNAL_CHECK_CATEGORY
            else FailureCategory.VALIDATION
        )
        collector.record(
            FailureEvent(
                stage=f"full-debug:{str(check.get('category') or 'unknown')}",
                operation=str(check.get("name") or "unknown-check"),
                category=category,
                cause_type="AuditInternalFailure" if category is FailureCategory.INTERNAL else "AuditCheckFailure",
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


def _raw_log_fallback() -> str:
    try:
        relative = RUNNER_LOG_PATH.relative_to(ROOT)
    except ValueError:
        return f"redacted runner output target={RUNNER_LOG_PATH}"
    return f"redacted runner output target={relative}"


def _render_internal_report_error(exc: BaseException, operation: str) -> str:
    collector = DiagnosticCollector()
    collector.record_exception(
        exc,
        stage="full-debug:audit-runner",
        operation=operation,
        category=FailureCategory.INTERNAL,
        retryable=False,
        final_status=FailureStatus.FAILED,
        fallback=_raw_log_fallback(),
    )
    return render_failure_summary(collector.groups())


def _render_runner_event(
    cause_type: str,
    cause: str,
    operation: str,
    *,
    category: FailureCategory = FailureCategory.INTERNAL,
) -> str:
    collector = DiagnosticCollector()
    collector.record(
        FailureEvent(
            stage="full-debug:audit-runner",
            operation=operation,
            category=category,
            cause_type=cause_type,
            cause=cause,
            retryable=category is FailureCategory.TRANSIENT,
            final_status=FailureStatus.FAILED,
            fallback=_raw_log_fallback(),
        )
    )
    return render_failure_summary(collector.groups())


def _prepare_audit_directory() -> bool:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(_render_internal_report_error(exc, "prepare audit directory"))
        return False
    return True


def _atomic_write_report(report: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=REPORT_PATH.parent,
            prefix=f".{REPORT_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary_path.replace(REPORT_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _remove_stale_audit_artifacts() -> bool:
    for path, operation in (
        (RUNNER_LOG_PATH, "remove stale audit runner log"),
        (REPORT_PATH, "remove stale audit report"),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(_render_internal_report_error(exc, operation))
            return False
    return True


def _run_audit(*, timeout_seconds: int) -> int | None:
    if timeout_seconds <= 0:
        print(
            _render_runner_event(
                "InvalidAuditTimeout",
                f"timeout_seconds must be positive, got {timeout_seconds}",
                "validate audit timeout",
                category=FailureCategory.INPUT,
            )
        )
        return None
    if not _remove_stale_audit_artifacts():
        return None

    temporary_path: Path | None = None
    returncode: int | None = None
    launch_failure: BaseException | None = None
    cleanup_failure: OSError | None = None
    try:
        RUNNER_LOG_PATH.write_text("", encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="replace",
            dir=AUDIT_DIR,
            prefix=".audit-runner-output-",
            suffix=".log",
            delete=False,
        ) as raw_handle:
            temporary_path = Path(raw_handle.name)
            try:
                process = subprocess.run(
                    [sys.executable, str(ROOT / "tools/full_project_audit.py")],
                    cwd=ROOT,
                    stdout=raw_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout_seconds,
                )
                returncode = process.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                launch_failure = exc
        _redact_stream(temporary_path, RUNNER_LOG_PATH)
    except OSError as exc:
        print(_render_internal_report_error(exc, "preserve redacted audit runner output"))
        return None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_failure = exc

    if cleanup_failure is not None:
        print(_render_internal_report_error(cleanup_failure, "remove temporary audit runner output"))
        return None
    if launch_failure is None:
        return returncode
    if isinstance(launch_failure, subprocess.TimeoutExpired):
        print(
            _render_runner_event(
                "AuditTimeout",
                f"full project audit exceeded timeout={timeout_seconds}s",
                "run full project audit",
                category=FailureCategory.TRANSIENT,
            )
        )
        return None
    print(_render_internal_report_error(launch_failure, "launch audit runner"))
    return None


def main() -> int:
    if not _prepare_audit_directory():
        return 1
    process_returncode = _run_audit(timeout_seconds=_DEFAULT_AUDIT_TIMEOUT_SECONDS)
    if process_returncode is None:
        return 1

    if not REPORT_PATH.is_file():
        print(
            _render_runner_event(
                "MissingAuditReport",
                f"{REPORT_PATH} was not produced by the current audit run",
                "load audit report",
            )
        )
        return process_returncode or 1

    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(_render_internal_report_error(exc, "parse audit report"))
        return process_returncode or 1
    if not isinstance(report, dict):
        print(
            _render_runner_event(
                "InvalidAuditReport",
                "top-level report is not an object",
                "parse audit report",
            )
        )
        return process_returncode or 1

    try:
        report = normalize_report_semantics(report)
        validate_report_consistency(report)
        _atomic_write_report(report)
    except (OSError, ValueError) as exc:
        print(_render_internal_report_error(exc, "validate audit report"))
        return process_returncode or 1

    groups = failure_groups_from_report(report)
    expected_returncode = 1 if groups else 0
    if process_returncode != expected_returncode:
        print(render_report_summary(report))
        print()
        print(
            _render_runner_event(
                "AuditExitMismatch",
                (
                    f"audit process exit={process_returncode}; "
                    f"report requires exit={expected_returncode}"
                ),
                "validate audit process/report agreement",
            )
        )
        return process_returncode or 1

    print(render_report_summary(report))
    return process_returncode


if __name__ == "__main__":
    raise SystemExit(main())
