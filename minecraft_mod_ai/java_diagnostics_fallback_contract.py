from __future__ import annotations

"""Fail-closed Gradle fallback for an unavailable JDT diagnostic service.

JDT remains the fast source diagnostic. A JDT transport/startup/deadline failure is
not a source-code failure, so the verifier falls back to the repository's
authoritative Gradle build. Gradle compile failures stay FAIL and carry the command
log needed by the repair loop.
"""

from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause
from .runner import BuildRunnerError, GradleRunner
from .validation_diagnostic_contract import diagnostic_errors

_INSTALLED_ATTR = "_mmm_java_diagnostics_gradle_fallback_installed"
_GRADLE_FALLBACK_TIMEOUT_SECONDS = 300
_MAX_GRADLE_LOG_CHARS = 16_000

_JDT_UNAVAILABLE_MARKERS = (
    "tool_runtime_unavailable",
    "verifier_unavailable",
    "jdt diagnostics unavailable",
    "jdt ls did not publish diagnostics",
    "validation deadline",
    "jdt_incomplete",
    "jdt_no_diagnostics",
)


def _jdt_unavailable(receipt: Any) -> bool:
    if not isinstance(receipt, Mapping):
        return True
    return any(
        str(item.get("code") or "") == "JDT_DIAGNOSTICS_UNAVAILABLE"
        for item in diagnostic_errors(receipt)
    )


def _jdt_unavailable_exception(exc: BaseException) -> bool:
    """Classify only JDT/runtime availability failures for fallback.

    The production tool boundary may wrap JDT exceptions in transport-specific
    exception classes, so inspect stable structured attributes and reviewed message
    markers instead of depending on one concrete exception type.
    """

    if isinstance(exc, TimeoutError):
        return True

    parts = [type(exc).__name__, str(exc)]
    for name in ("code", "reason", "status"):
        value = getattr(exc, name, None)
        if value is not None:
            parts.append(str(value))
    text = " ".join(parts).strip().lower()

    if any(marker in text for marker in _JDT_UNAVAILABLE_MARKERS):
        return True

    unavailable_terms = ("unavailable", "timeout", "timed out", "deadline")
    if "jdt" in text and any(term in text for term in unavailable_terms):
        return True
    if "java diagnostics" in text and any(term in text for term in unavailable_terms):
        return True
    return False


def _command_to_dict(command: Any) -> dict[str, Any]:
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(command):
            return dict(asdict(command))
    except (TypeError, ValueError):
        pass
    return {
        "name": str(getattr(command, "name", "")),
        "command": list(getattr(command, "command", ()) or ()),
        "exit_code": getattr(command, "exit_code", None),
        "duration_seconds": getattr(command, "duration_seconds", None),
        "log_path": str(getattr(command, "log_path", "") or ""),
        "timed_out": bool(getattr(command, "timed_out", False)),
    }


def _gradle_log_tail(report: Any) -> str:
    commands = list(getattr(report, "commands", ()) or ())
    if not commands:
        return ""

    failed = [
        command
        for command in commands
        if bool(getattr(command, "timed_out", False))
        or getattr(command, "exit_code", 0) not in (0, None)
    ]
    candidates = failed or commands
    for command in reversed(candidates):
        log_path = getattr(command, "log_path", None)
        if not log_path:
            continue
        try:
            text = Path(str(log_path)).read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
        except (OSError, TypeError, ValueError):
            continue
        if not text:
            continue
        if len(text) > _MAX_GRADLE_LOG_CHARS:
            return "...[Gradle log truncated]...\n" + text[-_MAX_GRADLE_LOG_CHARS:]
        return text
    return ""


def _gradle_message(report: Any) -> str:
    error = str(getattr(report, "error", "") or "").strip()
    log_tail = _gradle_log_tail(report)
    if error and log_tail and error not in log_tail:
        return f"{error}\n\n{log_tail}"
    return log_tail or error or "Gradle validation failed."


def _gradle_timed_out(report: Any) -> bool:
    return any(
        bool(getattr(command, "timed_out", False))
        for command in (getattr(report, "commands", ()) or ())
    )


def _gradle_diagnostic_receipt(
    report: Any,
    *,
    jdt_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    details = report.to_dict()
    common: dict[str, Any] = {
        "verifier": "gradle_build_fallback",
        "fallback_from": "jdt",
        "build_report": details,
    }
    if jdt_unavailable_reason:
        common["jdt_unavailable_reason"] = jdt_unavailable_reason

    if report.passed:
        return {
            "status": "PASS",
            "diagnostics": {},
            **common,
        }

    timed_out = _gradle_timed_out(report)
    return {
        "status": "TIMEOUT" if timed_out else "FAIL",
        "diagnostics": [
            {
                "severity": 1,
                "source": "gradle",
                "code": (
                    "GRADLE_BUILD_TIMEOUT"
                    if timed_out
                    else "GRADLE_BUILD_FAILED"
                ),
                "message": _gradle_message(report),
            }
        ],
        **common,
    }


def _unavailable_receipt(
    *,
    exc: BaseException,
    jdt_unavailable_reason: str | None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "diagnostics": {},
        "verifier": "gradle_build_fallback",
        "fallback_from": "jdt",
        "gradle_fallback_error": f"{type(exc).__name__}: {exc}",
    }
    if jdt_unavailable_reason:
        receipt["jdt_unavailable_reason"] = jdt_unavailable_reason
    return receipt


def install(service_type: type[Any]) -> None:
    """Install once on ``ProductionToolService`` without changing its public schema."""

    if bool(getattr(service_type, _INSTALLED_ATTR, False)):
        return
    original = service_type.java_diagnostics

    @wraps(original)
    def java_diagnostics(
        self: Any,
        project_root: str,
        relative_files: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        jdt_unavailable_reason: str | None = None
        try:
            receipt = original(
                self,
                project_root,
                relative_files=relative_files,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            if not _jdt_unavailable_exception(exc):
                raise
            receipt = None
            jdt_unavailable_reason = f"{type(exc).__name__}: {exc}"
        else:
            if not _jdt_unavailable(receipt):
                return receipt
            if isinstance(receipt, Mapping):
                jdt_unavailable_reason = str(
                    receipt.get("error")
                    or receipt.get("reason")
                    or "JDT diagnostics unavailable"
                )

        root = self._existing_dir(project_root)
        emit_root_cause(
            "java_diagnostics_fallback_start",
            stage="verify",
            operation="java_diagnostics",
            gate="verifier_fallback",
            result="START",
            reason="JDT diagnostics unavailable; running authoritative Gradle build",
            details={
                "project_root": str(root),
                "jdt_unavailable_reason": jdt_unavailable_reason,
                "jdt_timeout_seconds": timeout_seconds,
                "gradle_timeout_seconds": _GRADLE_FALLBACK_TIMEOUT_SECONDS,
            },
        )
        try:
            runner = GradleRunner(
                Path(self.workspace_root) / ".cache" / "gradle",
                command_timeout_seconds=_GRADLE_FALLBACK_TIMEOUT_SECONDS,
            )
            report = runner.build(root, run_gametest=False)
        except (BuildRunnerError, OSError, TimeoutError) as exc:
            emit_root_cause(
                "java_diagnostics_fallback_result",
                stage="verify",
                operation="java_diagnostics",
                gate="verifier_fallback",
                result="UNAVAILABLE",
                reason=f"{type(exc).__name__}: {exc}",
                details={
                    "project_root": str(root),
                    "jdt_unavailable_reason": jdt_unavailable_reason,
                },
                exc=exc,
            )
            return _unavailable_receipt(
                exc=exc,
                jdt_unavailable_reason=jdt_unavailable_reason,
            )

        result = _gradle_diagnostic_receipt(
            report,
            jdt_unavailable_reason=jdt_unavailable_reason,
        )
        emit_root_cause(
            "java_diagnostics_fallback_result",
            stage="verify",
            operation="java_diagnostics",
            gate="verifier_fallback",
            result=result["status"],
            reason=(
                "Gradle fallback verified the project"
                if report.passed
                else _gradle_message(report)
            ),
            details={
                "project_root": str(root),
                "build_report": report.to_dict(),
                "jdt_unavailable_reason": jdt_unavailable_reason,
            },
        )
        return result

    service_type.java_diagnostics = java_diagnostics
    setattr(service_type, _INSTALLED_ATTR, True)


__all__ = ["install"]
