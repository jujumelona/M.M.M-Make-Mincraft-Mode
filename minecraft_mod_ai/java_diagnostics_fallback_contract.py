from __future__ import annotations

"""Fail-closed Gradle fallback for an unavailable JDT diagnostic service.

JDT remains the fast source diagnostic.  A JDT transport/startup failure is not a
source-code failure, so the verifier falls back to the repository's authoritative
Gradle build instead of terminating the generation loop solely because the language
server is unavailable.  The fallback never converts a failed build into PASS.
"""

from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause
from .runner import BuildRunnerError, GradleRunner
from .validation_diagnostic_contract import diagnostic_errors

_INSTALLED_ATTR = "_mmm_java_diagnostics_gradle_fallback_installed"


def _jdt_unavailable(receipt: Any) -> bool:
    if not isinstance(receipt, Mapping):
        return True
    return any(
        str(item.get("code") or "") == "JDT_DIAGNOSTICS_UNAVAILABLE"
        for item in diagnostic_errors(receipt)
    )


def _gradle_diagnostic_receipt(report: Any) -> dict[str, Any]:
    details = report.to_dict()
    if report.passed:
        return {
            "status": "PASS",
            "diagnostics": {},
            "verifier": "gradle_build_fallback",
            "fallback_from": "jdt",
            "build_report": details,
        }
    return {
        "status": "PASS",
        "diagnostics": [
            {
                "severity": 1,
                "source": "gradle",
                "code": "GRADLE_BUILD_FAILED",
                "message": str(report.error or "Gradle build failed."),
            }
        ],
        "verifier": "gradle_build_fallback",
        "fallback_from": "jdt",
        "build_report": details,
    }


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
        receipt = original(
            self,
            project_root,
            relative_files=relative_files,
            timeout_seconds=timeout_seconds,
        )
        if not _jdt_unavailable(receipt):
            return receipt

        root = self._existing_dir(project_root)
        emit_root_cause(
            "java_diagnostics_fallback_start",
            stage="verify",
            operation="java_diagnostics",
            gate="verifier_fallback",
            result="START",
            reason="JDT diagnostics unavailable; running authoritative Gradle build",
            details={"project_root": str(root)},
        )
        try:
            runner = GradleRunner(
                Path(self.workspace_root) / ".cache" / "gradle",
                command_timeout_seconds=max(60, int(timeout_seconds)),
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
                details={"project_root": str(root)},
                exc=exc,
            )
            unavailable = dict(receipt) if isinstance(receipt, Mapping) else {
                "status": "UNAVAILABLE",
                "diagnostics": {},
            }
            unavailable["gradle_fallback_error"] = f"{type(exc).__name__}: {exc}"
            return unavailable

        result = _gradle_diagnostic_receipt(report)
        emit_root_cause(
            "java_diagnostics_fallback_result",
            stage="verify",
            operation="java_diagnostics",
            gate="verifier_fallback",
            result="PASS" if report.passed else "FAIL",
            reason=(
                "Gradle fallback verified the project"
                if report.passed
                else str(report.error or "Gradle fallback rejected the project")
            ),
            details={"project_root": str(root), "build_report": report.to_dict()},
        )
        return result

    service_type.java_diagnostics = java_diagnostics
    setattr(service_type, _INSTALLED_ATTR, True)


__all__ = ["install"]
