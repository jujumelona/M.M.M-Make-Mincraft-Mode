from __future__ import annotations

"""Real Gradle fallback support for generation-time JDT infrastructure outages.

The generation verifier owns fallback selection directly. This module only
contains the Gradle execution/receipt helper; ``install`` is intentionally a
no-op retained for import compatibility and performs no runtime rebinding.
"""

import re
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause

_MAX_LOG_TAIL_CHARS = 16 * 1024
_MAX_LOG_TAIL_LINES = 120
_JAVA_TOOLCHAIN_PATTERNS = (
    re.compile(r"\brelease version\s+\d+\s+not supported\b", re.IGNORECASE),
    re.compile(r"\binvalid target release\s*:\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bunsupportedclassversionerror\b", re.IGNORECASE),
    re.compile(r"\bno matching toolchains found\b", re.IGNORECASE),
    re.compile(r"\bcannot find a java installation\b", re.IGNORECASE),
)
_VALIDATION_INPUT_CHANGE_PATTERNS = (
    re.compile(r"\bproject inputs changed during validation\b", re.IGNORECASE),
    re.compile(r"\bresult is not certifiable\b", re.IGNORECASE),
)


def _bounded_log_tail(path: str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.is_symlink():
            return ""
        lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = "\n".join(lines[-_MAX_LOG_TAIL_LINES:])
    if len(tail) > _MAX_LOG_TAIL_CHARS:
        tail = tail[-_MAX_LOG_TAIL_CHARS:]
    return tail


def _is_java_toolchain_failure(*parts: str | None) -> bool:
    text = "\n".join(str(part or "") for part in parts)
    return any(pattern.search(text) is not None for pattern in _JAVA_TOOLCHAIN_PATTERNS)


def _is_validation_input_change(*parts: str | None) -> bool:
    """Return true only for the host's non-certifiable input-snapshot condition."""

    text = "\n".join(str(part or "") for part in parts)
    return all(pattern.search(text) is not None for pattern in _VALIDATION_INPUT_CHANGE_PATTERNS)


def _last_gradle_log(report: Any, report_dict: dict[str, Any]) -> str:
    if report.passed:
        return ""
    commands = report_dict.get("commands")
    if not isinstance(commands, list) or not commands:
        return ""
    final_command = commands[-1]
    if not isinstance(final_command, dict):
        return ""
    return _bounded_log_tail(str(final_command.get("log_path") or ""))


def _fallback_status(report: Any, last_log: str) -> tuple[str, str | None]:
    if report.passed:
        return "PASS", None
    if (
        str(report.status).strip().upper() == "UNAVAILABLE"
        or _is_java_toolchain_failure(report.error, last_log)
    ):
        return "UNAVAILABLE", "JAVA_TOOLCHAIN_UNAVAILABLE"
    if _is_validation_input_change(report.error, last_log):
        return "UNAVAILABLE", "VALIDATION_INPUTS_CHANGED"
    return "FAIL", "GRADLE_BUILD_FAILED"


def _fallback_diagnostics(
    report: Any,
    last_log: str,
    *,
    failure_code: str | None,
) -> list[dict[str, Any]]:
    if report.passed:
        return []
    message = str(report.error or "Gradle build failed.")
    if last_log:
        message += "\n\nGradle log tail:\n" + last_log
    return [
        {
            "severity": 1,
            "source": "gradle",
            "code": failure_code or "GRADLE_BUILD_FAILED",
            "message": message,
        }
    ]


def _failure_fields(failure_code: str | None) -> dict[str, Any]:
    if failure_code == "JAVA_TOOLCHAIN_UNAVAILABLE":
        return {
            "failure_class": "environment",
            "repairable": False,
            "code": failure_code,
        }
    if failure_code == "VALIDATION_INPUTS_CHANGED":
        return {
            "failure_class": "validation_state",
            "repairable": False,
            "code": failure_code,
        }
    return {}


def _fallback_reason(failure_code: str | None) -> str:
    if failure_code == "JAVA_TOOLCHAIN_UNAVAILABLE":
        return "JDT verifier unavailable and Gradle Java toolchain unavailable"
    if failure_code == "VALIDATION_INPUTS_CHANGED":
        return "JDT verifier unavailable and Gradle validation inputs changed before certification"
    return "JDT verifier unavailable; pinned Gradle build used as host verifier"


def _gradle_fallback_receipt(
    runtime: Any,
    root: Path,
    *,
    runtime_module: Any,
    jdt_error: BaseException,
    gradle_runner_factory: Any | None = None,
) -> dict[str, Any]:
    """Run the pinned Gradle build and return a verifier-compatible receipt."""

    from .runner import GradleRunner

    cache_root = Path(runtime.workspace_root).expanduser().resolve() / ".cache" / "gradle"
    runner = (gradle_runner_factory or GradleRunner)(cache_root)
    report = runner.build(Path(root).resolve(), run_gametest=False)
    report_dict = report.to_dict()
    last_log = _last_gradle_log(report, report_dict)
    status, failure_code = _fallback_status(report, last_log)
    diagnostics = _fallback_diagnostics(
        report,
        last_log,
        failure_code=failure_code,
    )
    receipt: dict[str, Any] = {
        "status": status,
        "complete": True,
        "session_id": "gradle-fallback",
        "model_id": f"gradle:{report.gradle_version}",
        "diagnostics": diagnostics,
        "error_count": len(diagnostics),
        "verifier_backend": "gradle_build",
        "fallback_from": "java_diagnostics",
        "jdt_unavailable_reason": f"{type(jdt_error).__name__}: {jdt_error}",
        "build": report_dict,
    }
    receipt.update(_failure_fields(failure_code))
    emit_root_cause(
        "generation_verifier_gradle_fallback_result",
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result=status,
        reason=_fallback_reason(failure_code),
        details={"result": receipt},
    )
    return runtime_module._bounded_result(receipt)


def install() -> None:
    """Compatibility hook; fallback dispatch is owned by the verifier itself."""

    return None


__all__ = ["install"]
