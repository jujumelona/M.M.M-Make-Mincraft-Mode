from __future__ import annotations

"""Real Gradle fallback support for generation-time JDT infrastructure outages.

The generation verifier owns fallback selection directly. This module only
contains the Gradle execution/receipt helper; ``install`` is intentionally a
no-op retained for import compatibility and performs no runtime rebinding.
"""

import re
from collections import deque
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
_FALLBACK_REASONS = {
    "JAVA_TOOLCHAIN_UNAVAILABLE": "JDT verifier unavailable and Gradle Java toolchain unavailable",
    "VALIDATION_INPUTS_CHANGED": (
        "JDT verifier unavailable and Gradle validation inputs changed before certification"
    ),
}
_DEFAULT_FALLBACK_REASON = "JDT verifier unavailable; pinned Gradle build used as host verifier"


def _bounded_log_tail(path: str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.is_symlink():
            return ""
        tail_lines: deque[str] = deque(maxlen=_MAX_LOG_TAIL_LINES)
        compiler_lines: list[str] = []
        compiler_chars = 0
        context_remaining = 0
        # Keep the first compiler errors even when --stacktrace pushes them out
        # of the tail. Bound memory as well as the returned model observation.
        with candidate.open(encoding="utf-8", errors="replace") as stream:
            while chunk := stream.readline(_MAX_LOG_TAIL_CHARS):
                line = chunk.rstrip("\r\n")
                tail_lines.append(line)
                if re.search(r"\.java:\d+:\s*(?:error|warning):", line):
                    context_remaining = 4
                if context_remaining:
                    remaining = _MAX_LOG_TAIL_CHARS // 2 - compiler_chars
                    if remaining > 1:
                        selected = line[:remaining - 1]
                        compiler_lines.append(selected)
                        compiler_chars += len(selected) + 1
                    context_remaining -= 1
    except OSError:
        return ""
    tail = "\n".join(tail_lines)
    if not compiler_lines:
        return tail[-_MAX_LOG_TAIL_CHARS:]
    evidence = "Compiler diagnostics:\n" + "\n".join(compiler_lines) + "\n\nLog tail:\n"
    return evidence + tail[-(_MAX_LOG_TAIL_CHARS - len(evidence)):]


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


def _is_gradle_environment_unavailable(report: Any, last_log: str) -> bool:
    return (
        str(report.status).strip().upper() == "UNAVAILABLE"
        or _is_java_toolchain_failure(report.error, last_log)
    )


def _fallback_status(report: Any, last_log: str) -> tuple[str, str | None]:
    if report.passed:
        return "PASS", None
    if _is_gradle_environment_unavailable(report, last_log):
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
    return _FALLBACK_REASONS.get(failure_code, _DEFAULT_FALLBACK_REASON)


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

    return


__all__ = ["install"]
