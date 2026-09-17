from __future__ import annotations

"""Real Gradle verification support for generation-time JDT uncertainty.

The canonical generation verifier calls this module directly. It intentionally performs
no runtime method rebinding; the generation verifier already owns fallback/corroboration
selection.
"""

import re
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

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


def _collect_log_evidence(stream: TextIO) -> tuple[deque[str], list[str]]:
    tail_lines: deque[str] = deque(maxlen=_MAX_LOG_TAIL_LINES)
    compiler_lines: list[str] = []
    compiler_chars = 0
    context_remaining = 0
    while chunk := stream.readline(_MAX_LOG_TAIL_CHARS):
        line = chunk.rstrip("\r\n")
        tail_lines.append(line)
        if re.search(r"\.java:\d+:\s*(?:error|warning):", line):
            context_remaining = 4
        if not context_remaining:
            continue
        remaining = _MAX_LOG_TAIL_CHARS // 2 - compiler_chars
        if remaining > 1:
            selected = line[: remaining - 1]
            compiler_lines.append(selected)
            compiler_chars += len(selected) + 1
        context_remaining -= 1
    return tail_lines, compiler_lines


def _bounded_log_tail(path: str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.is_symlink():
            return ""
        with candidate.open(encoding="utf-8", errors="replace") as stream:
            tail_lines, compiler_lines = _collect_log_evidence(stream)
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


def _structured_fallback_receipt(
    receipt: Mapping[str, Any],
    *,
    runtime_module: Any,
) -> dict[str, Any]:
    """Keep fallback verifier semantics structured for host adjudication."""

    sanitized = runtime_module._sanitize_observation(receipt)
    if not isinstance(sanitized, Mapping):
        raise runtime_module.AgentToolRuntimeError(
            "Gradle verification returned a non-mapping diagnostic receipt"
        )
    payload = dict(sanitized)
    payload["_mmm_observation"] = {
        "trust": "untrusted_data_only",
        "sanitized": True,
        "truncated": False,
    }
    return payload


def _fallback_variant_fields(corroboration: bool, reason_text: str) -> dict[str, Any]:
    if corroboration:
        return {
            "session_id": "gradle-corroboration",
            "corroboration_from": "java_diagnostics",
            "jdt_corroboration_reason": reason_text,
        }
    return {
        "session_id": "gradle-fallback",
        "fallback_from": "java_diagnostics",
        "jdt_unavailable_reason": reason_text,
    }


def _fallback_event(corroboration: bool) -> tuple[str, str]:
    if corroboration:
        return (
            "generation_verifier_gradle_corroboration_result",
            "JDT dependency-resolution diagnostics corroborated with pinned Gradle build",
        )
    return "generation_verifier_gradle_fallback_result", ""


def _gradle_fallback_receipt(
    runtime: Any,
    root: Path,
    *,
    runtime_module: Any,
    jdt_error: BaseException,
    gradle_runner_factory: Any | None = None,
    corroboration: bool = False,
) -> dict[str, Any]:
    return _gradle_fallback_receipt_impl(
        runtime, root, runtime_module=runtime_module, jdt_error=jdt_error,
        gradle_runner_factory=gradle_runner_factory, corroboration=corroboration,
    )


def _gradle_fallback_receipt_impl(
    runtime: Any,
    root: Path,
    *,
    runtime_module: Any,
    jdt_error: BaseException,
    gradle_runner_factory: Any | None = None,
    corroboration: bool = False,
) -> dict[str, Any]:
    """Run the pinned Gradle build and return a verifier-compatible receipt."""

    from .runner import GradleRunner

    cache_root = Path(runtime.workspace_root).expanduser().resolve() / ".cache" / "gradle"
    runner = (gradle_runner_factory or GradleRunner)(cache_root)
    report = runner.build(Path(root).resolve(), run_gametest=False)
    report_dict = report.to_dict()
    last_log = _last_gradle_log(report, report_dict)
    status, failure_code = _fallback_status(report, last_log)
    diagnostics = _fallback_diagnostics(report, last_log, failure_code=failure_code)
    reason_text = f"{type(jdt_error).__name__}: {jdt_error}"
    receipt: dict[str, Any] = {
        "status": status,
        "complete": True,
        "model_id": f"gradle:{report.gradle_version}",
        "diagnostics": diagnostics,
        "error_count": len(diagnostics),
        "verifier_backend": "gradle_build",
        "build": report_dict,
    }
    receipt.update(_fallback_variant_fields(corroboration, reason_text))
    receipt.update(_failure_fields(failure_code))
    event, corroboration_reason = _fallback_event(corroboration)
    emit_root_cause(
        event,
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result=status,
        reason=corroboration_reason or _fallback_reason(failure_code),
        details={"result": receipt},
    )
    return _structured_fallback_receipt(receipt, runtime_module=runtime_module)


__all__ = ["_gradle_fallback_receipt"]
