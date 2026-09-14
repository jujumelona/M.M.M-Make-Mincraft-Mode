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

    diagnostics: list[dict[str, Any]] = []
    last_log = ""
    if not report.passed:
        commands = report_dict.get("commands")
        if isinstance(commands, list) and commands:
            final_command = commands[-1]
            if isinstance(final_command, dict):
                last_log = _bounded_log_tail(str(final_command.get("log_path") or ""))

    toolchain_unavailable = (
        str(report.status).strip().upper() == "UNAVAILABLE"
        or _is_java_toolchain_failure(report.error, last_log)
    )
    status = "PASS" if report.passed else "UNAVAILABLE" if toolchain_unavailable else "FAIL"

    if not report.passed:
        message = str(report.error or "Gradle build failed.")
        if last_log:
            message += "\n\nGradle log tail:\n" + last_log
        diagnostics.append(
            {
                "severity": 1,
                "source": "gradle",
                "code": (
                    "JAVA_TOOLCHAIN_UNAVAILABLE"
                    if toolchain_unavailable
                    else "GRADLE_BUILD_FAILED"
                ),
                "message": message,
            }
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
    if toolchain_unavailable:
        receipt["failure_class"] = "environment"
        receipt["repairable"] = False
        receipt["code"] = "JAVA_TOOLCHAIN_UNAVAILABLE"

    emit_root_cause(
        "generation_verifier_gradle_fallback_result",
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result=status,
        reason=(
            "JDT verifier unavailable and Gradle Java toolchain unavailable"
            if toolchain_unavailable
            else "JDT verifier unavailable; pinned Gradle build used as host verifier"
        ),
        details={"result": receipt},
    )
    return runtime_module._bounded_result(receipt)


def install() -> None:
    """Compatibility hook; fallback dispatch is owned by the verifier itself."""

    return None


__all__ = ["install"]
