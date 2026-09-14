from __future__ import annotations

"""Install a trustworthy Gradle fallback for generation-time JDT outages.

The ordinary verifier remains JDT.  This module only handles infrastructure
unavailability after the JDT verifier has already failed closed.  The fallback
runs the repository's pinned Gradle build and returns a real PASS/FAIL receipt;
it never converts an unavailable verifier into synthetic success.
"""

from functools import wraps
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause

_MARKER = "_mmm_generation_gradle_fallback"
_MAX_LOG_TAIL_CHARS = 16 * 1024
_MAX_LOG_TAIL_LINES = 120


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
    if not report.passed:
        commands = report_dict.get("commands")
        last_log = ""
        if isinstance(commands, list) and commands:
            final_command = commands[-1]
            if isinstance(final_command, dict):
                last_log = _bounded_log_tail(str(final_command.get("log_path") or ""))
        message = str(report.error or "Gradle build failed.")
        if last_log:
            message += "\n\nGradle log tail:\n" + last_log
        diagnostics.append(
            {
                "severity": 1,
                "source": "gradle",
                "code": "GRADLE_BUILD_FAILED",
                "message": message,
            }
        )

    receipt: dict[str, Any] = {
        "status": "PASS" if report.passed else "FAIL",
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
    emit_root_cause(
        "generation_verifier_gradle_fallback_result",
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result="PASS" if report.passed else "FAIL",
        reason="JDT verifier unavailable; pinned Gradle build used as host verifier",
        details={"result": receipt},
    )
    return runtime_module._bounded_result(receipt)


def install() -> None:
    """Wrap the finalized generation verifier with a real-build fallback."""

    from . import generation_verifier_resilience as verifier_module

    current = verifier_module.run_generation_verifier
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def run_generation_verifier_with_gradle_fallback(
        runtime: Any,
        arguments: Any,
        *,
        runtime_module: Any,
        java_service_factory: Any | None = None,
    ) -> dict[str, Any]:
        try:
            return current(
                runtime,
                arguments,
                runtime_module=runtime_module,
                java_service_factory=java_service_factory,
            )
        except runtime_module.AgentToolRuntimeError as exc:
            # Only verifier infrastructure failure is eligible. Source diagnostics
            # remain ordinary FAIL results and must never be hidden by another backend.
            if "JDT is unavailable" not in str(exc):
                raise
            root, _ = runtime_module._discover_model_project_root(runtime.workspace_root)
            emit_root_cause(
                "generation_verifier_gradle_fallback_start",
                stage="generation",
                operation="run_gradle_build",
                gate="target_compile",
                result="START",
                reason=str(exc),
            )
            try:
                return _gradle_fallback_receipt(
                    runtime,
                    Path(root),
                    runtime_module=runtime_module,
                    jdt_error=exc,
                )
            except Exception as fallback_exc:
                emit_root_cause(
                    "generation_verifier_gradle_fallback_unavailable",
                    stage="generation",
                    operation="run_gradle_build",
                    gate="target_compile",
                    result="FAIL",
                    reason=f"{type(fallback_exc).__name__}: {fallback_exc}",
                    exc=fallback_exc,
                )
                raise runtime_module.AgentToolRuntimeError(
                    "Generation verification has no healthy backend: "
                    f"JDT unavailable ({exc}); Gradle fallback unavailable "
                    f"({type(fallback_exc).__name__}: {fallback_exc})"
                ) from fallback_exc

    setattr(run_generation_verifier_with_gradle_fallback, _MARKER, True)
    run_generation_verifier_with_gradle_fallback.__wrapped__ = current
    verifier_module.run_generation_verifier = run_generation_verifier_with_gradle_fallback


__all__ = ["install"]
