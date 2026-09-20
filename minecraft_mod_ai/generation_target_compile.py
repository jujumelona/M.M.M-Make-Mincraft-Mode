from __future__ import annotations

"""Host-owned compile oracle for the in-generation Java coding loop."""

from pathlib import Path
from typing import Any

from .compiler_diagnostics import compiler_log_diagnostics, normalize_source_path
from .root_cause_trace import emit_root_cause
from .runner import BuildRunnerError, GradleRunner


def _gradle_cache_for(project_root: Path) -> Path:
    """Reuse the owning production run cache even before first materialization.

    Generation runs inside a resumable checkpoint below the run root while the
    production build later uses the run-scoped .cache/gradle directory. Requiring
    that cache directory to already exist made the first target compile fall back
    to a project-local cold cache, so Gradle/dependency state was populated twice.
    """

    ancestors = (project_root, *project_root.parents)
    for ancestor in ancestors:
        checkpoint_root = ancestor / ".minecraft_ai" / ".mmm-custom-checkpoints"
        if checkpoint_root.is_dir():
            return ancestor / ".cache" / "gradle"

    for ancestor in ancestors:
        candidate = ancestor / ".cache" / "gradle"
        if candidate.is_dir():
            return candidate
    return project_root / ".minecraft_ai" / "cache" / "gradle"


def run_generation_target_compile(
    project_root: str | Path,
    *,
    target_path: str,
) -> dict[str, Any]:
    """Compile the staged project and classify only task-owned source failures as repairable.

    This is part of generation, not the post-generation RepairEngine. The same coder gets
    exact javac diagnostics immediately. Failures outside the owned target are deferred to
    normal project validation/repair so a custom coder cannot mutate someone else's source.
    """

    root = Path(project_root).expanduser().resolve()
    target = normalize_source_path(target_path, project_root=root)
    if not target or not target.endswith(".java"):
        raise ValueError("generation target_compile requires one project-relative Java target")

    cache = _gradle_cache_for(root)
    emit_root_cause(
        "generation_target_compile_start",
        stage="generation",
        operation="target_compile",
        gate="compile_backed_coder",
        result="START",
        details={"project_root": str(root), "target_path": target, "gradle_cache": str(cache)},
    )
    try:
        build = GradleRunner(cache).compile_java(root).to_dict()
    except (BuildRunnerError, OSError, TimeoutError) as exc:
        build = {
            "status": "UNAVAILABLE",
            "error": f"{type(exc).__name__}: {exc}",
            "commands": [],
        }

    diagnostics = compiler_log_diagnostics(build, project_root=root)
    owned = [item for item in diagnostics if item.get("path") == target]
    build_status = str(build.get("status") or "").strip().upper()

    if build_status == "PASS":
        status = "PASS"
        reason = "target compiler passed"
    elif owned:
        status = "FAIL"
        reason = "target compiler reported errors in the task-owned Java source"
    elif build_status in {"UNAVAILABLE", "TIMEOUT", "TIMED_OUT"}:
        status = "UNAVAILABLE"
        reason = str(build.get("error") or "target compiler unavailable")
    else:
        status = "DEFERRED"
        reason = (
            "build failed without a compiler diagnostic owned by this task; "
            "defer to project-level validation/repair"
        )

    receipt = {
        "schema_version": "mmm/generation-target-compile-v1",
        "status": status,
        "target_path": target,
        "diagnostics": owned,
        "all_compiler_diagnostics": diagnostics,
        "build": build,
        "reason": reason,
    }
    emit_root_cause(
        "generation_target_compile_result",
        stage="generation",
        operation="target_compile",
        gate="compile_backed_coder",
        result="PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "SKIP",
        reason=reason,
        details={
            "status": status,
            "target_path": target,
            "owned_diagnostic_count": len(owned),
            "compiler_diagnostic_count": len(diagnostics),
            "build_status": build_status,
        },
    )
    return receipt


__all__ = ["run_generation_target_compile"]
