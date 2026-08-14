from __future__ import annotations

"""Escalate ambiguous repair candidates to a stronger executable verifier."""

import os
import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence


def _mode() -> str:
    value = os.environ.get("MMM_ACTIVE_REPAIR_VERIFIER", "auto").strip().casefold()
    return value if value in {"auto", "on", "off"} else "auto"


def _risk(operations: Sequence[Mapping[str, Any]], verifier: Mapping[str, Any]) -> int:
    score = 0
    paths = [str(item.get("path", "")).replace("\\", "/").casefold() for item in operations]
    java_count = sum(path.endswith(".java") for path in paths)
    if java_count >= 2:
        score += 2
    if len(paths) >= 3:
        score += 1
    if any(path.endswith((".gradle", ".gradle.kts", "gradle.properties", "fabric.mod.json")) for path in paths):
        score += 3
    if any("src/main/resources/" in path or path.endswith(".json") for path in paths):
        score += 1
    if any(marker in path for path in paths for marker in ("network", "world", "entity", "screen", "event", "recipe")):
        score += 1
    status = str(verifier.get("jdt_status", ""))
    try:
        errors = int(verifier.get("jdt_error_count", 0) or 0)
    except (TypeError, ValueError):
        errors = 0
    if status in {"UNAVAILABLE", "VERIFIER_ERROR", "NOT_RUN", ""}:
        score += 3
    elif errors == 0:
        score += 1
    return score


def _active_build(self: Any, root: Path, operations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .performance_final_contract import _clone_source_snapshot
    from .runner import GradleRunner
    from .source_patch import TransactionalSourcePatcher

    stage: Path | None = None
    try:
        stage = _clone_source_snapshot(root)
        TransactionalSourcePatcher(stage).apply([dict(item) for item in operations])
        if not (stage / "build.gradle").is_file():
            return {"active_build_status": "SKIPPED_NO_GRADLE_PROJECT"}
        cache = Path(os.environ.get("MMM_ACTIVE_VERIFIER_GRADLE_CACHE", "~/.cache/mmm/gradle-active-verifier")).expanduser().resolve()
        report = GradleRunner(
            cache,
            download_timeout_seconds=120,
            command_timeout_seconds=600,
        ).build(
            stage,
            run_gametest=os.environ.get("MMM_ACTIVE_REPAIR_GAMETEST", "0").strip().casefold() in {"1", "true", "yes", "on"},
        )
        return {
            "active_build_status": report.status,
            "active_build_error": report.error,
            "active_build_commands": [
                {
                    "name": command.name,
                    "exit_code": command.exit_code,
                    "timed_out": command.timed_out,
                }
                for command in report.commands
            ],
        }
    except Exception as exc:
        return {
            "active_build_status": "VERIFIER_ERROR",
            "active_build_error": f"{type(exc).__name__}: {str(exc)[:600]}",
        }
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def install(optimization_module: Any) -> None:
    current = optimization_module._verify_repair_candidate
    if getattr(current, "_mmm_uncertainty_active_verifier", False):
        return

    @wraps(current)
    def verify_with_escalation(
        self: Any,
        root: Path | None,
        operations: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Any],
    ):
        score, verifier = current(self, root, operations, evidence)
        mode = _mode()
        if mode == "off" or root is None or not operations:
            return score, verifier
        risk = _risk(operations, verifier)
        threshold = 1 if mode == "on" else 4
        if risk < threshold:
            return score, {**dict(verifier), "active_verifier": "NOT_NEEDED", "active_risk": risk}

        extra = _active_build(self, Path(root).resolve(), operations)
        status = str(extra.get("active_build_status", ""))
        if status == "PASS":
            score += 650.0
        elif status == "FAIL":
            score -= 900.0
        elif status == "VERIFIER_ERROR":
            score -= 10.0
        verifier = {
            **dict(verifier),
            **extra,
            "active_verifier": "EXECUTED",
            "active_risk": risk,
        }
        return score, verifier

    verify_with_escalation._mmm_uncertainty_active_verifier = True  # type: ignore[attr-defined]
    verify_with_escalation.__wrapped__ = current  # type: ignore[attr-defined]
    optimization_module._verify_repair_candidate = verify_with_escalation


__all__ = ["install"]
