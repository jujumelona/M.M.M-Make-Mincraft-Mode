from __future__ import annotations

"""Pairwise counterexample-style verification for ambiguous repair candidates."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


def _paths(operations: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(item.get("path", "")).replace("\\", "/")
        for item in operations
        if str(item.get("path", "")).strip()
    }


def _content_map(operations: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in operations:
        path = str(item.get("path", "")).replace("\\", "/")
        if not path:
            continue
        content = item.get("content")
        replacements = item.get("replacements")
        if isinstance(content, str):
            result[path] = content
        elif replacements is not None:
            result[path] = json.dumps(replacements, ensure_ascii=False, sort_keys=True)
    return result


def build_discriminating_plan(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_paths = _paths(left)
    right_paths = _paths(right)
    changed = sorted(left_paths ^ right_paths)
    shared = sorted(left_paths & right_paths)
    left_content = _content_map(left)
    right_content = _content_map(right)
    differing_shared = [path for path in shared if left_content.get(path) != right_content.get(path)]
    focus = sorted(set(changed + differing_shared))[:24]
    probes: list[str] = ["gradle_build"]
    lowered = " ".join(focus).casefold()
    if any(path.endswith(".json") for path in focus):
        probes.append("json_resource_parse")
    if any(marker in lowered for marker in ("network", "packet", "payload")):
        probes.append("network_compile_boundary")
    if any(marker in lowered for marker in ("world", "entity", "event", "recipe", "loot", "dimension")):
        probes.append("gametest_if_available")
    if any(path.endswith((".gradle", ".gradle.kts", "gradle.properties", "fabric.mod.json")) for path in focus):
        probes.append("dependency_metadata")
    return {
        "focus_paths": focus,
        "left_only": sorted(left_paths - right_paths)[:16],
        "right_only": sorted(right_paths - left_paths)[:16],
        "different_shared": differing_shared[:16],
        "probes": probes,
    }


def _json_probe(root: Path, focus_paths: Sequence[str]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for path in focus_paths:
        if not path.endswith(".json"):
            continue
        target = root / path
        if not target.is_file():
            continue
        try:
            json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}:{type(exc).__name__}:{str(exc)[:240]}")
    return not errors, errors


def verify_candidate(
    root: Path,
    operations: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    from .performance_final_contract import _clone_source_snapshot
    from .runner import GradleRunner
    from .source_patch import TransactionalSourcePatcher

    stage: Path | None = None
    try:
        stage = _clone_source_snapshot(root)
        TransactionalSourcePatcher(stage).apply([dict(item) for item in operations])
        json_ok, json_errors = _json_probe(stage, list(plan.get("focus_paths", ())))
        if not (stage / "build.gradle").is_file():
            return {
                "status": "NO_GRADLE",
                "json_ok": json_ok,
                "json_errors": json_errors,
                "score_delta": 40.0 if json_ok else -400.0,
            }
        cache = Path(
            os.environ.get(
                "MMM_ACTIVE_VERIFIER_GRADLE_CACHE",
                "~/.cache/mmm/gradle-active-verifier",
            )
        ).expanduser().resolve()
        run_gametest = "gametest_if_available" in set(plan.get("probes", ()))
        report = GradleRunner(
            cache,
            download_timeout_seconds=120,
            command_timeout_seconds=600,
        ).build(stage, run_gametest=run_gametest)
        delta = 0.0
        if report.status == "PASS":
            delta += 700.0
        elif report.status == "FAIL":
            delta -= 900.0
        if json_ok:
            delta += 40.0
        else:
            delta -= 400.0
        return {
            "status": report.status,
            "error": report.error,
            "json_ok": json_ok,
            "json_errors": json_errors,
            "gametest_requested": run_gametest,
            "commands": [
                {
                    "name": command.name,
                    "exit_code": command.exit_code,
                    "timed_out": command.timed_out,
                }
                for command in report.commands
            ],
            "score_delta": delta,
        }
    except Exception as exc:
        return {
            "status": "VERIFIER_ERROR",
            "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            "score_delta": -10.0,
        }
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def discriminate(
    root: Path,
    left: tuple[float, int, Sequence[Mapping[str, Any]], Mapping[str, Any]],
    right: tuple[float, int, Sequence[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[
    tuple[float, int, Sequence[Mapping[str, Any]], dict[str, Any]],
    tuple[float, int, Sequence[Mapping[str, Any]], dict[str, Any]],
]:
    plan = build_discriminating_plan(left[2], right[2])
    left_result = verify_candidate(root, left[2], plan)
    right_result = verify_candidate(root, right[2], plan)
    left_verifier = {**dict(left[3]), "counterexample_plan": plan, "counterexample_result": left_result}
    right_verifier = {**dict(right[3]), "counterexample_plan": plan, "counterexample_result": right_result}
    return (
        (left[0] + float(left_result.get("score_delta", 0.0)), left[1], left[2], left_verifier),
        (right[0] + float(right_result.get("score_delta", 0.0)), right[1], right[2], right_verifier),
    )


__all__ = ["build_discriminating_plan", "discriminate", "verify_candidate"]
