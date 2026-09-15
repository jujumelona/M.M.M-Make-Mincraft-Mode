from __future__ import annotations

"""Pairwise generated-counterexample verification for ambiguous repair candidates."""

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .generated_counterexample_tests import (
    build_generated_test_spec,
    install_generated_junit,
    run_generated_test_spec,
)


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
    probes: list[str] = ["generated_counterexample_test", "gradle_build"]
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
        "schema_version": "mmm/counterexample-plan-v2",
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


def _synthetic_verification(
    *,
    plan: Mapping[str, Any],
    build_status: str,
    json_ok: bool,
    commands: Sequence[Mapping[str, Any]],
    generated_result: Mapping[str, Any] | None = None,
    generated_junit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    command_by_name = {str(item.get("name", "")): item for item in commands}
    scenarios: list[dict[str, Any]] = []
    if generated_result is not None:
        status = str(generated_result.get("status", "INCOMPLETE"))
        scenarios.append({
            "scenario_id": "generated_counterexample_test",
            "oracle": "same host-generated A/B invariant suite passes",
            "status": status if status in {"PASS", "FAIL"} else "NOT_RUN",
        })
    if generated_junit is not None and generated_junit.get("status") == "INSTALLED":
        scenarios.append({
            "scenario_id": "generated_junit_test",
            "oracle": "generated JUnit counterexample is executed by Gradle build",
            "status": "PASS" if build_status == "PASS" else ("FAIL" if build_status == "FAIL" else "NOT_RUN"),
        })
    if "json_resource_parse" in set(plan.get("probes", ())):
        scenarios.append({
            "scenario_id": "json_resource_parse",
            "oracle": "all focused JSON resources parse",
            "status": "PASS" if json_ok else "FAIL",
        })
    build_command = command_by_name.get("clean_build")
    if build_command is not None:
        build_pass = build_command.get("exit_code") == 0 and build_command.get("timed_out") is not True
        scenarios.append({
            "scenario_id": "gradle_clean_build",
            "oracle": "clean Gradle build exits zero without timeout",
            "status": "PASS" if build_pass else "FAIL",
        })
    else:
        scenarios.append({
            "scenario_id": "gradle_clean_build",
            "oracle": "clean Gradle build exits zero without timeout",
            "status": "PASS" if build_status == "PASS" else "NOT_RUN",
        })
    if "gametest_if_available" in set(plan.get("probes", ())):
        gametest = command_by_name.get("gametest")
        if gametest is None:
            scenario_status = "NOT_RUN"
        else:
            scenario_status = (
                "PASS"
                if gametest.get("exit_code") == 0 and gametest.get("timed_out") is not True
                else "FAIL"
            )
        scenarios.append({
            "scenario_id": "fabric_gametest",
            "oracle": "focused behavior GameTest exits zero",
            "status": scenario_status,
        })
    terminal = [str(item.get("status", "")) for item in scenarios]
    if scenarios and all(value == "PASS" for value in terminal):
        overall = "PASS"
    elif any(value == "FAIL" for value in terminal):
        overall = "FAIL"
    else:
        overall = "INCOMPLETE"
    return {
        "schema_version": "mmm/synthetic-verification-v1",
        "status": overall,
        "generator": "host-diff-derived-counterexample-v2",
        "isolated_snapshot": True,
        "same_test_for_both_candidates": True,
        "scenarios": scenarios,
    }


def _generated_delta(result: Mapping[str, Any]) -> float:
    status = str(result.get("status", ""))
    assertions = result.get("assertions")
    count = len(assertions) if isinstance(assertions, Sequence) and not isinstance(assertions, (str, bytes)) else 0
    if status == "FAIL":
        return -750.0
    if status == "PASS" and count:
        return 220.0
    return 0.0


def verify_candidate(
    root: Path,
    operations: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    generated_test: Mapping[str, Any] | None = None,
    candidate_verifier: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .performance_final_contract import _clone_source_snapshot
    from .runner import GradleRunner
    from .source_patch import TransactionalSourcePatcher

    stage: Path | None = None
    try:
        stage = _clone_source_snapshot(root)
        TransactionalSourcePatcher(stage).apply([dict(item) for item in operations])
        json_ok, json_errors = _json_probe(stage, list(plan.get("focus_paths", ())))
        test_spec = generated_test or {
            "schema_version": "mmm/generated-counterexample-test-v1",
            "same_test_for_both_candidates": True,
            "focus_paths": list(plan.get("focus_paths", ())),
            "project_namespaces": [],
            "assertions": [],
        }
        generated_result = run_generated_test_spec(
            stage,
            test_spec,
            candidate_verifier=candidate_verifier,
        )
        generated_junit = install_generated_junit(stage, test_spec)
        base_delta = _generated_delta(generated_result)

        if not (stage / "build.gradle").is_file():
            synthetic = _synthetic_verification(
                plan=plan,
                build_status="NO_GRADLE",
                json_ok=json_ok,
                commands=(),
                generated_result=generated_result,
                generated_junit=generated_junit,
            )
            return {
                "status": "NO_GRADLE",
                "json_ok": json_ok,
                "json_errors": json_errors,
                "generated_test": test_spec,
                "generated_test_result": generated_result,
                "generated_junit": generated_junit,
                "synthetic_verification": synthetic,
                "score_delta": base_delta + (40.0 if json_ok else -400.0),
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
        commands = [
            {
                "name": command.name,
                "exit_code": command.exit_code,
                "timed_out": command.timed_out,
            }
            for command in report.commands
        ]
        synthetic = _synthetic_verification(
            plan=plan,
            build_status=report.status,
            json_ok=json_ok,
            commands=commands,
            generated_result=generated_result,
            generated_junit=generated_junit,
        )
        delta = base_delta
        if report.status == "PASS":
            delta += 700.0
        elif report.status == "FAIL":
            delta -= 900.0
        if json_ok:
            delta += 40.0
        else:
            delta -= 400.0
        if synthetic["status"] == "FAIL":
            delta -= 350.0
        return {
            "status": report.status,
            "error": report.error,
            "json_ok": json_ok,
            "json_errors": json_errors,
            "gametest_requested": run_gametest,
            "commands": commands,
            "generated_test": test_spec,
            "generated_test_result": generated_result,
            "generated_junit": generated_junit,
            "synthetic_verification": synthetic,
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
    left_seed = left[3].get("counterexample_seed") if isinstance(left[3], Mapping) else None
    right_seed = right[3].get("counterexample_seed") if isinstance(right[3], Mapping) else None
    seed = left_seed if isinstance(left_seed, Mapping) else (right_seed if isinstance(right_seed, Mapping) else None)
    generated_test = build_generated_test_spec(
        root,
        focus_paths=list(plan.get("focus_paths", ())),
        left_operations=left[2],
        right_operations=right[2],
        evidence_seed=seed,
    )
    full_plan = {**plan, "generated_test": generated_test}
    left_result = verify_candidate(
        root,
        left[2],
        full_plan,
        generated_test=generated_test,
        candidate_verifier=left[3],
    )
    right_result = verify_candidate(
        root,
        right[2],
        full_plan,
        generated_test=generated_test,
        candidate_verifier=right[3],
    )
    left_verifier = {**dict(left[3]), "counterexample_plan": full_plan, "counterexample_result": left_result}
    right_verifier = {**dict(right[3]), "counterexample_plan": full_plan, "counterexample_result": right_result}
    return (
        (left[0] + float(left_result.get("score_delta", 0.0)), left[1], left[2], left_verifier),
        (right[0] + float(right_result.get("score_delta", 0.0)), right[1], right[2], right_verifier),
    )


__all__ = ["build_discriminating_plan", "discriminate", "verify_candidate"]
