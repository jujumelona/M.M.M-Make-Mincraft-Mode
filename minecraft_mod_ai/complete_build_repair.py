"""Build/repair execution and checkpoint ownership for complete production."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .model_router import ModelRouter
from .repair_guard import RepairEngine
from .repairability import source_repair_block_reason
from .runner import GradleRunner
from .scale_policy import ScalePolicy
from .work_graph import DurableWorkLedger, run_named_checkpoint


class BuildRepairOptions(Protocol):
    run_gametest: bool
    auto_repair: bool
    max_repair_attempts: int | None


def _attested_repair_build(repair_result: Any) -> dict[str, Any] | None:
    if not isinstance(repair_result, dict) or repair_result.get("status") != "PASS":
        return None
    evidence = repair_result.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("passed") is not True:
        return None
    build = evidence.get("build")
    if not isinstance(build, dict) or build.get("status") != "PASS":
        return None
    return build


def _blocked_repair_result(
    build: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Preserve a non-source failure without activating the source mutation loop."""

    return {
        "schema_version": "mmm/repair-result-v2",
        "status": "FAIL",
        "attempts": 0,
        "stop_reason": "non_source_repairable",
        "repairable": False,
        "reason": reason,
        "evidence": {
            "passed": False,
            "build": build,
        },
        "patch_receipts": [],
    }


def _run_source_repair(
    *,
    build: dict[str, Any],
    project_root: Path,
    cache: Path,
    run_gametest: bool,
    max_repair_attempts: int | None,
    router: ModelRouter | None,
    router_factory: Callable[[], ModelRouter],
    policy: ScalePolicy,
) -> tuple[dict[str, Any], ModelRouter]:
    active_router = router or router_factory()
    repair = RepairEngine(
        router=active_router, gradle_cache=cache, policy=policy
    ).repair(
        project_root,
        run_gametest=run_gametest,
        max_attempts=max_repair_attempts,
    )
    attested = _attested_repair_build(repair)
    if attested is not None:
        build = dict(attested)
    else:
        build = GradleRunner(cache).build(
            project_root, run_gametest=run_gametest
        ).to_dict()
    return {"build": build, "repair": repair}, active_router


def run_build_with_repair(
    *,
    project_root: Path,
    cache: Path,
    run_gametest: bool,
    auto_repair: bool,
    max_repair_attempts: int | None,
    router: ModelRouter | None,
    router_factory: Callable[[], ModelRouter],
    policy: ScalePolicy,
) -> tuple[dict[str, Any], ModelRouter | None]:
    """Run build and repair only failures that are actionable source defects."""

    build = GradleRunner(cache).build(project_root, run_gametest=run_gametest).to_dict()
    if build.get("status") == "PASS" or not auto_repair:
        return {"build": build, "repair": None}, router

    blocked_reason = source_repair_block_reason(build=build)
    if blocked_reason is not None:
        return {
            "build": build,
            "repair": _blocked_repair_result(build, reason=blocked_reason),
        }, router

    return _run_source_repair(
        build=build,
        project_root=project_root,
        cache=cache,
        run_gametest=run_gametest,
        max_repair_attempts=max_repair_attempts,
        router=router,
        router_factory=router_factory,
        policy=policy,
    )


def run_build_repair_checkpoint(
    *,
    ledger: DurableWorkLedger,
    graph_hash: str,
    validation_manifest: str,
    project_root: Path,
    run_root: Path,
    options: BuildRepairOptions,
    router: ModelRouter | None,
    router_factory: Callable[[], ModelRouter],
    policy: ScalePolicy,
    validate_cached: Callable[[Any], bool],
) -> tuple[dict[str, Any], ModelRouter | None]:
    """Own the durable Gradle/repair checkpoint and return any router it activated."""

    cache = run_root / ".cache/gradle"
    active_router = router

    def action() -> dict[str, Any]:
        nonlocal active_router
        bundle, active_router = run_build_with_repair(
            project_root=project_root,
            cache=cache,
            run_gametest=options.run_gametest,
            auto_repair=options.auto_repair,
            max_repair_attempts=options.max_repair_attempts,
            router=active_router,
            router_factory=router_factory,
            policy=policy,
        )
        return bundle

    bundle = run_named_checkpoint(
        ledger,
        "gradle-build",
        stage="build",
        input_value={
            "graph_hash": graph_hash,
            "project_manifest": validation_manifest,
            "run_gametest": options.run_gametest,
            "auto_repair": options.auto_repair,
            "max_repair_attempts": options.max_repair_attempts,
        },
        action=action,
        encode=lambda value: value,
        decode=lambda cached: cached,
        validate_cached=lambda cached: validate_cached(cached.get("build")),
    )
    return bundle, active_router
