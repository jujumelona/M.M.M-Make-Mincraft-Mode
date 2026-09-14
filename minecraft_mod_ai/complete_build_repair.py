"""Build/repair execution and checkpoint ownership for complete production."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .model_router import ModelRouter
from .repair_engine import RepairEngine
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
    """Run build, bounded repair when requested, and the fail-closed rebuild path."""

    build = GradleRunner(cache).build(project_root, run_gametest=run_gametest).to_dict()
    repair: dict[str, Any] | None = None
    active_router = router
    if build.get("status") != "PASS" and auto_repair:
        active_router = active_router or router_factory()
        repair = RepairEngine(
            router=active_router, gradle_cache=cache, policy=policy
        ).repair(
            project_root,
            run_gametest=run_gametest,
            max_attempts=max_repair_attempts,
        )
        attested = _attested_repair_build(repair)
        build = (
            dict(attested)
            if attested is not None
            else GradleRunner(cache).build(
                project_root, run_gametest=run_gametest
            ).to_dict()
        )
    return {"build": build, "repair": repair}, active_router


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
