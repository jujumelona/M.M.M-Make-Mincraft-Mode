from __future__ import annotations

"""Deterministic dependency executor for already-expanded ArtifactJobs."""

import heapq
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .artifact_job import ArtifactJob
from .artifact_job_checkpoint import execute_checkpointed_job
from .artifact_ports import PortRegistry
from .model_completion_trace import completion_trace_context, report_active_completions
from .model_concurrency import (
    ModelExecutionBudget,
    planning_work_unit_timeout_seconds,
    run_with_model_execution_deadline,
)
from .task_template_runner import execute_artifact_template


class ArtifactGraphError(RuntimeError):
    pass


def _producer_index(jobs: list[ArtifactJob]) -> dict[str, str]:
    producers: dict[str, str] = {}
    for job in jobs:
        for port_name in job.produces:
            prior = producers.get(port_name)
            if prior is not None and prior != job.job_id:
                raise ArtifactGraphError(
                    f"ARTIFACT_DUPLICATE_PRODUCER: {port_name!r} is produced by "
                    f"{prior!r} and {job.job_id!r}"
                )
            producers[port_name] = job.job_id
    return producers


def _declared_java_source_ports(jobs: list[ArtifactJob]) -> dict[str, str]:
    """Map admitted producer ports to actual Java paths, not guessed class names."""
    from .implementation_identity import ExecutorType
    from .task_template_catalog import load_template

    paths: dict[str, str] = {}
    for job in jobs:
        if job.executor_type is not ExecutorType.PYTHON_GENERATOR or not job.canonical_leaf:
            continue
        # Publication is optional, as in execute_generator_job. A standalone
        # feature can materialize its Java source without exposing graph ports.
        if not job.produces:
            continue
        inputs = job.deterministic_inputs.get("_canonical_inputs")
        if not isinstance(inputs, dict):
            continue
        manifest = load_template(job.canonical_leaf)
        if len(job.produces) != len(manifest["outputs"]):
            raise ArtifactGraphError(
                "ARTIFACT_GENERATOR_OUTPUT_PORT_ARITY: " + job.job_id
                + f" expected={len(manifest['outputs'])} actual={len(job.produces)}"
            )
        try:
            port = next(x["name"] for x in manifest["inputs"]
                        if x["type"] == "specification")
            spec = inputs[port]
        except (KeyError, StopIteration, TypeError):
            continue
        if not isinstance(spec, dict) or spec.get("language") != "java":
            continue
        for name, output in zip(job.produces, manifest["outputs"], strict=True):
            if output["type"] == "code_fragment":
                path = spec.get("target_path")
                if not isinstance(path, str) or not path.endswith(".java"):
                    raise ArtifactGraphError("ARTIFACT_JAVA_SOURCE_PATH_MISSING: " + job.job_id)
                paths[name] = path
    return paths



def _wire_validated_java_startup(jobs, receipts, *, context, base_dir) -> None:
    """Connect only materialized, graph-validated Fabric Java producer owners."""
    if not isinstance(context, dict):
        return
    root_value = context.get("project_root") or base_dir
    if not root_value:
        return
    root = Path(root_value).resolve()
    if not (root / "src/main/resources/fabric.mod.json").is_file():
        return
    approved = []
    by_path = {}
    for job, receipt in zip(jobs, receipts, strict=True):
        if not job.canonical_leaf or not isinstance(receipt, dict):
            continue
        materialization = receipt.get("materialization")
        if not isinstance(materialization, dict):
            continue
        raw = materialization.get("path")
        if not isinstance(raw, str) or not raw.endswith(".java"):
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (Path(base_dir).resolve() if base_dir else root) / path
        # Check the lexical path before following symlinks; otherwise a
        # producer path alias could launder another generated class as its
        # own verified Java initializer.
        if not path.is_absolute():
            raise ArtifactGraphError("ARTIFACT_JAVA_STARTUP_SOURCE_INVALID")
        relative = path.relative_to(root) if path.is_relative_to(root) else None
        if relative is None or ".." in relative.parts:
            raise ArtifactGraphError("ARTIFACT_JAVA_STARTUP_SOURCE_OUTSIDE_WORKSPACE")
        current = root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise ArtifactGraphError("ARTIFACT_JAVA_STARTUP_SOURCE_SYMLINK")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ArtifactGraphError("ARTIFACT_JAVA_STARTUP_SOURCE_OUTSIDE_WORKSPACE")
        by_path.setdefault(resolved, []).append((job.canonical_leaf, materialization))

    for resolved, contributions in by_path.items():
        # Anchored registries compose several jobs into one compilation unit.
        # Verify every intermediate receipt reaches the actual final bytes via
        # other validated, same-path Java patches, independent of finish order.
        import hashlib

        from .artifact_java_materialization import (
            read_java_patch_history,
            verify_java_patch_chain,
        )
        if not resolved.is_file():
            raise ArtifactGraphError("ARTIFACT_JAVA_STARTUP_SOURCE_UNVERIFIED")
        final_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
        relative = resolved.relative_to(root).as_posix()
        history = read_java_patch_history(root, relative, final_hash)
        if not verify_java_patch_chain([row for _, row in contributions] + history, final_hash):
            raise ArtifactGraphError(
                "ARTIFACT_JAVA_STARTUP_SOURCE_UNVERIFIED: " + resolved.relative_to(root).as_posix())
        approved.extend((resolved.relative_to(root).as_posix(), leaf)
                        for leaf, _ in contributions)
    if approved:
        from .host_fabric_initializer_wiring import wire_completed_fabric_java
        wire_completed_fabric_java(root, approved)


def _configured_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ArtifactGraphError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ArtifactGraphError(f"{name} must be a positive integer")
    return value


def _parallelism(job_count: int) -> int:
    """Bound fan-out so a large graph cannot overwhelm local model/I/O resources."""
    default = min(8, max(2, os.cpu_count() or 2))
    return min(job_count, _configured_positive_int("MMM_ARTIFACT_MAX_WORKERS", default))


def _generation_parallelism(job_count: int, router: Any) -> int:
    """Avoid starting more competing native inference leases than actual slots.

    A graph may own eight artifact threads while a single-GPU llama server only
    runs one decoder. The old surplus threads consumed their whole work-unit
    deadline waiting for capacity, causing unrelated generation jobs to fail
    before they made any model request.
    """
    workers = _parallelism(job_count)
    from .model_concurrency import (
        router_native_model_parallelism,
        router_owns_native_model,
    )
    if router_owns_native_model(router, role="coder"):
        workers = min(workers, router_native_model_parallelism(router, role="coder"))
    return max(1, workers)


def _validation_parallelism(job_count: int, generation_workers: int) -> int:
    """Keep validation concurrent without allowing it to dominate the machine."""
    default = min(4, max(1, generation_workers))
    return min(job_count, _configured_positive_int("MMM_ARTIFACT_VALIDATION_WORKERS", default))


def _execute_one(
    job: ArtifactJob,
    *,
    context: dict[str, Any] | None,
    router: Any,
    registry: PortRegistry,
    base_dir: Any,
) -> dict[str, Any]:
    with completion_trace_context(job_id=job.job_id, canonical_leaf=job.canonical_leaf):
        return execute_checkpointed_job(
            job,
            context=context,
            router=router,
            registry=registry,
            base_dir=base_dir,
            execute=execute_artifact_template,
        )


def _validate_completed_job(
    job: ArtifactJob,
    receipt: dict[str, Any],
    *,
    context: dict[str, Any] | None,
    registry: PortRegistry,
    base_dir: Any,
) -> None:
    """Validate one materialized artifact without mutating scheduler state."""
    from .resolved_version_context import execution_context

    resolved = execution_context(context, job)
    if resolved is not None:
        resolved.assert_context(receipt.get("context_id"))
    if receipt.get("status") != "PASS":
        raise ArtifactGraphError(
            f"ARTIFACT_JOB_FAILED: {job.job_id!r} returned {receipt.get('status')!r}"
        )

    missing_outputs = [
        name
        for name in job.produces
        if not registry.has(name)
    ]
    if missing_outputs:
        raise ArtifactGraphError(
            f"ARTIFACT_OUTPUT_PORT_MISSING: {job.job_id!r} did not publish "
            f"{missing_outputs}"
        )

    materialization = receipt.get("materialization")
    if isinstance(materialization, dict):
        raw_path = materialization.get("path")
        if raw_path:
            path = Path(str(raw_path))
            if not path.is_absolute() and base_dir is not None:
                candidate = Path(base_dir) / path
                if candidate.exists() or not path.exists():
                    path = candidate
            if not path.exists():
                raise ArtifactGraphError(
                    f"ARTIFACT_OUTPUT_MISSING: {job.job_id!r} materialized {path} but it does not exist"
                )

    validator: Callable[[ArtifactJob, dict[str, Any]], Any] | None = None
    if isinstance(context, dict):
        candidate = context.get("artifact_validator")
        if callable(candidate):
            validator = candidate
    if validator is not None:
        validation = validator(job, receipt)
        if validation is False:
            raise ArtifactGraphError(f"ARTIFACT_VALIDATION_FAILED: {job.job_id!r}")
        if isinstance(validation, dict) and validation.get("status") not in {None, "PASS"}:
            raise ArtifactGraphError(
                f"ARTIFACT_VALIDATION_FAILED: {job.job_id!r} returned {validation.get('status')!r}"
            )


def execute_artifact_graph(
    jobs: Iterable[ArtifactJob],
    *,
    context: dict[str, Any] | None = None,
    router: Any = None,
    port_registry: PortRegistry | None = None,
    base_dir: Any = None,
) -> dict[str, Any]:
    """Run generation and validation as a bounded event-driven dependency pipeline.

    Readiness is maintained with producer indegrees and a deterministic heap. No job
    list is rescanned after each completion, so scheduler work grows with graph edges
    instead of degenerating toward quadratic work on large artifact sets. Generation
    and validation carry shared, progress-refreshable deadlines. Teardown cancels
    their model/publication scopes without waiting for an uncooperative worker.
    """
    ordered_jobs = list(jobs)
    from .resolved_version_context import execution_context

    context = dict(context or {})
    for job in ordered_jobs:
        execution_context(context, job)
    if len({job.job_id for job in ordered_jobs}) != len(ordered_jobs):
        raise ArtifactGraphError("ARTIFACT_DUPLICATE_JOB_ID")
    registry = port_registry or PortRegistry()
    if context and context.get("resolved_version_context"):
        from .resolved_version_context import ResolvedVersionContext

        registry.bind_context(ResolvedVersionContext.from_dict(context["resolved_version_context"]).context_id)
    if not ordered_jobs:
        return {
            "status": "PASS",
            "completed_jobs": [],
            "receipts": [],
            "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
        }

    producers = _producer_index(ordered_jobs)
    # The scheduler already knows which approved producer owns each code port.
    # Make its materialized path visible to the consumer before Java authoring.
    context["_artifact_java_source_ports"] = _declared_java_source_ports(ordered_jobs)
    missing_external: dict[str, list[str]] = {}
    for job in ordered_jobs:
        missing = [
            name
            for name in job.requires
            if name not in producers and not registry.has(name)
        ]
        if missing:
            missing_external[job.job_id] = missing
    if missing_external:
        raise ArtifactGraphError(f"ARTIFACT_GRAPH_MISSING_PRODUCER: {missing_external}")

    if len(ordered_jobs) == 1:
        job = ordered_jobs[0]
        deadline = time.monotonic() + planning_work_unit_timeout_seconds()
        receipt = run_with_model_execution_deadline(
            deadline,
            _execute_one,
            job,
            context=context,
            router=router,
            registry=registry,
            base_dir=base_dir,
        )
        _validate_completed_job(
            job,
            receipt,
            context=context,
            registry=registry,
            base_dir=base_dir,
        )
        _wire_validated_java_startup([job], [receipt], context=context, base_dir=base_dir)
        return {
            "status": "PASS",
            "completed_jobs": [job.job_id],
            "receipts": [receipt],
            "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
        }

    order = {job.job_id: index for index, job in enumerate(ordered_jobs)}
    by_id = {job.job_id: job for job in ordered_jobs}
    dependency_ids: dict[str, set[str]] = {}
    dependents: dict[str, list[str]] = {job.job_id: [] for job in ordered_jobs}
    for job in ordered_jobs:
        required_producers = {
            producers[name]
            for name in job.requires
            if name in producers
        }
        dependency_ids[job.job_id] = required_producers
        for producer_id in required_producers:
            dependents[producer_id].append(job.job_id)
    indegree = {job_id: len(required) for job_id, required in dependency_ids.items()}
    for dependent_ids in dependents.values():
        dependent_ids.sort(key=order.__getitem__)

    ready_heap: list[tuple[int, str]] = [
        (order[job.job_id], job.job_id)
        for job in ordered_jobs
        if indegree[job.job_id] == 0
    ]
    heapq.heapify(ready_heap)
    pending_ids = set(by_id)
    generation_futures: dict[Future[dict[str, Any]], ArtifactJob] = {}
    validation_futures: dict[Future[None], tuple[ArtifactJob, dict[str, Any]]] = {}
    future_deadlines: dict[Future[Any], ModelExecutionBudget] = {}
    results: dict[str, dict[str, Any]] = {}
    validated_jobs: set[str] = set()
    generation_workers = _generation_parallelism(len(ordered_jobs), router)
    validation_workers = _validation_parallelism(len(ordered_jobs), generation_workers)

    def submit_ready(pool: ThreadPoolExecutor) -> int:
        submitted = 0
        slots = generation_workers - len(generation_futures)
        while slots > 0 and ready_heap:
            _index, job_id = heapq.heappop(ready_heap)
            if job_id not in pending_ids:
                continue
            pending_ids.remove(job_id)
            job = by_id[job_id]
            deadline = ModelExecutionBudget(time.monotonic() + planning_work_unit_timeout_seconds())
            future = pool.submit(
                run_with_model_execution_deadline,
                deadline,
                _execute_one,
                job,
                context=context,
                router=router,
                registry=registry,
                base_dir=base_dir,
            )
            generation_futures[future] = job
            future_deadlines[future] = deadline
            submitted += 1
            slots -= 1
        return submitted

    generation_pool = ThreadPoolExecutor(
        max_workers=generation_workers,
        thread_name_prefix="mmm-artifact-generate",
    )
    validation_pool = ThreadPoolExecutor(
        max_workers=validation_workers,
        thread_name_prefix="mmm-artifact-validate",
    )
    try:
        submit_ready(generation_pool)
        while pending_ids or generation_futures or validation_futures:
            if not generation_futures and not validation_futures:
                blocked = {
                    job_id: [
                        name
                        for name in by_id[job_id].requires
                        if producers.get(name) not in validated_jobs
                        and not (producers.get(name) is None and registry.has(name))
                    ]
                    for job_id in sorted(pending_ids, key=order.__getitem__)
                }
                raise ArtifactGraphError(f"ARTIFACT_GRAPH_DEADLOCK: {blocked}")

            active = tuple(generation_futures) + tuple(validation_futures)
            nearest_deadline = min(future_deadlines[future].deadline for future in active)
            timeout = max(0.0, nearest_deadline - time.monotonic())
            done, _ = wait(
                active,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                expired = [
                    future
                    for future in active
                    if now >= future_deadlines[future].deadline
                ]
                if not expired:
                    continue
                owners = [
                    generation_futures[future].job_id
                    if future in generation_futures
                    else validation_futures[future][0].job_id
                    for future in expired
                ]
                report_active_completions(owners, reason="ARTIFACT_GRAPH_WORK_TIMEOUT")
                for future in expired:
                    future_deadlines[future].cancel()
                    future.cancel()
                raise ArtifactGraphError(f"ARTIFACT_GRAPH_WORK_TIMEOUT: {owners}")

            newly_validated: list[str] = []
            for future in done:
                future_deadlines.pop(future, None)
                if future in generation_futures:
                    job = generation_futures.pop(future)
                    receipt = future.result(timeout=0)
                    validation_deadline = ModelExecutionBudget(
                        time.monotonic() + planning_work_unit_timeout_seconds()
                    )
                    validation_future = validation_pool.submit(
                        run_with_model_execution_deadline,
                        validation_deadline,
                        _validate_completed_job,
                        job,
                        receipt,
                        context=context,
                        registry=registry,
                        base_dir=base_dir,
                    )
                    validation_futures[validation_future] = (job, receipt)
                    future_deadlines[validation_future] = validation_deadline
                else:
                    job, receipt = validation_futures.pop(future)
                    future.result(timeout=0)
                    validated_jobs.add(job.job_id)
                    results[job.job_id] = receipt
                    newly_validated.append(job.job_id)

            # Apply all completions as one deterministic readiness update. The heap
            # restores original job order even when futures finish in a different order.
            for producer_id in sorted(newly_validated, key=order.__getitem__):
                for dependent_id in dependents[producer_id]:
                    indegree[dependent_id] -= 1
                    if indegree[dependent_id] < 0:
                        raise ArtifactGraphError(
                            f"ARTIFACT_GRAPH_INDEGREE_CORRUPT: {dependent_id}"
                        )
                    if indegree[dependent_id] == 0:
                        heapq.heappush(
                            ready_heap,
                            (order[dependent_id], dependent_id),
                        )

            # A dependent becomes eligible only after its producer validates, while
            # unrelated generation keeps all available slots busy.
            submit_ready(generation_pool)
    finally:
        for budget in future_deadlines.values():
            budget.cancel()
        for future in generation_futures:
            future.cancel()
        for future in validation_futures:
            future.cancel()
        generation_pool.shutdown(wait=False, cancel_futures=True)
        validation_pool.shutdown(wait=False, cancel_futures=True)

    completed_jobs = [job.job_id for job in ordered_jobs]
    receipts = [results[job.job_id] for job in ordered_jobs]
    _wire_validated_java_startup(
        ordered_jobs, receipts, context=context, base_dir=base_dir,
    )
    return {
        "status": "PASS",
        "completed_jobs": completed_jobs,
        "receipts": receipts,
        "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
    }
