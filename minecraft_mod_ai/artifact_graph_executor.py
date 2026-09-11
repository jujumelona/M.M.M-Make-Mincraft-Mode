from __future__ import annotations

"""Deterministic dependency executor for already-expanded ArtifactJobs."""

import heapq
import os
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .artifact_job import ArtifactJob
from .artifact_job_checkpoint import execute_checkpointed_job
from .artifact_ports import PortRegistry
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
    instead of degenerating toward quadratic work on large artifact sets.
    """
    ordered_jobs = list(jobs)
    from .resolved_version_context import execution_context

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
        receipt = _execute_one(job, context=context, router=router, registry=registry, base_dir=base_dir)
        _validate_completed_job(job, receipt, context=context, base_dir=base_dir)
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
    for producer_id in dependents:
        dependents[producer_id].sort(key=order.__getitem__)

    ready_heap: list[tuple[int, str]] = [
        (order[job.job_id], job.job_id)
        for job in ordered_jobs
        if indegree[job.job_id] == 0
    ]
    heapq.heapify(ready_heap)
    pending_ids = set(by_id)
    generation_futures: dict[Future[dict[str, Any]], ArtifactJob] = {}
    validation_futures: dict[Future[None], tuple[ArtifactJob, dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {}
    validated_jobs: set[str] = set()
    generation_workers = _parallelism(len(ordered_jobs))
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
            generation_futures[
                pool.submit(
                    _execute_one,
                    job,
                    context=context,
                    router=router,
                    registry=registry,
                    base_dir=base_dir,
                )
            ] = job
            submitted += 1
            slots -= 1
        return submitted

    with ThreadPoolExecutor(
        max_workers=generation_workers,
        thread_name_prefix="mmm-artifact-generate",
    ) as generation_pool, ThreadPoolExecutor(
        max_workers=validation_workers,
        thread_name_prefix="mmm-artifact-validate",
    ) as validation_pool:
        submit_ready(generation_pool)
        try:
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
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                newly_validated: list[str] = []
                for future in done:
                    if future in generation_futures:
                        job = generation_futures.pop(future)
                        receipt = future.result()
                        validation_future = validation_pool.submit(
                            _validate_completed_job,
                            job,
                            receipt,
                            context=context,
                            base_dir=base_dir,
                        )
                        validation_futures[validation_future] = (job, receipt)
                    else:
                        job, receipt = validation_futures.pop(future)
                        future.result()
                        validated_jobs.add(job.job_id)
                        results[job.job_id] = receipt
                        newly_validated.append(job.job_id)

                # Apply all completions as one deterministic readiness update. The
                # heap restores original job order even when futures finish in a
                # different order on every run.
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

                # Refill generation immediately after every generation or validation
                # completion. A dependent becomes eligible only after its producer's
                # validation succeeds, while unrelated work continues in parallel.
                submit_ready(generation_pool)
        except BaseException:
            for future in generation_futures:
                future.cancel()
            for future in validation_futures:
                future.cancel()
            raise

    completed_jobs = [job.job_id for job in ordered_jobs]
    receipts = [results[job.job_id] for job in ordered_jobs]
    return {
        "status": "PASS",
        "completed_jobs": completed_jobs,
        "receipts": receipts,
        "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
    }
