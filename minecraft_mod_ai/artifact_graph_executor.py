from __future__ import annotations

"""Deterministic dependency executor for already-expanded ArtifactJobs."""

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


def _parallelism(job_count: int) -> int:
    """Bound fan-out so a large graph cannot overwhelm local model/I/O resources."""
    configured = os.getenv("MMM_ARTIFACT_MAX_WORKERS", "").strip()
    if configured:
        try:
            limit = max(1, int(configured))
        except ValueError as exc:
            raise ArtifactGraphError("MMM_ARTIFACT_MAX_WORKERS must be a positive integer") from exc
    else:
        limit = min(8, max(2, os.cpu_count() or 2))
    return min(job_count, limit)


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
    """Validate a completed artifact immediately while sibling generation continues."""
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
    """Run a bounded event-driven DAG and release dependents as soon as producers pass."""
    ordered_jobs = list(jobs)
    if len({job.job_id for job in ordered_jobs}) != len(ordered_jobs):
        raise ArtifactGraphError("ARTIFACT_DUPLICATE_JOB_ID")
    if not ordered_jobs:
        registry = port_registry or PortRegistry()
        return {
            "status": "PASS",
            "completed_jobs": [],
            "receipts": [],
            "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
        }

    registry = port_registry or PortRegistry()
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

    pending: dict[str, ArtifactJob] = {job.job_id: job for job in ordered_jobs}
    running: dict[Future[dict[str, Any]], ArtifactJob] = {}
    results: dict[str, dict[str, Any]] = {}
    worker_count = _parallelism(len(ordered_jobs))

    def ready(job: ArtifactJob) -> bool:
        return all(registry.has(name) for name in job.requires)

    def submit_ready(pool: ThreadPoolExecutor) -> int:
        submitted = 0
        slots = worker_count - len(running)
        if slots <= 0:
            return 0
        for job in ordered_jobs:
            if slots <= 0:
                break
            if job.job_id not in pending or not ready(job):
                continue
            pending.pop(job.job_id)
            future = pool.submit(
                _execute_one,
                job,
                context=context,
                router=router,
                registry=registry,
                base_dir=base_dir,
            )
            running[future] = job
            submitted += 1
            slots -= 1
        return submitted

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="mmm-artifact",
    ) as pool:
        submit_ready(pool)
        while pending or running:
            if not running:
                blocked = {
                    job.job_id: [name for name in job.requires if not registry.has(name)]
                    for job in pending.values()
                }
                raise ArtifactGraphError(f"ARTIFACT_GRAPH_DEADLOCK: {blocked}")

            done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            try:
                for future in done:
                    job = running.pop(future)
                    receipt = future.result()
                    _validate_completed_job(
                        job,
                        receipt,
                        context=context,
                        base_dir=base_dir,
                    )
                    results[job.job_id] = receipt
                    # The checkpoint executor publishes produced ports before returning.
                    # Refill capacity immediately so direct dependents do not wait for
                    # unrelated slow siblings from the previous readiness set.
                    submit_ready(pool)
            except BaseException:
                for future in running:
                    future.cancel()
                raise

            submit_ready(pool)

    completed_jobs = [job.job_id for job in ordered_jobs]
    receipts = [results[job.job_id] for job in ordered_jobs]
    return {
        "status": "PASS",
        "completed_jobs": completed_jobs,
        "receipts": receipts,
        "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
    }
