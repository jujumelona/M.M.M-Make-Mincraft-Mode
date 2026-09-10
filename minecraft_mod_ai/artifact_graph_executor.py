from __future__ import annotations

"""Deterministic dependency executor for already-expanded ArtifactJobs."""

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def execute_artifact_graph(
    jobs: Iterable[ArtifactJob],
    *,
    context: dict[str, Any] | None = None,
    router: Any = None,
    port_registry: PortRegistry | None = None,
    base_dir: Any = None,
) -> dict[str, Any]:
    """Run each dependency-ready wave concurrently, preserving deterministic receipts."""
    pending = list(jobs)
    if len({job.job_id for job in pending}) != len(pending):
        raise ArtifactGraphError("ARTIFACT_DUPLICATE_JOB_ID")
    registry = port_registry or PortRegistry()
    producers = _producer_index(pending)

    missing_external: dict[str, list[str]] = {}
    for job in pending:
        missing = [
            name
            for name in job.requires
            if name not in producers and not registry.has(name)
        ]
        if missing:
            missing_external[job.job_id] = missing
    if missing_external:
        raise ArtifactGraphError(f"ARTIFACT_GRAPH_MISSING_PRODUCER: {missing_external}")

    receipts: list[dict[str, Any]] = []
    completed: list[str] = []
    while pending:
        runnable = [
            job for job in pending if all(registry.has(name) for name in job.requires)
        ]
        if not runnable:
            blocked = {
                job.job_id: [name for name in job.requires if not registry.has(name)]
                for job in pending
            }
            raise ArtifactGraphError(f"ARTIFACT_GRAPH_DEADLOCK: {blocked}")

        # Jobs in one wave have all dependencies satisfied before the wave starts.
        # Execute them concurrently, then commit receipts in original graph order so
        # scheduling jitter never changes externally visible output ordering.
        wave_results: dict[str, dict[str, Any]] = {}
        if len(runnable) == 1:
            job = runnable[0]
            wave_results[job.job_id] = _execute_one(
                job,
                context=context,
                router=router,
                registry=registry,
                base_dir=base_dir,
            )
        else:
            with ThreadPoolExecutor(
                max_workers=_parallelism(len(runnable)),
                thread_name_prefix="mmm-artifact",
            ) as pool:
                futures = {
                    pool.submit(
                        _execute_one,
                        job,
                        context=context,
                        router=router,
                        registry=registry,
                        base_dir=base_dir,
                    ): job
                    for job in runnable
                }
                try:
                    for future in as_completed(futures):
                        job = futures[future]
                        wave_results[job.job_id] = future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

        for job in runnable:
            receipt = wave_results[job.job_id]
            if receipt.get("status") != "PASS":
                raise ArtifactGraphError(
                    f"ARTIFACT_JOB_FAILED: {job.job_id!r} returned {receipt.get('status')!r}"
                )
            receipts.append(receipt)
            completed.append(job.job_id)
            pending.remove(job)

    return {
        "status": "PASS",
        "completed_jobs": completed,
        "receipts": receipts,
        "ports": {name: port.to_dict() for name, port in registry.all_ports().items()},
    }
