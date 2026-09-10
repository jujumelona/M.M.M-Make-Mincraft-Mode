from __future__ import annotations

"""Deterministic dependency executor for already-expanded ArtifactJobs."""

from collections.abc import Iterable
from typing import Any

from .artifact_job import ArtifactJob
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


def execute_artifact_graph(
    jobs: Iterable[ArtifactJob],
    *,
    context: dict[str, Any] | None = None,
    router: Any = None,
    port_registry: PortRegistry | None = None,
    base_dir: Any = None,
) -> dict[str, Any]:
    """Run jobs only when every declared scoped dependency is available."""
    pending = list(jobs)
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
        raise ArtifactGraphError(
            f"ARTIFACT_GRAPH_MISSING_PRODUCER: {missing_external}"
        )

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

        for job in runnable:
            receipt = execute_artifact_template(
                job,
                context=context,
                router=router,
                port_registry=registry,
                base_dir=base_dir,
            )
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
