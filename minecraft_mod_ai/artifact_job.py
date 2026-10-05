from __future__ import annotations

"""Concrete ArtifactJob representation for leaf-level generation units."""

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
from typing import Any

from .artifact_ports import PortKind
from .implementation_identity import ExecutorType


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"ARTIFACT_JOB_STRING_REQUIRED: {field}")
    return value


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"ARTIFACT_JOB_ARRAY_REQUIRED: {field}")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"ARTIFACT_JOB_STRING_ARRAY_REQUIRED: {field}")
    return tuple(value)


def parse_artifact_jobs(
    raw_jobs: Iterable[Mapping[str, Any]],
) -> tuple["ArtifactJob", ...]:
    """Parse the one canonical serialized ArtifactJob representation."""

    parsed: list[ArtifactJob] = []
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"ARTIFACT_JOB_OBJECT_REQUIRED: index={index}"
            )
        parsed.append(ArtifactJob.from_dict(dict(raw)))
    return tuple(parsed)


def artifact_owner_module_ids(jobs: Iterable[Any]) -> frozenset[str]:
    """Return module IDs whose production is owned by the canonical artifact graph."""

    owners: set[str] = set()
    for job in jobs or ():
        if isinstance(job, Mapping):
            owner = str(job.get("owner_module") or "").strip()
        else:
            owner = str(getattr(job, "owner_module", "") or "").strip()
        if owner:
            owners.add(owner)
    return frozenset(owners)


@dataclass
class ArtifactJob:
    job_id: str
    template_id: str
    owner_module: str
    executor_type: ExecutorType = field(default=ExecutorType.TEMPLATE, kw_only=True)
    target_path: str = ""
    anchor: str = ""
    operation: str = ""
    expected_sha256: str | None = None
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    required_ports: tuple[dict[str, str], ...] = ()
    ai_slots: tuple[dict[str, Any], ...] = ()
    deterministic_inputs: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    validation_receipts: list[dict[str, Any]] = field(default_factory=list)
    rendered_output: str = ""
    context_id: str = ""
    canonical_leaf: str = ""
    implementation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "template_id": self.template_id,
            "owner_module": self.owner_module,
            "executor_type": self.executor_type.value,
            "target_path": self.target_path,
            "anchor": self.anchor,
            "operation": self.operation,
            "expected_sha256": self.expected_sha256,
            "requires": list(self.requires),
            "produces": list(self.produces),
            "required_ports": list(self.required_ports),
            "ai_slots": list(self.ai_slots),
            "deterministic_inputs": dict(self.deterministic_inputs),
            "status": self.status,
            "validation_receipts": list(self.validation_receipts),
            "rendered_output": self.rendered_output,
            "context_id": self.context_id,
            "canonical_leaf": self.canonical_leaf,
            "implementation_id": self.implementation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactJob:
        if not isinstance(data, dict):
            raise ValueError("ARTIFACT_JOB_OBJECT_REQUIRED")
        expected_fields = {
            "job_id",
            "template_id",
            "owner_module",
            "executor_type",
            "target_path",
            "anchor",
            "operation",
            "expected_sha256",
            "requires",
            "produces",
            "required_ports",
            "ai_slots",
            "deterministic_inputs",
            "status",
            "validation_receipts",
            "rendered_output",
            "context_id",
            "canonical_leaf",
            "implementation_id",
        }
        if set(data) != expected_fields:
            raise ValueError(
                "ARTIFACT_JOB_FIELDS_INVALID: "
                f"missing={sorted(expected_fields - set(data))}, "
                f"unknown={sorted(set(data) - expected_fields)}"
            )
        if not isinstance(data["requires"], list):
            raise ValueError("ARTIFACT_JOB_REQUIRES_ARRAY_REQUIRED")
        if not isinstance(data["produces"], list):
            raise ValueError("ARTIFACT_JOB_PRODUCES_ARRAY_REQUIRED")
        if not isinstance(data["required_ports"], list):
            raise ValueError("ARTIFACT_JOB_REQUIRED_PORTS_ARRAY_REQUIRED")
        if not isinstance(data["ai_slots"], list):
            raise ValueError("ARTIFACT_JOB_AI_SLOTS_ARRAY_REQUIRED")
        if not isinstance(data["deterministic_inputs"], dict):
            raise ValueError("ARTIFACT_JOB_INPUTS_OBJECT_REQUIRED")
        if not isinstance(data["validation_receipts"], list):
            raise ValueError("ARTIFACT_JOB_RECEIPTS_ARRAY_REQUIRED")
        if data["expected_sha256"] is not None and not isinstance(
            data["expected_sha256"], str
        ):
            raise ValueError("ARTIFACT_JOB_EXPECTED_SHA_STRING_REQUIRED")
        for index, port in enumerate(data["required_ports"]):
            if not isinstance(port, dict):
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_OBJECT_REQUIRED: index={index}"
                )
        for index, slot in enumerate(data["ai_slots"]):
            if not isinstance(slot, dict):
                raise ValueError(
                    f"ARTIFACT_JOB_AI_SLOT_OBJECT_REQUIRED: index={index}"
                )
        for index, receipt in enumerate(data["validation_receipts"]):
            if not isinstance(receipt, dict):
                raise ValueError(
                    f"ARTIFACT_JOB_RECEIPT_OBJECT_REQUIRED: index={index}"
                )
        return cls(
            job_id=_string(data["job_id"], "job_id"),
            template_id=_string(data["template_id"], "template_id"),
            owner_module=_string(data["owner_module"], "owner_module"),
            executor_type=ExecutorType(data["executor_type"]),
            target_path=_string(data["target_path"], "target_path"),
            anchor=_string(data["anchor"], "anchor"),
            operation=_string(data["operation"], "operation"),
            expected_sha256=data["expected_sha256"],
            requires=_string_list(data["requires"], "requires"),
            produces=_string_list(data["produces"], "produces"),
            required_ports=tuple(dict(k) for k in data["required_ports"]),
            ai_slots=tuple(dict(s) for s in data["ai_slots"]),
            deterministic_inputs=dict(data["deterministic_inputs"]),
            status=_string(data["status"], "status"),
            validation_receipts=[dict(item) for item in data["validation_receipts"]],
            rendered_output=_string(data["rendered_output"], "rendered_output"),
            context_id=_string(data["context_id"], "context_id"),
            canonical_leaf=_string(data["canonical_leaf"], "canonical_leaf"),
            implementation_id=_string(data["implementation_id"], "implementation_id"),
        )



def validate_artifact_job_graph(
    jobs: Iterable[ArtifactJob],
    *,
    module_ids: Iterable[str] = (),
) -> tuple[ArtifactJob, ...]:
    """Fail closed on canonical artifact-job structure before production."""

    materialized = tuple(jobs)
    allowed_owners = {
        str(value).strip()
        for value in module_ids
        if str(value).strip()
    }
    seen_jobs: set[str] = set()
    producers: dict[str, str] = {}

    for job in materialized:
        job_id = str(job.job_id or "").strip()
        owner = str(job.owner_module or "").strip()
        if not job_id:
            raise ValueError("ARTIFACT_JOB_ID_REQUIRED")
        if not owner:
            raise ValueError(f"ARTIFACT_JOB_OWNER_REQUIRED: {job_id!r}")
        if not job.context_id.strip():
            raise ValueError(f"ARTIFACT_JOB_CONTEXT_REQUIRED: {job_id!r}")
        if not job.canonical_leaf.strip():
            raise ValueError(f"ARTIFACT_JOB_CANONICAL_LEAF_REQUIRED: {job_id!r}")
        if not job.implementation_id.strip():
            raise ValueError(f"ARTIFACT_JOB_IMPLEMENTATION_REQUIRED: {job_id!r}")
        if job.executor_type is ExecutorType.TEMPLATE:
            if not job.template_id.strip():
                raise ValueError(f"ARTIFACT_JOB_TEMPLATE_REQUIRED: {job_id!r}")
            if not job.target_path.strip() or not job.operation.strip():
                raise ValueError(f"ARTIFACT_JOB_TARGET_REQUIRED: {job_id!r}")
        elif job.executor_type is ExecutorType.PYTHON_GENERATOR:
            if job.template_id.strip():
                raise ValueError(
                    f"ARTIFACT_GENERATOR_TEMPLATE_FORBIDDEN: {job_id!r}"
                )
            if not job.implementation_id.startswith("python_generator:"):
                raise ValueError(
                    f"ARTIFACT_GENERATOR_IMPLEMENTATION_INVALID: {job_id!r}"
                )
        if allowed_owners and owner not in allowed_owners:
            raise ValueError(
                f"ARTIFACT_JOB_FOREIGN_OWNER: {job_id!r} -> {owner!r}"
            )
        if job_id in seen_jobs:
            raise ValueError(f"ARTIFACT_DUPLICATE_JOB_ID: {job_id!r}")
        seen_jobs.add(job_id)

        requires = tuple(str(value).strip() for value in job.requires)
        produces = tuple(str(value).strip() for value in job.produces)
        if any(not value for value in (*requires, *produces)):
            raise ValueError(f"ARTIFACT_JOB_PORT_EMPTY: {job_id!r}")
        if len(requires) != len(set(requires)):
            raise ValueError(f"ARTIFACT_JOB_REQUIRE_DUPLICATE: {job_id!r}")
        if len(produces) != len(set(produces)):
            raise ValueError(f"ARTIFACT_JOB_PRODUCE_DUPLICATE: {job_id!r}")
        overlap = sorted(set(requires) & set(produces))
        if overlap:
            raise ValueError(
                f"ARTIFACT_JOB_SELF_DEPENDENCY: {job_id!r} {overlap}"
            )

        required_names: set[str] = set()
        for index, port in enumerate(job.required_ports):
            if not isinstance(port, Mapping):
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_INVALID: {job_id!r}[{index}]"
                )
            if set(port) != {"name", "kind", "target_type"}:
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_FIELDS: {job_id!r}[{index}]"
                )
            name = str(port.get("name") or "").strip()
            kind = str(port.get("kind") or "").strip()
            target_type = str(port.get("target_type") or "").strip()
            if not name or not kind or not target_type:
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_EMPTY: {job_id!r}[{index}]"
                )
            try:
                PortKind(kind)
            except ValueError as exc:
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_KIND: {job_id!r}[{index}] {kind!r}"
                ) from exc
            if name not in requires:
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_NOT_REQUIRED: {job_id!r} {name!r}"
                )
            if name in required_names:
                raise ValueError(
                    f"ARTIFACT_JOB_REQUIRED_PORT_DUPLICATE: {job_id!r} {name!r}"
                )
            required_names.add(name)

        for port in produces:
            prior = producers.get(port)
            if prior is not None and prior != job_id:
                raise ValueError(
                    f"ARTIFACT_DUPLICATE_PRODUCER: {port!r} by {prior!r} and {job_id!r}"
                )
            producers[port] = job_id

    missing = {
        job.job_id: [port for port in job.requires if port not in producers]
        for job in materialized
    }
    missing = {
        job_id: ports
        for job_id, ports in missing.items()
        if ports
    }
    if missing:
        raise ValueError(f"ARTIFACT_GRAPH_MISSING_PRODUCER: {missing}")

    dependencies: dict[str, set[str]] = {
        job.job_id: {
            producers[port]
            for port in job.requires
        }
        for job in materialized
    }
    outgoing: dict[str, set[str]] = {
        job.job_id: set()
        for job in materialized
    }
    indegree = {
        job_id: len(required)
        for job_id, required in dependencies.items()
    }
    for job_id, required in dependencies.items():
        for producer_id in required:
            outgoing[producer_id].add(job_id)

    ready = sorted(
        job_id
        for job_id, degree in indegree.items()
        if degree == 0
    )
    emitted = 0
    while ready:
        job_id = ready.pop(0)
        emitted += 1
        for dependent in sorted(outgoing[job_id]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if emitted != len(materialized):
        cyclic = sorted(
            job_id
            for job_id, degree in indegree.items()
            if degree > 0
        )
        raise ValueError(f"ARTIFACT_GRAPH_CYCLE: {cyclic}")
    return materialized
