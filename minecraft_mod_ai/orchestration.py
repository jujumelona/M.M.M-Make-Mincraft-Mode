"""Typed project graph and execution receipts for deterministic mod builds.

The planner may suggest work, but this module owns the executable dependency
graph.  Graph nodes reference code-owned capability names; retrieved documents
and model output cannot add tools or bypass approval.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from .spec import Proposal, SpecValidationError, canonical_json


NODE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
RECEIPT_STATUS_VALUES = frozenset({"succeeded", "failed", "blocked", "skipped"})


class TaskState(str, Enum):
    PLANNED = "planned"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TaskNode:
    node_id: str
    title: str
    capability: str | None
    depends_on: tuple[str, ...] = ()
    state: TaskState = TaskState.PLANNED
    approval_required: bool = False
    deterministic_gate: bool = False
    reason: str = ""

    def validate(self) -> None:
        if not NODE_ID_PATTERN.fullmatch(self.node_id):
            raise SpecValidationError(f"Invalid task node id: {self.node_id!r}")
        if not self.title.strip():
            raise SpecValidationError(f"Task title is missing: {self.node_id}")
        if self.capability is not None and not NODE_ID_PATTERN.fullmatch(
            self.capability
        ):
            raise SpecValidationError(
                f"Invalid task capability: {self.capability!r}"
            )
        if self.node_id in self.depends_on:
            raise SpecValidationError(f"Task cannot depend on itself: {self.node_id}")
        if self.state is TaskState.BLOCKED and not self.reason.strip():
            raise SpecValidationError(
                f"Blocked task must explain the missing implementation: {self.node_id}"
            )


@dataclass(frozen=True)
class TaskGraph:
    schema_version: str
    proposal_hash: str
    nodes: tuple[TaskNode, ...]

    def validate(self) -> None:
        if self.schema_version != "minecraft-mod-ai/task-dag-v2":
            raise SpecValidationError(
                f"Unsupported task graph schema: {self.schema_version!r}"
            )
        if not self.proposal_hash.startswith("sha256:"):
            raise SpecValidationError("Task graph proposal_hash must be SHA-256.")
        by_id: dict[str, TaskNode] = {}
        for node in self.nodes:
            node.validate()
            if node.node_id in by_id:
                raise SpecValidationError(f"Duplicate task node: {node.node_id}")
            by_id[node.node_id] = node
        for node in self.nodes:
            unknown = set(node.depends_on) - set(by_id)
            if unknown:
                raise SpecValidationError(
                    f"Task {node.node_id} has unknown dependencies: {sorted(unknown)}"
                )
        self.topological_order()

    def topological_order(self) -> tuple[TaskNode, ...]:
        by_id = {node.node_id: node for node in self.nodes}
        pending = {node_id: set(node.depends_on) for node_id, node in by_id.items()}
        ordered: list[TaskNode] = []
        ready = sorted(node_id for node_id, deps in pending.items() if not deps)
        while ready:
            node_id = ready.pop(0)
            if node_id not in pending:
                continue
            ordered.append(by_id[node_id])
            del pending[node_id]
            for candidate in sorted(pending):
                pending[candidate].discard(node_id)
                if not pending[candidate] and candidate not in ready:
                    ready.append(candidate)
            ready.sort()
        if pending:
            raise SpecValidationError(
                f"Task graph contains a dependency cycle: {sorted(pending)}"
            )
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_hash": self.proposal_hash,
            "nodes": [
                {
                    **asdict(node),
                    "state": node.state.value,
                    "depends_on": list(node.depends_on),
                }
                for node in self.nodes
            ],
        }

    @property
    def graph_hash(self) -> str:
        encoded = canonical_json(self.to_dict()).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class WorkerReceipt:
    """Machine-readable evidence emitted by one code-owned worker."""

    schema_version: str
    receipt_id: str
    node_id: str
    worker: str
    proposal_hash: str
    capability_manifest_hash: str
    status: str
    observed_at: str
    input_hash: str
    result_hash: str
    evidence: tuple[str, ...]
    exit_code: int | None = None
    error: str | None = None

    def validate(self) -> None:
        if self.schema_version != "minecraft-mod-ai/worker-receipt-v1":
            raise SpecValidationError("Unsupported worker receipt schema.")
        if not NODE_ID_PATTERN.fullmatch(self.node_id):
            raise SpecValidationError(f"Invalid receipt node id: {self.node_id!r}")
        if self.status not in RECEIPT_STATUS_VALUES:
            raise SpecValidationError(f"Invalid worker receipt status: {self.status}")
        for value_name, value in (
            ("proposal_hash", self.proposal_hash),
            ("capability_manifest_hash", self.capability_manifest_hash),
            ("input_hash", self.input_hash),
            ("result_hash", self.result_hash),
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise SpecValidationError(
                    f"Worker receipt {value_name} is not a SHA-256 value."
                )
        if self.status == "succeeded" and not self.evidence:
            raise SpecValidationError(
                "A successful worker receipt must contain external evidence."
            )
        if self.status == "failed" and not self.error:
            raise SpecValidationError("A failed worker receipt must contain an error.")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise SpecValidationError("Worker receipt exit_code must be an integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence": list(self.evidence),
        }


def compile_task_graph(proposal: Proposal) -> TaskGraph:
    """Compile a closed, content-sensitive DAG from one immutable proposal."""

    spec = proposal.spec
    nodes: list[TaskNode] = [
        TaskNode(
            "requirements.freeze",
            "Freeze the approved requirements and target profile",
            "plan.mod",
            deterministic_gate=True,
        ),
        TaskNode(
            "evidence.resolve",
            "Resolve version-locked primary evidence",
            "evidence.search",
            ("requirements.freeze",),
            deterministic_gate=True,
        ),
        TaskNode(
            "fabric.scaffold",
            "Create the pinned Fabric project",
            "fabric.scaffold",
            ("evidence.resolve",),
            approval_required=True,
        ),
    ]
    generation_dependencies: list[str] = ["fabric.scaffold"]
    if spec.contents:
        nodes.append(
            TaskNode(
                "fabric.content.generate",
                "Generate registered content and data resources",
                "fabric.scaffold",
                ("fabric.scaffold",),
                approval_required=True,
            )
        )
        generation_dependencies.append("fabric.content.generate")
    if spec.boss is not None:
        nodes.append(
            TaskNode(
                "fabric.entity.generate",
                "Generate the explicitly requested entity implementation",
                "fabric.scaffold",
                ("fabric.scaffold",),
                approval_required=True,
            )
        )
        generation_dependencies.append("fabric.entity.generate")
    for request in proposal.deferred_requests:
        safe_capability = re.sub(r"[^a-z0-9_.-]+", "-", request.capability.lower())
        safe_capability = safe_capability.strip(".-") or "unknown"
        nodes.append(
            TaskNode(
                f"deferred.{safe_capability}",
                f"Deferred requested capability: {request.capability}",
                None,
                ("requirements.freeze",),
                state=TaskState.BLOCKED,
                reason=request.reason,
            )
        )

    nodes.extend(
        (
            TaskNode(
                "quality.source.validate",
                "Validate schemas, paths, resources and request fidelity",
                "quality.validate",
                tuple(sorted(set(generation_dependencies))),
                approval_required=True,
                deterministic_gate=True,
            ),
            TaskNode(
                "build.gradle",
                "Compile and assemble with the pinned Gradle profile",
                "build.gradle",
                ("quality.source.validate",),
                approval_required=True,
                deterministic_gate=True,
            ),
            TaskNode(
                "test.gametest",
                "Run the generated unit tests and server GameTests",
                "test.gametest",
                ("build.gradle",),
                approval_required=True,
                deterministic_gate=True,
            ),
            TaskNode(
                "quality.jar.validate",
                "Inspect the built JAR and verify required contents",
                "quality.validate",
                ("test.gametest",),
                approval_required=True,
                deterministic_gate=True,
            ),
            TaskNode(
                "release.package",
                "Package source, evidence, provenance and verified binary",
                "release.package",
                ("quality.jar.validate",),
                approval_required=True,
                deterministic_gate=True,
            ),
        )
    )
    graph = TaskGraph(
        schema_version="minecraft-mod-ai/task-dag-v2",
        proposal_hash=proposal.calculate_hash(),
        nodes=tuple(nodes),
    )
    graph.validate()
    return graph


def project_ir(proposal: Proposal) -> dict[str, Any]:
    graph = compile_task_graph(proposal)
    spec = proposal.spec
    return {
        "schema_version": "minecraft-mod-ai/project-ir-v2",
        "project_id": spec.mod_id,
        "spec_hash": proposal.calculate_hash(),
        "platform_lock": asdict(spec.platform),
        "evidence_snapshot_hash": proposal.evidence_snapshot_hash,
        "capability_manifest_hash": proposal.capability_manifest_hash,
        "imported_source_snapshot_hash": (
            proposal.imported_source_snapshot_hash or None
        ),
        "registry": {
            "content_ids": [content.content_id for content in spec.contents],
            "boss_id": spec.boss.entity_id if spec.boss else None,
        },
        "task_graph_hash": graph.graph_hash,
        "task_dag": graph.to_dict(),
    }


def make_worker_receipt(
    *,
    node_id: str,
    worker: str,
    proposal: Proposal,
    result: dict[str, Any],
    evidence: Iterable[str],
    status: str,
    exit_code: int | None = None,
    error: str | None = None,
) -> WorkerReceipt:
    observed_at = datetime.now(timezone.utc).isoformat()
    input_payload = {
        "node_id": node_id,
        "proposal_hash": proposal.calculate_hash(),
        "capability_manifest_hash": proposal.capability_manifest_hash,
    }
    input_hash = "sha256:" + hashlib.sha256(
        canonical_json(input_payload).encode("utf-8")
    ).hexdigest()
    result_hash = "sha256:" + hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    receipt_seed = canonical_json(
        {
            **input_payload,
            "worker": worker,
            "status": status,
            "observed_at": observed_at,
            "result_hash": result_hash,
        }
    ).encode("utf-8")
    receipt = WorkerReceipt(
        schema_version="minecraft-mod-ai/worker-receipt-v1",
        receipt_id="sha256:" + hashlib.sha256(receipt_seed).hexdigest(),
        node_id=node_id,
        worker=worker,
        proposal_hash=proposal.calculate_hash(),
        capability_manifest_hash=proposal.capability_manifest_hash,
        status=status,
        observed_at=observed_at,
        input_hash=input_hash,
        result_hash=result_hash,
        evidence=tuple(evidence),
        exit_code=exit_code,
        error=error,
    )
    receipt.validate()
    return receipt


def receipt_json_line(receipt: WorkerReceipt) -> str:
    receipt.validate()
    return json.dumps(
        receipt.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "TaskGraph",
    "TaskNode",
    "TaskState",
    "WorkerReceipt",
    "compile_task_graph",
    "make_worker_receipt",
    "project_ir",
    "receipt_json_line",
]
