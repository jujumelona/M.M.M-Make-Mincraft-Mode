from __future__ import annotations

"""Concrete ArtifactJob representation for leaf-level generation units."""

from dataclasses import dataclass, field
from typing import Any

from .implementation_identity import ExecutorType


@dataclass
class ArtifactJob:
    job_id: str
    template_id: str
    owner_module: str
    executor_type: ExecutorType = ExecutorType.TEMPLATE
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
        return cls(
            job_id=str(data["job_id"]),
            template_id=str(data["template_id"]),
            owner_module=str(data["owner_module"]),
            executor_type=ExecutorType(data.get("executor_type", "template")),
            target_path=str(data.get("target_path", "")),
            anchor=str(data.get("anchor", "")),
            operation=str(data.get("operation", "")),
            expected_sha256=data.get("expected_sha256"),
            requires=tuple(str(k) for k in data.get("requires", ())),
            produces=tuple(str(k) for k in data.get("produces", ())),
            required_ports=tuple(dict(k) for k in data.get("required_ports", ())),
            ai_slots=tuple(dict(s) for s in data.get("ai_slots", ())),
            deterministic_inputs=dict(data.get("deterministic_inputs", {})),
            status=str(data.get("status", "PENDING")),
            validation_receipts=list(data.get("validation_receipts", [])),
            rendered_output=str(data.get("rendered_output", "")),
            context_id=str(data.get("context_id", "")),
            canonical_leaf=str(data.get("canonical_leaf", "")),
            implementation_id=str(data.get("implementation_id", "")),
        )
