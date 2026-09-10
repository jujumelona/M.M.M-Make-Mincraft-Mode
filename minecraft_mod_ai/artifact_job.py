from __future__ import annotations

"""Concrete ArtifactJob representation for leaf-level generation units."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactJob:
    job_id: str
    template_id: str
    owner_module: str
    target_path: str = ""
    anchor: str = ""
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    ai_slots: tuple[dict[str, Any], ...] = ()
    deterministic_inputs: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    validation_receipts: list[dict[str, Any]] = field(default_factory=list)
    rendered_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "template_id": self.template_id,
            "owner_module": self.owner_module,
            "target_path": self.target_path,
            "anchor": self.anchor,
            "requires": list(self.requires),
            "produces": list(self.produces),
            "ai_slots": list(self.ai_slots),
            "deterministic_inputs": dict(self.deterministic_inputs),
            "status": self.status,
            "validation_receipts": list(self.validation_receipts),
            "rendered_output": self.rendered_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactJob":
        return cls(
            job_id=str(data["job_id"]),
            template_id=str(data["template_id"]),
            owner_module=str(data["owner_module"]),
            target_path=str(data.get("target_path", "")),
            anchor=str(data.get("anchor", "")),
            requires=tuple(str(k) for k in data.get("requires", ())),
            produces=tuple(str(k) for k in data.get("produces", ())),
            ai_slots=tuple(dict(s) for s in data.get("ai_slots", ())),
            deterministic_inputs=dict(data.get("deterministic_inputs", {})),
            status=str(data.get("status", "PENDING")),
            validation_receipts=list(data.get("validation_receipts", [])),
            rendered_output=str(data.get("rendered_output", "")),
        )
