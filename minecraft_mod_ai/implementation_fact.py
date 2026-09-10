from __future__ import annotations

"""Unified implementation facts carrying provenance across planning, research, and design."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .prompt_fact_types import FactType, PromptFact


class FactProvenance(str, Enum):
    PROMPT = "prompt"
    RESEARCH = "research"
    DESIGN = "design"
    EXISTING_PROJECT = "existing_project"


@dataclass(frozen=True)
class ImplementationFact:
    """Atomic implementation fact with explicit source provenance."""

    fact_id: str
    fact_type: FactType
    subject: str
    object: str | None = None
    value: Any = None
    provenance: FactProvenance | str = FactProvenance.DESIGN
    evidence_refs: tuple[str, ...] = ()
    parent_requirement: str = ""
    display_name: str = ""
    source_clause: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": (
                self.fact_type.value
                if isinstance(self.fact_type, FactType)
                else str(self.fact_type)
            ),
            "subject": self.subject,
            "object": self.object,
            "value": self.value,
            "provenance": (
                self.provenance.value
                if isinstance(self.provenance, FactProvenance)
                else str(self.provenance)
            ),
            "evidence_refs": list(self.evidence_refs),
            "parent_requirement": self.parent_requirement,
            "display_name": self.display_name,
            "source_clause": self.source_clause,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationFact:
        raw_type = data["fact_type"]
        fact_type = FactType(raw_type) if isinstance(raw_type, str) else raw_type
        raw_prov = data.get("provenance", "design")
        prov = (
            FactProvenance(raw_prov)
            if isinstance(raw_prov, str) and raw_prov in FactProvenance.__members__.values()
            else raw_prov
        )
        return cls(
            fact_id=str(data["fact_id"]),
            fact_type=fact_type,
            subject=str(data["subject"]),
            object=data.get("object"),
            value=data.get("value"),
            provenance=prov,
            evidence_refs=tuple(data.get("evidence_refs", ())),
            parent_requirement=str(data.get("parent_requirement", "")),
            display_name=str(data.get("display_name", "")),
            source_clause=str(data.get("source_clause", "")),
        )


def prompt_fact_to_implementation_fact(fact: PromptFact) -> ImplementationFact:
    """Lift a user-facing PromptFact into an ImplementationFact with prompt provenance."""
    return ImplementationFact(
        fact_id=fact.fact_id,
        fact_type=fact.fact_type,
        subject=fact.subject,
        object=fact.object,
        value=fact.value,
        provenance=FactProvenance.PROMPT,
        source_clause=fact.source_clause,
    )
