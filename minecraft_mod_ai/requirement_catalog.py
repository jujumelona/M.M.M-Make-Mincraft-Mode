from __future__ import annotations

"""Requirement Catalog and Structured Capability Specification Engine.

Preserves verbatim user requirements (REQ-001..N) and maps them cleanly into
atomic, verifiable CapabilitySpec nodes with full input/output/state/dependency/acceptance tracking.
"""

from dataclasses import dataclass
from typing import Any

from .canonical_capability_ontology import CapabilityResolution


@dataclass(frozen=True)
class RequirementSpec:
    id: str
    statement: str
    mandatory: bool = True
    provides: tuple[str, ...] = ()
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "mandatory": self.mandatory,
            "provides": list(self.provides),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    source_requirement_ids: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    persistence_requirement: bool = False
    networking_requirement: bool = False
    acceptance_requirements: tuple[str, ...] = ()
    search_intents: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_requirement_ids": list(self.source_requirement_ids),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "state": list(self.state),
            "side_effects": list(self.side_effects),
            "dependencies": list(self.dependencies),
            "persistence_requirement": self.persistence_requirement,
            "networking_requirement": self.networking_requirement,
            "acceptance_requirements": list(self.acceptance_requirements),
            "search_intents": list(self.search_intents),
        }


@dataclass(frozen=True)
class RequirementCatalog:
    requirements: tuple[RequirementSpec, ...] = ()
    capabilities: tuple[CapabilitySpec, ...] = ()

    def get_mandatory_requirements(self) -> tuple[RequirementSpec, ...]:
        return tuple(r for r in self.requirements if r.mandatory)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "capabilities": [c.to_dict() for c in self.capabilities],
        }


def build_requirement_catalog(
    prompt: str,
    resolution: CapabilityResolution,
) -> RequirementCatalog:
    """Build a formal RequirementCatalog mapping natural user prompt into atomic CapabilitySpecs."""
    reqs: list[RequirementSpec] = []
    caps: list[CapabilitySpec] = []

    # Extract user sentences / clauses as base requirement specifications
    sentences = [s.strip() for s in prompt.replace("\r", "\n").split("\n") if s.strip()]
    if not sentences:
        sentences = [prompt.strip()] if prompt.strip() else ["Default Mod Feature Requirement"]

    primary_req_id = "REQ-001"
    for i, sent in enumerate(sentences, 1):
        req_id = f"REQ-{i:03d}"
        if i == 1:
            primary_req_id = req_id
        matched_caps = tuple(n.capability_id for n in resolution.nodes) if i == 1 else ()
        reqs.append(
            RequirementSpec(
                id=req_id,
                statement=sent,
                mandatory=True,
                provides=matched_caps,
            )
        )

    # Convert resolution capability nodes into rich CapabilitySpecs
    dep_map: dict[str, list[str]] = {}
    for u, v in resolution.edges:
        dep_map.setdefault(u, []).append(v)

    for node in resolution.nodes:
        cap_id = node.capability_id
        node_deps = tuple(dep_map.get(cap_id, ()))
        is_state = "state" in cap_id or "persistence" in cap_id
        is_net = "network" in cap_id or "sync" in cap_id

        search_intents = (
            f"minecraft {cap_id.replace('.', ' ')} mod",
            f"{node.origin} {cap_id.split('.')[-1]} github",
        )

        caps.append(
            CapabilitySpec(
                id=cap_id,
                source_requirement_ids=(primary_req_id,),
                inputs=(),
                outputs=(),
                state=(cap_id,) if is_state else (),
                side_effects=(),
                dependencies=node_deps,
                persistence_requirement=is_state,
                networking_requirement=is_net,
                acceptance_requirements=(f"REQ-{cap_id.upper().replace('.', '-')}",),
                search_intents=search_intents,
            )
        )

    return RequirementCatalog(requirements=tuple(reqs), capabilities=tuple(caps))
