import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical_capability_ontology import (
    CapabilityResolution,
    resolve_capabilities_from_phrase_structured,
)


_IMPLEMENTATION_PRIMITIVE_PREFIXES = (
    "block_entity.",
    "packet.",
    "registry.",
    "screen.",
)


@dataclass(frozen=True)
class RequirementSpec:
    id: str
    statement: str
    original_span: str = ""
    normalized_statement: str = ""
    mandatory: bool = True
    provides: tuple[str, ...] = ()
    confidence: float = 1.0
    status: str = "RESOLVED"  # "RESOLVED" | "UNRESOLVED"
    gameplay_capabilities: tuple[str, ...] = ()
    implementation_capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "original_span": self.original_span or self.statement,
            "normalized_statement": self.normalized_statement or self.statement,
            "mandatory": self.mandatory,
            "provides": list(self.provides),
            "confidence": self.confidence,
            "status": self.status,
            "gameplay_capabilities": list(
                self.gameplay_capabilities or self.provides
            ),
            "implementation_capabilities": list(self.implementation_capabilities),
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

    def validate_coverage_invariants(self) -> tuple[bool, list[str]]:
        """Verify that every mandatory requirement has at least one bound capability or explicit UNRESOLVED status."""
        violations: list[str] = []
        for req in self.requirements:
            if req.mandatory and not req.provides and req.status != "UNRESOLVED":
                violations.append(f"Requirement {req.id} ({req.statement!r}) is mandatory but has 0 bound capabilities")
        return len(violations) == 0, violations

    def validate_semantic_bindings(
        self,
        *,
        prompt: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Reject technical primitives posing as user-facing requirements.

        Coverage alone only proves that a row has *some* capability.  This pass
        requires a gameplay-level binding for every mandatory source span and
        rejects a whole multi-action prompt collapsed onto one implementation API.
        Unknown mandatory text remains explicitly unresolved; callers can use
        ``generation_ready`` to prevent code generation from starting.
        """

        violations: list[str] = []
        if prompt and len(self.get_mandatory_requirements()) == 1:
            clauses = _split_into_semantic_clauses(prompt)
            if len(clauses) > 1:
                violations.append("COMPOSITE_PROMPT_COLLAPSED_TO_ONE_REQUIREMENT")
        for requirement in self.get_mandatory_requirements():
            if requirement.status == "UNRESOLVED":
                violations.append(f"MANDATORY_UNRESOLVED: {requirement.id}")
                continue
            gameplay = tuple(
                capability
                for capability in (
                    requirement.gameplay_capabilities or requirement.provides
                )
                if not capability.casefold().startswith(
                    _IMPLEMENTATION_PRIMITIVE_PREFIXES
                )
            )
            if not gameplay:
                violations.append(
                    f"GAMEPLAY_CAPABILITY_REQUIRED: {requirement.id}"
                )
                continue
            resolved = resolve_capabilities_from_phrase_structured(
                requirement.original_span or requirement.statement
            )
            expected = {
                node.capability_id
                for node in resolved.nodes
                if node.origin == "explicit"
                and not node.capability_id.startswith("unresolved:")
            }
            if expected and not expected.intersection(gameplay):
                violations.append(
                    f"SEMANTIC_CAPABILITY_LOST: {requirement.id}"
                )
        return not violations, violations

    @property
    def generation_ready(self) -> bool:
        covered, _ = self.validate_coverage_invariants()
        semantic, _ = self.validate_semantic_bindings()
        return covered and semantic

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "capabilities": [c.to_dict() for c in self.capabilities],
        }


def _split_into_semantic_clauses(text: str) -> list[str]:
    """Split natural language prompt into distinct semantic requirement clauses.

    Only language-independent structural boundaries are used:
    - Newlines / list items
    - Comma and semicolon
    - Bullet characters and arrows

    Semantic sub-clause splitting within a sentence (conjunction-based) is left
    to the model layer; the host deterministic parser only handles explicit
    authored boundaries that are valid in every script.
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    # Split on explicit structural boundaries (newlines, bullets, numbered items)
    lines = [line.strip() for line in cleaned.replace("\r", "\n").split("\n") if line.strip()]
    clauses: list[str] = []

    # Language-neutral intra-line delimiters: comma, semicolon, bullet, arrow
    pattern = re.compile(
        r"(?<=[^\s])(?:\s*,\s*|\s*;\s*|\s*[•→\u2022\u25b6\u25cf\u2013\u2014]\s*|\s+->\s*|\s+=>\s*)(?=[^\s])",
        re.UNICODE,
    )

    for line in lines:
        # Strip leading list markers (e.g. "1. ", "- ", "• ")
        line = re.sub(r"^\s*(?:\d+\.|[-*•])\s+", "", line)
        parts = [part.strip() for part in pattern.split(line) if part.strip()]
        if parts:
            clauses.extend(parts)
        else:
            clauses.append(line)

    return clauses if clauses else [cleaned]


def _evidence_capability_id(value: Any) -> str:
    """Convert a provides receipt back to its authored capability identity."""

    text = str(value or "").strip()
    if text.casefold().startswith("capability:"):
        text = text[len("capability:") :].strip()
    return text


def build_requirement_catalog(
    prompt: str,
    resolution: CapabilityResolution | None = None,
    *,
    evidence_request_catalog: Mapping[str, Any] | None = None,
    explicit_capabilities: Sequence[str] | None = None,
) -> RequirementCatalog:
    """Build a formal RequirementCatalog with fine-grained span tracking and many-to-many capability bindings.

    If an authoritative evidence_request_catalog is provided, it is used as the single source of truth.
    """
    reqs: list[RequirementSpec] = []
    caps: list[CapabilitySpec] = []
    cap_to_reqs: dict[str, list[str]] = {}

    # Case 1: Authoritative evidence_request_catalog provided
    if evidence_request_catalog and isinstance(evidence_request_catalog.get("requirements"), Sequence):
        raw_reqs = evidence_request_catalog["requirements"]
        for i, r in enumerate(raw_reqs, 1):
            if not isinstance(r, Mapping):
                continue
            req_id = str(r.get("requirement_id") or r.get("id") or f"REQ-{i:03d}")
            stmt = str(r.get("statement") or r.get("description") or f"Requirement {req_id}")
            provides: list[str] = []
            seen_provides: set[str] = set()

            def add_provide(raw: Any) -> None:
                capability = _evidence_capability_id(raw)
                key = capability.casefold()
                if capability and key not in seen_provides:
                    seen_provides.add(key)
                    provides.append(capability)

            add_provide(r.get("capability"))
            raw_provides = r.get("provides")
            if isinstance(raw_provides, Sequence) and not isinstance(
                raw_provides, (str, bytes, bytearray)
            ):
                for provide in raw_provides:
                    add_provide(provide)

            source_span = r.get("source_span")
            original_span = (
                str(source_span.get("text") or stmt)
                if isinstance(source_span, Mapping)
                else str(r.get("original_span") or stmt)
            )

            req_spec = RequirementSpec(
                id=req_id,
                statement=stmt,
                original_span=original_span,
                normalized_statement=stmt,
                mandatory=bool(r.get("mandatory", True)),
                provides=tuple(provides),
                confidence=float(r.get("confidence", 1.0)),
                status="RESOLVED" if provides else "UNRESOLVED",
                gameplay_capabilities=tuple(
                    str(value)
                    for value in r.get("gameplay_capabilities", provides)
                    if str(value).strip()
                ),
                implementation_capabilities=tuple(
                    str(value)
                    for value in r.get("implementation_capabilities", ())
                    if str(value).strip()
                ),
            )
            reqs.append(req_spec)
            for c in provides:
                cap_to_reqs.setdefault(c, []).append(req_id)

    else:
        # Case 2: Parse from natural language prompt clauses
        clauses = _split_into_semantic_clauses(prompt)
        if not clauses:
            clauses = ["Default Mod Feature Requirement"]

        if resolution is None:
            resolution = resolve_capabilities_from_phrase_structured(prompt)

        # Build list of available resolution capability nodes
        nodes = list(resolution.nodes) if resolution else []
        if explicit_capabilities:
            for exp in explicit_capabilities:
                if not any(n.capability_id == exp for n in nodes):
                    from .canonical_capability_ontology import CapabilityResolutionNode
                    nodes.append(CapabilityResolutionNode(capability_id=exp, source_span=exp, origin="explicit"))

        for i, clause in enumerate(clauses, 1):
            req_id = f"REQ-{i:03d}"
            clause_low = clause.lower()

            # Match capabilities specifically relevant to this clause span
            matched: list[str] = []
            for node in nodes:
                span_low = (node.source_span or node.capability_id).lower()
                stem = node.capability_id.split(".")[-1].lower()
                if span_low in clause_low or stem in clause_low:
                    matched.append(node.capability_id)

            # If no specific match was found, and we only have 1 clause or this is the first clause, attach available nodes
            if not matched and len(clauses) == 1:
                matched = [n.capability_id for n in nodes]
            elif not matched:
                # Try structured sub-resolution on this exact clause
                clause_res = resolve_capabilities_from_phrase_structured(clause)
                matched = [n.capability_id for n in clause_res.nodes]

            status = "RESOLVED" if matched else "UNRESOLVED"
            req_spec = RequirementSpec(
                id=req_id,
                statement=clause,
                original_span=clause,
                normalized_statement=clause,
                mandatory=True,
                provides=tuple(dict.fromkeys(matched)),
                status=status,
                gameplay_capabilities=tuple(
                    node.capability_id
                    for node in nodes
                    if node.capability_id in matched
                    and node.origin == "explicit"
                ),
                implementation_capabilities=tuple(
                    node.capability_id
                    for node in nodes
                    if node.capability_id in matched
                    and node.origin == "dependency_required"
                ),
            )
            reqs.append(req_spec)
            for c in req_spec.provides:
                cap_to_reqs.setdefault(c, []).append(req_id)

    # Ensure all distinct capabilities referenced across requirements are reified into CapabilitySpecs
    all_cap_ids = list(dict.fromkeys(c for r in reqs for c in r.provides))
    if resolution:
        for n in resolution.nodes:
            if n.capability_id not in all_cap_ids:
                all_cap_ids.append(n.capability_id)

    dep_map: dict[str, list[str]] = {}
    if resolution:
        for u, v in resolution.edges:
            dep_map.setdefault(u, []).append(v)

    for cap_id in all_cap_ids:
        bound_req_ids = tuple(cap_to_reqs.get(cap_id, (reqs[0].id if reqs else "REQ-001",)))
        node_deps = tuple(dep_map.get(cap_id, ()))
        is_state = "state" in cap_id or "persistence" in cap_id
        is_net = "network" in cap_id or "sync" in cap_id

        search_intents = (
            f"minecraft {cap_id.replace('.', ' ')} mod",
            f"{cap_id.split('.')[-1]} github",
        )

        caps.append(
            CapabilitySpec(
                id=cap_id,
                source_requirement_ids=bound_req_ids,
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
