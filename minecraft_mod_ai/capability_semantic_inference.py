from __future__ import annotations

"""Host-governed semantic inference for unresolved capability concepts.

Unknown prompt concepts may be expanded into provisional capability nodes only when
an actual semantic router supplies evidence for that decomposition. The host owns
capability IDs, edges, and provenance; deterministic code must never invent
``primary/state/logic`` requirements merely because text was not recognized by the
canonical ontology.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical_capability_ontology import (
    CapabilityResolution,
    CapabilityResolutionNode,
)


@dataclass(frozen=True)
class ProvisionalCapabilityProposal:
    capability_id: str
    source_span: str
    category: str
    description: str
    suggested_dependencies: tuple[str, ...]
    search_queries: tuple[str, ...]

    def to_node(self) -> CapabilityResolutionNode:
        return CapabilityResolutionNode(
            capability_id=self.capability_id,
            source_span=self.source_span,
            origin="provisional_inferred",
            confidence=0.75,
            is_required=True,
        )


def _proposal_slug(value: str) -> str:
    """Return a stable ASCII identifier without making language-specific guesses."""

    clean = str(value or "").strip().casefold()
    slug = re.sub(r"[^a-z0-9_.-]+", "_", clean).strip("_.-")
    slug = re.sub(r"_+", "_", slug)
    if slug:
        return slug[:40]
    return "semantic_" + hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12]


def infer_provisional_capabilities(
    unresolved_spans: Sequence[str],
    *,
    router: Any = None,
) -> tuple[tuple[CapabilityResolutionNode, ...], tuple[tuple[str, str], ...]]:
    """Infer provisional capabilities only from explicit semantic-router evidence.

    In particular, the absence of a router is not evidence that an unknown phrase
    implies three synthetic primary/state/logic requirements. Returning no proposal
    keeps deterministic parsing conservative and prevents incidental prompt wording
    from expanding an otherwise complete design graph.
    """

    inferred_nodes: list[CapabilityResolutionNode] = []
    inferred_edges: list[tuple[str, str]] = []

    if not callable(router):
        return (), ()

    for span in unresolved_spans:
        clean = str(span or "").strip()
        if not clean:
            continue

        proposals: list[ProvisionalCapabilityProposal] = []
        try:
            response = router(
                "Decompose this Minecraft mod concept into atomic gameplay capabilities. "
                "Return only capabilities actually supported by the request: " + clean
            )
            if isinstance(response, Sequence) and not isinstance(
                response, (str, bytes, bytearray)
            ):
                for item in response:
                    if not isinstance(item, Mapping):
                        continue
                    raw_name = str(item.get("name") or "").strip()
                    if not raw_name:
                        continue
                    slug = _proposal_slug(raw_name)
                    cap_id = f"provisional:{slug}"
                    dependencies = tuple(
                        str(dep).strip()
                        for dep in item.get("dependencies", ())
                        if str(dep).strip()
                    )
                    proposals.append(
                        ProvisionalCapabilityProposal(
                            capability_id=cap_id,
                            source_span=clean,
                            category=str(item.get("category") or "custom"),
                            description=str(item.get("description") or clean),
                            suggested_dependencies=dependencies,
                            search_queries=(
                                f"{slug.replace('.', ' ')} mod",
                                f"minecraft {slug.replace('.', ' ')}",
                            ),
                        )
                    )
        except Exception:
            proposals = []

        for prop in proposals:
            inferred_nodes.append(prop.to_node())
            for dep in prop.suggested_dependencies:
                edge = (prop.capability_id, dep)
                if edge not in inferred_edges:
                    inferred_edges.append(edge)

    return tuple(inferred_nodes), tuple(inferred_edges)


def enrich_resolution_with_semantic_inference(
    resolution: CapabilityResolution,
    *,
    router: Any = None,
) -> CapabilityResolution:
    """Replace unresolved placeholders only with semantically evidenced proposals.

    ``unresolved:*`` nodes are parser bookkeeping, not executable capabilities, so
    they are never promoted into a reuse graph. If no router evidence exists, they
    simply disappear and the caller can rely on authoritative design/catalog nodes
    or its ordinary empty-graph fallback.
    """

    if not resolution.unresolved_spans:
        return resolution

    inferred_nodes, inferred_edges = infer_provisional_capabilities(
        resolution.unresolved_spans,
        router=router,
    )

    retained_nodes = [
        node
        for node in resolution.nodes
        if not node.capability_id.startswith("unresolved:")
    ]
    all_nodes = tuple(dict.fromkeys([*retained_nodes, *inferred_nodes]))
    all_edges = tuple(dict.fromkeys([*resolution.edges, *inferred_edges]))

    return CapabilityResolution(
        nodes=all_nodes,
        edges=all_edges,
        unresolved_spans=resolution.unresolved_spans,
    )
