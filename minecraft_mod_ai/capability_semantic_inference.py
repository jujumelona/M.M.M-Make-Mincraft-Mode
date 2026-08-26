from __future__ import annotations

"""Host-Governed Semantic Inference for Unresolved Capability Concepts.

When user prompts contain novel or domain-specific concepts not present in the canonical
ontology (e.g., "은행 대출", "초전도체 자기부상", "워프 드라이브"), this module proposes
atomic sub-capabilities marked with the 'provisional:*' namespace and explicit dependency edges.

The model proposes atomic decompositions, but the host strictly owns capability IDs,
edges, and provenance. All provisional nodes remain distinct from canonical verified capabilities.
"""

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .canonical_capability_ontology import (
    CapabilityResolutionNode,
    CapabilityResolution,
    romanize_korean_universal,
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


def infer_provisional_capabilities(
    unresolved_spans: Sequence[str],
    *,
    router: Any = None,
) -> tuple[CapabilityResolutionNode, ..., tuple[tuple[str, str], ...]]:
    """Infer provisional atomic capabilities and dependency edges for unresolved concept spans."""
    inferred_nodes: list[CapabilityResolutionNode] = []
    inferred_edges: list[tuple[str, str]] = []

    for span in unresolved_spans:
        clean = str(span or "").strip()
        if not clean:
            continue
        # Deterministic atomic breakdown of unresolved concept
        raw_slug = romanize_korean_universal(clean)
        slug = re.sub(r"[^a-z0-9_]+", "_", raw_slug.casefold()).strip("_")
        slug = re.sub(r"_+", "_", slug)
        if not slug:
            slug = "unresolved_feature"

        primary_id = f"provisional:{slug[:40]}"
        state_id = f"provisional:{slug[:30]}.state"

        inferred_nodes.append(
            CapabilityResolutionNode(
                capability_id=primary_id,
                source_span=clean,
                origin="provisional_inferred",
                confidence=0.80,
                is_required=True,
            )
        )
        inferred_nodes.append(
            CapabilityResolutionNode(
                capability_id=state_id,
                source_span=clean,
                origin="dependency_required",
                confidence=0.75,
                is_required=False,
            )
        )

        inferred_edges.append((primary_id, state_id))
        inferred_edges.append((primary_id, "persistence.state_store"))
        inferred_edges.append((primary_id, "network.action_sync"))

    return tuple(inferred_nodes), tuple(inferred_edges)


def enrich_resolution_with_semantic_inference(
    resolution: CapabilityResolution,
    *,
    router: Any = None,
) -> CapabilityResolution:
    """Enrich a CapabilityResolution by expanding unresolved spans into provisional capability subgraphs."""
    if not resolution.unresolved_spans:
        return resolution

    inferred_nodes, inferred_edges = infer_provisional_capabilities(
        resolution.unresolved_spans,
        router=router,
    )

    # Filter out original unresolved:* placeholder nodes in favor of inferred provisional nodes
    retained_nodes = [
        node for node in resolution.nodes
        if not node.capability_id.startswith("unresolved:")
    ]
    all_nodes = tuple(dict.fromkeys([*retained_nodes, *inferred_nodes]))
    all_edges = tuple(dict.fromkeys([*resolution.edges, *inferred_edges]))

    return CapabilityResolution(
        nodes=all_nodes,
        edges=all_edges,
        unresolved_spans=resolution.unresolved_spans,
    )
