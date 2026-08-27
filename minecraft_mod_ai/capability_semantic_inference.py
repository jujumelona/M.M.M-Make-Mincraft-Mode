from __future__ import annotations

"""Host-governed semantic inference for unresolved capability concepts.

Unknown prompt concepts may be expanded into provisional capability nodes only when
an actual semantic router supplies evidence for that decomposition. The host owns
capability IDs, edges, and provenance; deterministic code must never invent
``primary/state/logic`` requirements merely because text was not recognized by the
canonical ontology.
"""

import hashlib
import json
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
        return slug[:64]
    return "semantic_" + hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12]


def _structured_capability_id(value: Any) -> str:
    """Preserve model-authored semantic IDs; hash only opaque descriptions."""

    text = str(value or "").strip().casefold().removeprefix("capability:")
    if re.fullmatch(r"[a-z][a-z0-9]*(?:[._:/-][a-z0-9_]+)+", text):
        return text
    slug = _proposal_slug(text)
    return f"provisional:{slug}"


def _router_payload(router: Any, span: str) -> Any:
    """Invoke either the production ``ModelRouter`` contract or a legacy callable.

    Production passes a ModelRouter instance, which is intentionally not callable.
    Treating only ``callable(router)`` as usable silently disabled semantic inference
    in reuse planning and caused opaque ``semantic_<hash>`` capabilities to leak into
    retrieval.  This adapter keeps one semantic contract across both call sites.
    """

    if router is None:
        return None
    generate_text = getattr(router, "generate_text", None)
    if callable(generate_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "Resolve the authored Minecraft mod requirement into independent "
                    "gameplay capabilities needed to preserve its meaning. Do not add "
                    "genre conventions or implementation choices that are not supported "
                    "by the requirement. Return JSON with a capabilities array. Each "
                    "capability may contain capability_id, source_span, description, "
                    "category, and dependencies."
                ),
            },
            {"role": "user", "content": span},
        ]
        raw = generate_text(
            "planner",
            messages,
            response_format="json",
            enable_tools=False,
        )
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    if callable(router):
        return router(
            "Decompose this Minecraft mod requirement into independent gameplay "
            "capabilities actually supported by the request: " + span
        )
    return None


def _proposal_items(payload: Any) -> tuple[Any, ...]:
    if isinstance(payload, Mapping):
        for key in (
            "capabilities",
            "requirements",
            "gameplay_capability_candidates",
            "proposals",
        ):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(value)
        return ()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(payload)
    return ()


def infer_provisional_capabilities(
    unresolved_spans: Sequence[str],
    *,
    router: Any = None,
) -> tuple[tuple[CapabilityResolutionNode, ...], tuple[tuple[str, str], ...]]:
    """Infer provisional capabilities only from explicit semantic-router evidence.

    Absence or failure of semantic evidence never authorizes deterministic synthetic
    children.  The caller can distinguish that state from a successful decomposition
    and must not turn it into an executable opaque retrieval capability.
    """

    inferred_nodes: list[CapabilityResolutionNode] = []
    inferred_edges: list[tuple[str, str]] = []

    if router is None:
        return (), ()

    for span in unresolved_spans:
        clean = str(span or "").strip()
        if not clean:
            continue

        proposals: list[ProvisionalCapabilityProposal] = []
        try:
            payload = _router_payload(router, clean)
            for item in _proposal_items(payload):
                if isinstance(item, str):
                    raw_name = item.strip()
                    item_map: Mapping[str, Any] = {}
                elif isinstance(item, Mapping):
                    item_map = item
                    raw_name = str(
                        item.get("capability_id")
                        or item.get("capability")
                        or item.get("name")
                        or ""
                    ).strip()
                else:
                    continue
                if not raw_name:
                    continue
                cap_id = _structured_capability_id(raw_name)
                raw_dependencies = item_map.get("dependencies", ())
                dependencies = tuple(
                    _structured_capability_id(dep)
                    for dep in raw_dependencies
                    if str(dep or "").strip()
                ) if isinstance(raw_dependencies, Sequence) and not isinstance(
                    raw_dependencies, (str, bytes, bytearray)
                ) else ()
                source_span = str(
                    item_map.get("source_span")
                    or item_map.get("source_text")
                    or clean
                ).strip() or clean
                description = str(item_map.get("description") or source_span).strip()
                category = str(item_map.get("category") or "custom").strip() or "custom"
                semantic_terms = cap_id.removeprefix("provisional:").replace(".", " ")
                proposals.append(
                    ProvisionalCapabilityProposal(
                        capability_id=cap_id,
                        source_span=source_span,
                        category=category,
                        description=description,
                        suggested_dependencies=dependencies,
                        search_queries=(
                            semantic_terms,
                            f"minecraft {semantic_terms}",
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

    return tuple(dict.fromkeys(inferred_nodes)), tuple(dict.fromkeys(inferred_edges))


def enrich_resolution_with_semantic_inference(
    resolution: CapabilityResolution,
    *,
    router: Any = None,
) -> CapabilityResolution:
    """Replace unresolved placeholders only with semantically evidenced proposals.

    ``unresolved:*`` nodes are parser bookkeeping, not executable capabilities, so
    they are never promoted into a reuse graph. If no router evidence exists, they
    remain unresolved metadata for the caller to gate rather than becoming fake
    searchable capabilities.
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

    resolved_spans = {
        node.source_span
        for node in inferred_nodes
        if str(node.source_span or "").strip()
    }
    remaining_unresolved = tuple(
        span
        for span in resolution.unresolved_spans
        if str(span or "").strip() not in resolved_spans
    )

    return CapabilityResolution(
        nodes=all_nodes,
        edges=all_edges,
        unresolved_spans=remaining_unresolved,
    )
