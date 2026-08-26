from __future__ import annotations

"""Evidence-Driven Capability Implementation Locator.

Replaces naive path-token fuzzy scoring with multi-layered structural evidence:
registry identifiers, API call-sites, FQCN/type hierarchies, method signatures,
resource references, and test suites.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonical_capability_ontology import search_queries_for_capability
from .repository_artifact_index import RepositoryArtifactIndex

_TOKEN_SPLIT = re.compile(r"[._\-/ ]+")


@dataclass(frozen=True)
class CapabilitySeedEvidence:
    capability: str
    node_id: str
    score: float
    evidence_types: tuple[str, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "node_id": self.node_id,
            "score": round(self.score, 4),
            "evidence_types": list(self.evidence_types),
            "provenance": dict(self.provenance),
        }


class CapabilityImplementationLocator:
    """Locates capability implementation seed nodes using structured evidence."""

    @classmethod
    def locate_seeds(
        cls,
        capability: str,
        index: RepositoryArtifactIndex,
        *,
        max_seeds: int = 8,
    ) -> tuple[CapabilitySeedEvidence, ...]:
        """Locate candidate implementation seeds using multi-evidence scoring."""
        cap_tokens = [t.lower() for t in _TOKEN_SPLIT.split(capability) if len(t) > 2]
        search_terms = search_queries_for_capability(capability)
        term_tokens = [t.lower() for query in search_terms for t in _TOKEN_SPLIT.split(query) if len(t) > 2]
        all_tokens = set(cap_tokens + term_tokens)

        scores: dict[str, float] = {}
        ev_types: dict[str, list[str]] = {}

        # 1. Exact registry ID match (+10)
        for reg_id, path in index.registry_to_path.items():
            reg_norm = reg_id.lower()
            if any(tok in reg_norm for tok in all_tokens):
                scores[path] = scores.get(path, 0.0) + 10.0
                ev_types.setdefault(path, []).append("registry_id_match")

        # 2. FQCN / Type hierarchy match (+8)
        for fqcn, path in index.fqcn_to_path.items():
            fqcn_parts = [p.lower() for p in _TOKEN_SPLIT.split(fqcn)]
            matched_count = sum(1 for tok in all_tokens if tok in fqcn_parts)
            if matched_count > 0:
                scores[path] = scores.get(path, 0.0) + (8.0 * min(matched_count, 3))
                ev_types.setdefault(path, []).append("fqcn_type_match")

        # 3. Symbol index match (+7)
        for sym, paths in index.symbol_to_paths.items():
            sym_lower = sym.lower()
            if any(tok == sym_lower or tok in sym_lower for tok in all_tokens):
                for p in paths:
                    scores[p] = scores.get(p, 0.0) + 7.0
                    ev_types.setdefault(p, []).append("symbol_declaration_match")

        # 4. Resource logical ID match (+6)
        for res_id, path in index.resource_to_path.items():
            res_lower = res_id.lower()
            if any(tok in res_lower for tok in all_tokens):
                scores[path] = scores.get(path, 0.0) + 6.0
                ev_types.setdefault(path, []).append("resource_dependency_match")

        # 5. Path match (+2 diagnostic only)
        for path in index.files_by_path:
            p_lower = path.lower()
            if any(tok in p_lower for tok in all_tokens):
                scores[path] = scores.get(path, 0.0) + 2.0
                ev_types.setdefault(path, []).append("path_token_match")

        # Filter out seeds that ONLY have path token match with low score
        ranked: list[CapabilitySeedEvidence] = []
        for path, score in scores.items():
            types = tuple(dict.fromkeys(ev_types.get(path, [])))
            # Require at least one structural evidence type if score is low
            if types == ("path_token_match",) and score < 4.0:
                continue
            ranked.append(
                CapabilitySeedEvidence(
                    capability=capability,
                    node_id=path,
                    score=score,
                    evidence_types=types,
                    provenance={"path": path},
                )
            )

        ranked.sort(key=lambda s: s.score, reverse=True)
        return tuple(ranked[:max_seeds])
