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
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEARCH_STOPWORDS = frozenset(
    {
        "minecraft",
        "fabric",
        "forge",
        "neoforge",
        "mod",
        "mods",
        "custom",
        "java",
        "system",
    }
)


def _tokens(value: str) -> set[str]:
    expanded = _CAMEL_BOUNDARY.sub(" ", str(value or ""))
    return {
        token.casefold()
        for token in _TOKEN_SPLIT.split(expanded)
        if len(token) > 2 and token.casefold() not in _SEARCH_STOPWORDS
    }


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
        max_seeds: int | None = None,
    ) -> tuple[CapabilitySeedEvidence, ...]:
        """Locate all evidence-bearing implementation seeds by default.

        ``max_seeds`` is an explicit operator resource control only.  The reuse
        pipeline deliberately leaves it unset so ordinal rank cannot turn an
        existing implementation into a false negative.
        """
        if max_seeds is not None and max_seeds < 1:
            raise ValueError("max_seeds must be positive when explicitly configured.")
        cap_tokens = _tokens(capability)
        search_terms = search_queries_for_capability(capability)
        term_tokens = {
            token
            for query in search_terms
            for token in _tokens(query)
        }
        all_tokens = cap_tokens | term_tokens

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
            symbol_tokens = _tokens(sym)
            if symbol_tokens & all_tokens:
                for p in paths:
                    scores[p] = scores.get(p, 0.0) + 7.0
                    ev_types.setdefault(p, []).append("symbol_declaration_match")

        # 4. Method declarations and API call-sites are structural body evidence.
        # They locate implementations whose class and path names do not describe the
        # gameplay feature (for example, an offer engine hidden in ``HandlerA``).
        for method, paths in index.method_to_paths.items():
            if _tokens(method) & all_tokens:
                for path in paths:
                    scores[path] = scores.get(path, 0.0) + 6.0
                    ev_types.setdefault(path, []).append("method_signature_match")
        for call, paths in index.api_call_to_paths.items():
            if _tokens(call) & all_tokens:
                for path in paths:
                    scores[path] = scores.get(path, 0.0) + 4.0
                    ev_types.setdefault(path, []).append("api_callsite_match")

        # 5. Resource logical ID match (+6)
        for res_id, path in index.resource_to_path.items():
            res_lower = res_id.lower()
            if any(tok in res_lower for tok in all_tokens):
                scores[path] = scores.get(path, 0.0) + 6.0
                ev_types.setdefault(path, []).append("resource_dependency_match")

        # 6. Path match (+2 diagnostic only)
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

        ranked.sort(key=lambda s: (-s.score, s.node_id))
        if max_seeds is None:
            return tuple(ranked)
        return tuple(ranked[:max_seeds])
