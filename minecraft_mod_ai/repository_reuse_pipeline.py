from __future__ import annotations

"""Grounded-RAG to source-reuse bridge.

The pre-design model never selects donor code.  The host takes only repositories whose
source bodies were actually materialized into grounded evidence cards, binds them to the
frozen capability graph, and sends every semantically admissible repository through the
immutable source-transplant inspector.  A donor remains an adaptation input until later
compile/integration gates prove it inside the generated project.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from .ecosystem_discovery import EcosystemDiscoveryClient
from .platform_catalog import PlatformAdapter, adapter_for_target
from .source_transplant import DonorSlice, inspect_repository_slice

_SCHEMA = "mmm/grounded-repository-reuse-plan-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+|[가-힣]{2,}", re.IGNORECASE)
_STOP = frozenset(
    {
        "minecraft",
        "fabric",
        "mod",
        "mods",
        "system",
        "feature",
        "implementation",
        "source",
        "code",
        "the",
        "and",
        "for",
        "with",
        "from",
    }
)


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"[_./:+-]+", " ", str(value or "").casefold())
    return {
        token
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token not in _STOP
    }


def _github_repository(source_id: str, source_url: str) -> str:
    source_id = str(source_id or "").strip()
    if source_id.casefold().startswith("github:"):
        value = source_id.split(":", 1)[1].strip().removesuffix(".git")
        if value.count("/") == 1:
            return value
    try:
        parsed = urlparse(str(source_url or "").strip())
    except ValueError:
        return ""
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def _grounded_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    research = design.get("_pre_design_research")
    if not isinstance(research, Mapping):
        return ()
    notes = research.get("domain_notes")
    if not isinstance(notes, list):
        return ()
    cards: list[dict[str, Any]] = []
    by_repository: dict[str, int] = {}
    for note in notes:
        if not isinstance(note, Mapping):
            continue
        raw_cards = note.get("grounded_evidence_cards")
        if not isinstance(raw_cards, list):
            continue
        for raw in raw_cards:
            if not isinstance(raw, Mapping):
                continue
            repository = _github_repository(
                str(raw.get("source_id") or ""),
                str(raw.get("source_url") or ""),
            )
            if not repository:
                continue
            evidence_text = " ".join(
                str(raw.get(key) or "")
                for key in ("source_title", "exact_excerpt")
            )
            item = {
                "repository": repository,
                "page_refs": [str(raw.get("page_ref") or "")],
                "source_ids": [str(raw.get("source_id") or "")],
                "source_urls": [str(raw.get("source_url") or "")],
                "evidence_text": evidence_text,
                "evidence_tokens": sorted(_tokens(evidence_text)),
            }
            key = repository.casefold()
            existing_index = by_repository.get(key)
            if existing_index is None:
                by_repository[key] = len(cards)
                cards.append(item)
                continue
            existing = cards[existing_index]
            existing["page_refs"] = list(
                dict.fromkeys([*existing["page_refs"], *item["page_refs"]])
            )
            existing["source_ids"] = list(
                dict.fromkeys([*existing["source_ids"], *item["source_ids"]])
            )
            existing["source_urls"] = list(
                dict.fromkeys([*existing["source_urls"], *item["source_urls"]])
            )
            existing["evidence_text"] = (
                f"{existing['evidence_text']} {item['evidence_text']}"
            ).strip()
            existing["evidence_tokens"] = sorted(_tokens(existing["evidence_text"]))
    return tuple(cards)


def _frozen_graph(design: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    plan = design.get("_pre_retrieval_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("Repository reuse requires the frozen pre-retrieval plan.")
    plan_sha256 = str(plan.get("plan_sha256") or "").strip()
    graph = plan.get("capability_graph")
    if not plan_sha256 or not isinstance(graph, Mapping):
        raise ValueError("Frozen pre-retrieval plan has no capability graph receipt.")
    payload = dict(graph)
    payload["source_plan_sha256"] = plan_sha256
    return payload, plan_sha256


def _adapter(design: Mapping[str, Any]) -> PlatformAdapter:
    selection = design.get("_platform_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Repository reuse requires a resolved platform selection.")
    target = selection.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("Platform selection has no resolved target coordinates.")
    version = str(target.get("minecraft_version") or "").strip()
    loader = str(target.get("loader") or "").strip().casefold()
    if not version or not loader:
        raise ValueError("Repository reuse target coordinates are incomplete.")
    return adapter_for_target(version, loader)


def _search_terms(graph: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    raw = graph.get("search_terms")
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, Mapping):
            continue
        capability = str(item.get("capability") or "").strip()
        terms = item.get("terms")
        values = tuple(
            dict.fromkeys(
                str(term).strip()
                for term in terms
                if str(term).strip()
            )
        ) if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes, bytearray)) else ()
        if capability:
            result[capability] = values
    return result


def _candidate_overlap(
    capability: str,
    terms: Sequence[str],
    card: Mapping[str, Any],
) -> int:
    wanted = _tokens(" ".join((capability, *terms)))
    available = set(str(item) for item in card.get("evidence_tokens", ()))
    return len(wanted & available)


def _donor_rank(donor: DonorSlice) -> tuple[int, float, float, str]:
    exact = 1 if donor.target_compatibility == "exact" else 0
    return (
        exact,
        float(donor.confidence),
        -float(donor.adaptation_cost),
        donor.repository.casefold(),
    )


def build_repository_reuse_plan(
    design: Mapping[str, Any],
    *,
    discovery_client: EcosystemDiscoveryClient | None = None,
) -> dict[str, Any]:
    """Compile grounded repository evidence into a donor-adaptation plan.

    There is no ordinal top-k.  Every repository that has a host-materialized evidence
    card and positive deterministic overlap with the frozen capability search intent is
    inspected.  Inspection failures are retained as rejection receipts rather than being
    converted into reuse claims.
    """

    graph, source_plan_sha256 = _frozen_graph(design)
    adapter = _adapter(design)
    adapter.validate()
    client = discovery_client or EcosystemDiscoveryClient()
    cards = _grounded_repository_cards(design)
    terms_by_capability = _search_terms(graph)
    capabilities = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in graph.get("nodes", ())
            if str(item).strip()
        )
    )
    if not capabilities:
        raise ValueError("Repository reuse requires at least one frozen capability.")

    decisions: list[dict[str, Any]] = []
    inspection_receipts: list[dict[str, Any]] = []
    for capability in capabilities:
        terms = terms_by_capability.get(capability, (capability.replace(".", " "),))
        admissible = [
            (card, _candidate_overlap(capability, terms, card))
            for card in cards
        ]
        admissible = [item for item in admissible if item[1] > 0]
        donors: list[DonorSlice] = []
        for card, overlap in admissible:
            repository = str(card["repository"])
            try:
                donor = inspect_repository_slice(
                    repository=repository,
                    capability=capability,
                    adapter=adapter,
                    discovery_client=client,
                )
            except Exception as exc:  # provider/transport failure is an explicit rejection receipt
                inspection_receipts.append(
                    {
                        "capability": capability,
                        "repository": repository,
                        "page_refs": list(card.get("page_refs", ())),
                        "overlap": overlap,
                        "status": "inspection_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if (
                donor is None
                or not donor.closure_complete
                or donor.target_compatibility not in {"exact", "adapt"}
            ):
                inspection_receipts.append(
                    {
                        "capability": capability,
                        "repository": repository,
                        "page_refs": list(card.get("page_refs", ())),
                        "overlap": overlap,
                        "status": "rejected_unverified_or_incomplete",
                    }
                )
                continue
            donors.append(donor)
            inspection_receipts.append(
                {
                    "capability": capability,
                    "repository": repository,
                    "page_refs": list(card.get("page_refs", ())),
                    "overlap": overlap,
                    "status": "admissible_source_transplant",
                    "commit_sha": donor.commit_sha,
                    "license_id": donor.license_id,
                    "target_compatibility": donor.target_compatibility,
                    "closure_complete": donor.closure_complete,
                    "file_count": len(donor.files),
                }
            )

        selected = max(donors, key=_donor_rank) if donors else None
        if selected is None:
            decisions.append(
                {
                    "capability": capability,
                    "mode": "fresh",
                    "source_id": "",
                    "component_refs": [],
                    "rationale": (
                        "No host-grounded repository candidate passed immutable commit, "
                        "license, target compatibility, capability seed, and dependency-closure gates."
                    ),
                }
            )
            continue
        donor_payload = selected.to_dict()
        decisions.append(
            {
                "capability": capability,
                "mode": "source_transplant",
                "source_id": f"host-donor:{selected.repository}@{selected.commit_sha}",
                "component_refs": [],
                "donor": donor_payload,
                "rationale": (
                    "Grounded RAG source was pinned and passed host source-transplant "
                    "license, target, immutable-file, capability-seed, and dependency-closure gates. "
                    "It remains adaptation input until target compile/integration validates it."
                ),
            }
        )

    return {
        "schema_version": _SCHEMA,
        "source_plan_sha256": source_plan_sha256,
        "capability_graph": graph,
        "capabilities": decisions,
        "grounded_repository_count": len(cards),
        "inspection_receipts": inspection_receipts,
        "selection_policy": (
            "positive frozen-intent overlap -> inspect every admissible grounded repository -> "
            "hard source-transplant gates -> deterministic exact/confidence/cost tie-break"
        ),
    }


__all__ = [
    "build_repository_reuse_plan",
]
