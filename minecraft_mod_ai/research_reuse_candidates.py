from __future__ import annotations

"""Non-authoritative repository candidates projected from grounded RAG.

This module is deliberately host-only. It may discover that a materialized GitHub
record is worth *checking* as a donor, but it can never authorize source reuse. The
existing grounded_source_reuse owner remains responsible for immutable revision,
license, dependency/source closure, target compatibility, and compile proof.

Candidate projection is a recall stage. It therefore does not impose semantic top-N
limits: completion and candidate selection are decided later from evidence, not by an
arbitrary cardinality cut in retrieval.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from .source_intent_aliases import source_intent_terms

_SCHEMA = "mmm/repository-candidate-receipt-v1"
_MAX_EVIDENCE_CHARS = 640
_REPOSITORY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[가-힣]{2,}", re.IGNORECASE)
_STOP = frozenset({
    "minecraft", "fabric", "forge", "neoforge", "mod", "mods", "source", "code",
    "implementation", "system", "feature", "github", "https", "http", "www", "com",
    "the", "and", "for", "with", "from",
})


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _tokens(value: Any) -> set[str]:
    folded = re.sub(r"[_./:+-]+", " ", _text(value).casefold())
    return ({
        token
        for token in _TOKEN_RE.findall(folded)
        if len(token) > 1 and token not in _STOP
    } | (set(source_intent_terms(value)) - _STOP))


def _repository_from_value(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return ""
    if raw.casefold().startswith("github:"):
        raw = raw.split(":", 1)[1].strip()
    direct = raw.removesuffix(".git")
    if _REPOSITORY_ID_RE.fullmatch(direct):
        return direct
    try:
        parsed = urlparse(raw)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "github.com", "www.github.com"
    }:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    repository = f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return repository if _REPOSITORY_ID_RE.fullmatch(repository) else ""


def _repository_from_record(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    values = [
        record.get("repository"),
        metadata.get("repository") if isinstance(metadata, Mapping) else None,
        metadata.get("source_url") if isinstance(metadata, Mapping) else None,
        record.get("source_id"),
        record.get("url"),
        record.get("source_url"),
        record.get("html_url"),
    ]
    for value in values:
        repository = _repository_from_value(value)
        if repository:
            return repository
    return ""


def _record_url(record: Mapping[str, Any], repository: str) -> str:
    for key in ("url", "source_url", "html_url"):
        value = _text(record.get(key))
        if value and _repository_from_value(value):
            return value
    return f"https://github.com/{repository}"


def _evidence_text(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    parts = [
        record.get("title"),
        record.get("name"),
        record.get("description"),
        record.get("snippet"),
        metadata.get("description") if isinstance(metadata, Mapping) else None,
        record.get("content"),
    ]
    return _text(" ".join(_text(part) for part in parts if _text(part)))


def _catalog_evidence(record: Mapping[str, Any], query_sha: str) -> dict[str, Any]:
    """Carry the real pre-design evidence, never the requested feature as evidence."""
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    source_id = _text(record.get("source_id"))
    return {
        "provider": source_id.split(":", 1)[0] if ":" in source_id else "unknown",
        "source_id": source_id,
        "catalog_url": _text(record.get("url")),
        "source_url": _text(metadata.get("source_url") or record.get("source_url")),
        "title": _text(record.get("title")),
        "summary": _text(record.get("description") or metadata.get("description") or
                         record.get("content"))[:_MAX_EVIDENCE_CHARS],
        "content_sha256": _text(record.get("content_sha256")),
        "evidence_origin": _text(record.get("evidence_origin")),
        "body_retrieved": record.get("body_retrieved") is True,
        "query_sha256": query_sha,
        "source_reuse_authority": "verification_required",
    }


def _source_code_domain(domain: Mapping[str, Any]) -> bool:
    kinds = domain.get("evidence_kinds")
    labels = (
        {str(value) for value in kinds}
        if isinstance(kinds, Sequence)
        and not isinstance(kinds, (str, bytes, bytearray))
        else set()
    )
    if "source_code" in labels:
        return True
    providers = domain.get("providers")
    catalogues = (
        {str(value) for value in providers}
        if isinstance(providers, Sequence)
        and not isinstance(providers, (str, bytes, bytearray))
        else set()
    )
    # A catalogue-first Minecraft API research domain may already contain an
    # exact GitHub sourceUrl. Project THAT actual link without converting
    # the entire predesign domain into a broad GitHub search provider.
    return "minecraft_api" in labels and bool(catalogues & {"curseforge", "modrinth"})


def _optional_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Apply only an explicitly requested presentation/consumer limit.

    Normal planning retrieval passes ``None`` and retains every deduplicated candidate.
    This keeps backwards compatibility for callers that explicitly request a bounded
    preview without allowing a hidden default cap to alter research semantics.
    """
    if limit is None:
        return rows
    return rows[:max(0, int(limit))]


def project_repository_candidates(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
    *,
    per_domain_limit: int | None = None,
) -> list[dict[str, Any]]:
    """No GitHub donor candidates are admitted in CurseForge-only mode."""
    return []
    if not _source_code_domain(domain):
        return []
    domain_id = _text(domain.get("domain_id"))
    intent_text = " ".join(
        [
            _text(domain.get("objective")),
            " ".join(_text(v) for v in domain.get("requirements", []) if _text(v))
            if isinstance(domain.get("requirements"), list)
            else "",
            " ".join(_text(v) for v in domain.get("queries", []) if _text(v))
            if isinstance(domain.get("queries"), list)
            else "",
        ]
    )
    wanted = _tokens(intent_text)
    by_repo: dict[str, dict[str, Any]] = {}
    raw_queries = grounded.get("queries")
    for query_index, query in enumerate(raw_queries if isinstance(raw_queries, list) else []):
        if not isinstance(query, Mapping):
            continue
        query_sha = _text(query.get("query_sha256")) or _sha(_text(query.get("query")))
        records = query.get("evidence_records")
        for record_index, record in enumerate(records if isinstance(records, list) else []):
            if not isinstance(record, Mapping):
                continue
            repository = _repository_from_record(record)
            if not repository:
                continue
            evidence_text = _evidence_text(record)
            overlap = len(wanted & _tokens(evidence_text))
            # Semantic feature coverage dominates raw search position.
            # An Alien Pets search hit must not outrank a Space Combat hit just
            # because it appeared first in the provider's catalogue.
            rank_score = min(overlap, 8) * 1.0 + (1.0 / (1.0 + query_index + record_index))
            key = repository.casefold()
            source_id = _text(record.get("source_id")) or f"github:{repository}"
            source_url = _record_url(record, repository)
            provenance = _catalog_evidence(record, query_sha)
            current = by_repo.get(key)
            if current is None:
                by_repo[key] = {
                    "schema_version": _SCHEMA,
                    "repository": repository,
                    "domain_ids": [domain_id] if domain_id else [],
                    "source_ids": [source_id],
                    "source_urls": [source_url],
                    "query_sha256": [query_sha],
                    "evidence_text": evidence_text[:_MAX_EVIDENCE_CHARS],
                    "evidence_tokens": sorted(_tokens(evidence_text))[:64],
                    "catalog_evidence": [provenance],
                    "candidate_score": round(rank_score, 6),
                    "origin": "host_grounded_retrieval",
                    "reference_only": True,
                    "source_reuse_authority": "verification_required",
                    "explicit_reference": False,
                }
                continue
            current["candidate_score"] = max(float(current["candidate_score"]), round(rank_score, 6))
            if provenance not in current["catalog_evidence"]:
                current["catalog_evidence"].append(provenance)
            for field, value in (
                ("source_ids", source_id),
                ("source_urls", source_url),
                ("query_sha256", query_sha),
            ):
                if value and value not in current[field]:
                    current[field].append(value)
            merged_text = _text(f"{current['evidence_text']} {evidence_text}")
            current["evidence_text"] = merged_text[:_MAX_EVIDENCE_CHARS]
            current["evidence_tokens"] = sorted(_tokens(merged_text))[:64]

    ranked = sorted(
        by_repo.values(),
        key=lambda row: (-float(row.get("candidate_score", 0.0)), str(row.get("repository", "")).casefold()),
    )
    return _optional_limit(ranked, per_domain_limit)


def merge_repository_candidates(
    existing: Sequence[Mapping[str, Any]] | None,
    new: Sequence[Mapping[str, Any]] | None,
    *,
    global_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Merge every candidate receipt without ever upgrading its authority."""
    by_repo: dict[str, dict[str, Any]] = {}
    for raw in [*(existing or ()), *(new or ())]:
        if not isinstance(raw, Mapping):
            continue
        repository = _repository_from_value(raw.get("repository"))
        if not repository:
            continue
        if raw.get("reference_only") is not True or raw.get("source_reuse_authority") != "verification_required":
            continue
        key = repository.casefold()
        candidate = dict(raw)
        candidate["repository"] = repository
        candidate["reference_only"] = True
        candidate["source_reuse_authority"] = "verification_required"
        current = by_repo.get(key)
        if current is None:
            by_repo[key] = candidate
            continue
        current["candidate_score"] = max(
            float(current.get("candidate_score", 0.0) or 0.0),
            float(candidate.get("candidate_score", 0.0) or 0.0),
        )
        for field in ("domain_ids", "source_ids", "source_urls", "query_sha256", "catalog_evidence"):
            values = candidate.get(field)
            if not isinstance(values, list):
                continue
            target = current.setdefault(field, [])
            for value in values:
                if value and value not in target:
                    target.append(value)
        merged_text = _text(f"{current.get('evidence_text', '')} {candidate.get('evidence_text', '')}")
        current["evidence_text"] = merged_text[:_MAX_EVIDENCE_CHARS]
        current["evidence_tokens"] = sorted(_tokens(merged_text))[:64]
    ranked = sorted(
        by_repo.values(),
        key=lambda row: (-float(row.get("candidate_score", 0.0) or 0.0), str(row.get("repository", "")).casefold()),
    )
    return _optional_limit(ranked, global_limit)


def authored_design_repository_candidates(
    content_design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join existing planning candidates with actual predesign research receipts.

    Input may be absent on an initial run; never invent a catalogue or pre-
    approved donor. This only projects host-stamped reference-only candidates.
    """
    state = content_design.get("_planning_state")
    state_rows = state.get("repository_candidates", ()) if isinstance(state, Mapping) else ()
    research = content_design.get("_pre_design_research")
    deterministic = research.get("deterministic") if isinstance(research, Mapping) else None
    research_rows = (
        deterministic.get("repository_candidates", ())
        if isinstance(deterministic, Mapping) else ()
    )
    source_handoff = content_design.get("_pre_design_source_handoff")
    source_rows = (
        source_handoff.get("repository_candidates", ())
        if isinstance(source_handoff, Mapping)
        and source_handoff.get("schema_version") == "mmm/pre-design-source-candidate-handoff-v1"
        else ()
    )
    combined = merge_repository_candidates(
        state_rows if isinstance(state_rows, (list, tuple)) else (),
        research_rows if isinstance(research_rows, (list, tuple)) else (),
    )
    return merge_repository_candidates(
        combined,
        source_rows if isinstance(source_rows, (list, tuple)) else (),
    )



def ensure_authored_source_handoff(
    content_design: Mapping[str, Any],
    *,
    router: Any,
    prompt: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One shared producer/consumer boundary for both authored compile routes.

    Never calls a generative model. Retains a host-produced source-evidence
    receipt by prompt identity and leaves all candidates proof-required.
    """
    content = dict(content_design)
    from .pre_design_research_pipeline import (
        _canonical_sha256, collect_source_candidate_handoff,
    )

    expected_prompt = _canonical_sha256(prompt)
    old_research = content.get("_pre_design_research")
    stale_research = (
        isinstance(old_research, Mapping)
        and old_research.get("prompt_sha256") != expected_prompt
    )
    if stale_research:
        # An explicitly mismatched prompt invalidates its derived candidates.
        content.pop("_pre_design_research", None)
    cached = content.get("_pre_design_source_handoff")
    stale_handoff = (
        isinstance(cached, Mapping)
        and cached.get("prompt_sha256") != expected_prompt
    )
    if stale_handoff:
        content.pop("_pre_design_source_handoff", None)
        cached = None
    if stale_research or stale_handoff:
        # Previous planning candidates may derive from the same stale search.
        # Preserve unrelated planning state, not its unbound donor identities.
        state = content.get("_planning_state")
        if isinstance(state, Mapping) and "repository_candidates" in state:
            content["_planning_state"] = {
                key: value for key, value in state.items()
                if key != "repository_candidates"
            }

    candidates = authored_design_repository_candidates(content)
    if candidates:
        return content, candidates
    if not isinstance(cached, Mapping) or cached.get("prompt_sha256") != expected_prompt:
        try:
            cached = collect_source_candidate_handoff(router, prompt)
        except Exception as exc:
            # Provider failure is evidence, never a production-blocking exception.
            cached = {
                "schema_version": "mmm/pre-design-source-candidate-handoff-v1",
                "prompt_sha256": expected_prompt,
                "repository_candidates": [],
                "status": "provider_error",
                "error_type": type(exc).__name__,
            }
    content["_pre_design_source_handoff"] = dict(cached)
    return content, authored_design_repository_candidates(content)



def planning_state_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read every host-stamped candidate receipt from the planning-state SSOT."""
    state = design.get("_planning_state")
    rows = state.get("repository_candidates") if isinstance(state, Mapping) else None
    cards: list[dict[str, Any]] = []
    for raw in rows if isinstance(rows, list) else ():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("origin") != "host_grounded_retrieval":
            continue
        if raw.get("reference_only") is not True or raw.get("source_reuse_authority") != "verification_required":
            continue
        repository = _repository_from_value(raw.get("repository"))
        if not repository:
            continue
        cards.append(
            {
                "repository": repository,
                "page_refs": [f"planning:{value}" for value in raw.get("query_sha256", []) if value],
                "source_ids": list(raw.get("source_ids") or []),
                "source_urls": list(raw.get("source_urls") or []),
                "evidence_text": _text(raw.get("evidence_text"))[:_MAX_EVIDENCE_CHARS],
                "evidence_tokens": list(raw.get("evidence_tokens") or [])[:64],
                "catalog_evidence": [
                    dict(item) for item in raw.get("catalog_evidence", [])
                    if isinstance(item, Mapping)
                ] if isinstance(raw.get("catalog_evidence"), list) else [],
                "domain_ids": list(raw.get("domain_ids") or []),
                "candidate_score": float(raw.get("candidate_score") or 0.0),
                "explicit_reference": False,
                "candidate_origin": "host_grounded_retrieval",
                "reference_only": True,
                "source_reuse_authority": "verification_required",
            }
        )
    return tuple(cards)


__all__ = [
    "authored_design_repository_candidates",
    "ensure_authored_source_handoff",
    "merge_repository_candidates",
    "planning_state_repository_cards",
    "project_repository_candidates",
]
