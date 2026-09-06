from __future__ import annotations

"""Compact, non-authoritative repository candidates projected from grounded RAG.

This module is deliberately host-only.  It may discover that a materialized GitHub
record is worth *checking* as a donor, but it can never authorize source reuse.  The
existing grounded_source_reuse owner remains responsible for immutable revision,
license, dependency/source closure, target compatibility, and compile proof.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

_SCHEMA = "mmm/repository-candidate-receipt-v1"
_MAX_PER_DOMAIN = 3
_MAX_GLOBAL = 8
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
    return {
        token
        for token in _TOKEN_RE.findall(folded)
        if len(token) > 1 and token not in _STOP
    }


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
    return _text(" ".join(_text(part) for part in parts if _text(part)))[:_MAX_EVIDENCE_CHARS]


def _source_code_domain(domain: Mapping[str, Any]) -> bool:
    kinds = domain.get("evidence_kinds")
    return isinstance(kinds, Sequence) and not isinstance(kinds, (str, bytes, bytearray)) and (
        "source_code" in {str(value) for value in kinds}
    )


def project_repository_candidates(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
    *,
    per_domain_limit: int = _MAX_PER_DOMAIN,
) -> list[dict[str, Any]]:
    """Project bounded GitHub donor *candidates* from host-materialized source records."""
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
            rank_score = (1.0 / (1.0 + query_index + record_index)) + min(overlap, 8) * 0.125
            key = repository.casefold()
            source_id = _text(record.get("source_id")) or f"github:{repository}"
            source_url = _record_url(record, repository)
            current = by_repo.get(key)
            if current is None:
                by_repo[key] = {
                    "schema_version": _SCHEMA,
                    "repository": repository,
                    "domain_ids": [domain_id] if domain_id else [],
                    "source_ids": [source_id],
                    "source_urls": [source_url],
                    "query_sha256": [query_sha],
                    "evidence_text": evidence_text,
                    "evidence_tokens": sorted(_tokens(evidence_text)),
                    "candidate_score": round(rank_score, 6),
                    "origin": "host_grounded_retrieval",
                    "reference_only": True,
                    "source_reuse_authority": "verification_required",
                    "explicit_reference": False,
                }
                continue
            current["candidate_score"] = max(float(current["candidate_score"]), round(rank_score, 6))
            for field, value in (
                ("source_ids", source_id),
                ("source_urls", source_url),
                ("query_sha256", query_sha),
            ):
                if value and value not in current[field]:
                    current[field].append(value)
            merged_text = _text(f"{current['evidence_text']} {evidence_text}")[:_MAX_EVIDENCE_CHARS]
            current["evidence_text"] = merged_text
            current["evidence_tokens"] = sorted(_tokens(merged_text))

    ranked = sorted(
        by_repo.values(),
        key=lambda row: (-float(row.get("candidate_score", 0.0)), str(row.get("repository", "")).casefold()),
    )
    return ranked[: max(1, int(per_domain_limit))]


def merge_repository_candidates(
    existing: Sequence[Mapping[str, Any]] | None,
    new: Sequence[Mapping[str, Any]] | None,
    *,
    global_limit: int = _MAX_GLOBAL,
) -> list[dict[str, Any]]:
    """Merge candidate receipts without ever upgrading their authority."""
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
        for field in ("domain_ids", "source_ids", "source_urls", "query_sha256"):
            values = candidate.get(field)
            if not isinstance(values, list):
                continue
            target = current.setdefault(field, [])
            for value in values:
                if value and value not in target:
                    target.append(value)
        merged_text = _text(f"{current.get('evidence_text', '')} {candidate.get('evidence_text', '')}")[:_MAX_EVIDENCE_CHARS]
        current["evidence_text"] = merged_text
        current["evidence_tokens"] = sorted(_tokens(merged_text))
    ranked = sorted(
        by_repo.values(),
        key=lambda row: (-float(row.get("candidate_score", 0.0) or 0.0), str(row.get("repository", "")).casefold()),
    )
    return ranked[: max(1, int(global_limit))]


def planning_state_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read only host-stamped candidate receipts from the planning-state SSOT."""
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
                "explicit_reference": False,
                "candidate_origin": "host_grounded_retrieval",
                "reference_only": True,
                "source_reuse_authority": "verification_required",
            }
        )
    return tuple(cards[:_MAX_GLOBAL])


__all__ = [
    "merge_repository_candidates",
    "planning_state_repository_cards",
    "project_repository_candidates",
]
