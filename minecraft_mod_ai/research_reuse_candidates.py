from __future__ import annotations
"""CurseForge-origin *source links* across plan/execute; never donor authority.

A catalogue hit cannot authorize copying compiled or source code. Only an
explicit source URL supplied by the CurseForge record can become a reference
candidate for subsequent inspection, licensing and target-compile proof.
No GitHub code search or repository discovery is performed here.
"""
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


def _source_link(record: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = record
    source_id = str(metadata.get("source_id") or record.get("source_id") or "")
    provider = str(metadata.get("provider") or record.get("provider") or "").casefold()
    if provider != "curseforge" and not source_id.startswith("curseforge:"):
        return None
    url = str(metadata.get("source_url") or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    # The provider may link to a public Git host. Following that *specific*
    # source URL is not GitHub searching; an unverified source stays a reference.
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.hostname.casefold() not in {"github.com", "gitlab.com"} or len(parts) < 2:
        return None
    owner, name = parts[:2]
    name = name.removesuffix(".git")
    if not all(value and value.replace("-", "").replace("_", "").replace(".", "").isalnum()
               for value in (owner, name)):
        return None
    repository = owner + "/" + name
    return {
        "repository": repository,
        "origin": "curseforge_source_url",
        "reference_only": True,
        "source_reuse_authority": "verification_required",
        "source_ids": [source_id],
        "source_urls": [url],
        "catalog_evidence": [{
            "provider": "curseforge", "source_id": source_id,
            "summary": str(record.get("summary") or metadata.get("summary") or "")[:400],
            "catalog_url": str(metadata.get("catalog_url") or ""),
        }],
    }


def project_repository_candidates(domain: Mapping[str, Any], grounded: Mapping[str, Any], *,
                                  per_domain_limit: int | None = None) -> list[dict[str, Any]]:
    """Project source-linked CurseForge records; ordinary catalogue hits are not donors."""
    rows = []
    for query in grounded.get("queries", ()):
        if not isinstance(query, Mapping):
            continue
        for item in query.get("evidence_records", ()):
            if isinstance(item, Mapping) and (candidate := _source_link(item)) is not None:
                rows.append(candidate)
    return merge_repository_candidates([], rows, global_limit=per_domain_limit)


def merge_repository_candidates(existing: Sequence[Mapping[str, Any]] | None,
                                new: Sequence[Mapping[str, Any]] | None, *,
                                global_limit: int | None = None) -> list[dict[str, Any]]:
    """Deduplicate reference candidates without granting source or compile authority."""
    result: dict[str, dict[str, Any]] = {}
    for row in (*(existing or ()), *(new or ())):
        if not isinstance(row, Mapping) or row.get("reference_only") is not True:
            continue
        if row.get("source_reuse_authority") != "verification_required":
            continue
        if not any(str(value).startswith("curseforge:")
                   for value in row.get("source_ids", ())):
            continue
        name = str(row.get("repository") or "").strip()
        if not name or name.count("/") != 1:
            continue
        if name not in result:
            result[name] = dict(row)
        else:
            prev = result[name]
            for field in ("source_ids", "source_urls", "catalog_evidence"):
                merged = list(prev.get(field, ()))
                for value in row.get(field, ()):
                    if value not in merged:
                        merged.append(value)
                prev[field] = merged
    values = list(result.values())
    return values[:global_limit] if global_limit is not None else values


def authored_design_repository_candidates(content_design: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipt = content_design.get("_pre_design_source_handoff")
    if not isinstance(receipt, Mapping):
        return []
    return merge_repository_candidates((), receipt.get("repository_candidates", ()))


def planning_state_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(authored_design_repository_candidates(design))


def ensure_authored_source_handoff(content_design: Mapping[str, Any], *,
                                   router: Any, prompt: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preserve scoped CurseForge source-link candidates between plan and execute."""
    from .pre_design_research_pipeline import _canonical_sha256, collect_source_candidate_handoff
    content = dict(content_design)
    expected = _canonical_sha256(prompt)
    old = content.get("_pre_design_research")
    if isinstance(old, Mapping) and old.get("prompt_sha256") != expected:
        content.pop("_pre_design_research", None)
        state = content.get("_planning_state")
        if isinstance(state, Mapping):
            content["_planning_state"] = {k: v for k, v in state.items() if k != "repository_candidates"}
    cached = content.get("_pre_design_source_handoff")
    if not isinstance(cached, Mapping) or cached.get("prompt_sha256") != expected:
        try:
            cached = collect_source_candidate_handoff(router, prompt)
        except Exception as exc:
            cached = {"schema_version": "mmm/pre-design-source-candidate-handoff-v1",
                      "prompt_sha256": expected, "repository_candidates": [],
                      "status": "provider_error", "error_type": type(exc).__name__}
    candidates = merge_repository_candidates((), cached.get("repository_candidates", ()))
    content["_pre_design_source_handoff"] = {
        **dict(cached), "repository_candidates": candidates,
        "candidate_count": len(candidates),
    }
    return content, candidates


__all__ = ["project_repository_candidates", "merge_repository_candidates",
           "authored_design_repository_candidates", "planning_state_repository_cards",
           "ensure_authored_source_handoff"]
