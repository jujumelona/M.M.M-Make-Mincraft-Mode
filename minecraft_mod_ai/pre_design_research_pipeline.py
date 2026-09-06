from __future__ import annotations

"""Canonical host-owned pre-design research pipeline.

Retrieval, source materialization, grounding validation, termination, and sufficiency
remain host-owned. The model is never an evidence authority. Minecraft ecosystem
discovery uses the same catalog-first provider policy as planning-state research.
"""

import hashlib
import json
import re
import traceback
from collections.abc import Mapping
from typing import Any

from .agent_capability_context import target_neutral_research_scope
from .catalog_first_grounded_rag import forced_rag_bundle
from .central_research import normalize_research_brief
from .external_procedural_skill_contract import attach_procedural_skillbank
from .minecraft_knowledge_contract import (
    compile_minecraft_knowledge_plan,
    evaluate_route_coverage,
)
from .pre_design_domain_research import research_document_domain
from .pre_design_rag_quality_contract import _source_body
from .research_coordinator import collect_technology_radar
from .retrieval import BUILTIN_CORPUS
from .small_model_execution_extensions_contract import compose_research_skillbank
from .technology_radar import build_technology_radar

_QUERY_STOP_TERMS = frozenset(
    {
        "fabric",
        "minecraft",
        "mod",
        "mods",
        "requested",
        "existing",
        "host",
        "resolved",
        "target",
        "implementation",
        "mechanic",
        "system",
        "game",
        "player",
        "players",
        "official",
        "documentation",
        "docs",
        "source",
    }
)


class PreDesignResearchFailure(RuntimeError):
    """Raised for host-owned research invariant failures."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _emit_research_diagnostic(event: str, **fields: Any) -> None:
    print(
        "PRE-DESIGN RESEARCH DIAGNOSTIC: "
        + json.dumps(
            {"event": event, **fields},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _exception_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _active_requirement_texts(prompt: str) -> list[str]:
    """Return active authored requirements without rebuilding semantic authority."""
    try:
        from . import planning_authority

        active = planning_authority._ACTIVE_REQUEST_CATALOG.get()
    except Exception:
        active = None
    values: list[str] = []
    if active is not None and active[0] == prompt:
        catalog = active[1]
        raw_requirements = (
            catalog.get("requirements", []) if isinstance(catalog, Mapping) else []
        )
        for raw in raw_requirements if isinstance(raw_requirements, list) else []:
            if not isinstance(raw, Mapping):
                continue
            span = raw.get("source_span")
            span_text = (
                str(span.get("text") or "").strip()
                if isinstance(span, Mapping)
                else ""
            )
            text = (
                span_text
                or str(raw.get("semantic_statement") or "").strip()
                or str(raw.get("statement") or "").strip()
            )
            text = " ".join(text.split()).strip()
            if text and text not in values:
                values.append(text)
    normalized_prompt = " ".join(str(prompt or "").split()).strip()
    if not values and normalized_prompt:
        values.append(normalized_prompt)
    return values


def _pre_design_brief(prompt: str) -> dict[str, Any]:
    requirements = _active_requirement_texts(prompt)
    queries = list(requirements)
    architecture_query = (
        "Minecraft Fabric architecture items entities dimensions world interaction "
        "networking persistence data components testing patterns"
    )
    if architecture_query not in queries:
        queries.append(architecture_query)
    candidate = {
        "summary": (
            "Design-critical target-neutral research only. User-authored requirements "
            "remain authoritative even when every external provider returns zero evidence."
        ),
        "domains": [
            {
                "domain_id": "request",
                "objective": (
                    "Resolve target-neutral Minecraft mechanics and implementation patterns "
                    "for every authored requirement without expanding or deleting scope."
                ),
                "requirements": requirements,
                "evidence_kinds": [
                    "minecraft_api",
                    "runtime_behavior",
                    "local_project",
                    "testing",
                ],
                "queries": queries,
                "providers": [
                    "curseforge",
                    "modrinth",
                    "official_docs",
                    "project_rag",
                    "github",
                ],
                "depends_on": [],
            }
        ],
        "unresolved_questions": [],
    }
    return normalize_research_brief(
        prompt,
        {"title": "pre-design research"},
        candidate,
    )


def _evidence_tokens(value: Any) -> set[str]:
    folded = re.sub(r"[_./:+-]+", " ", str(value).casefold())
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[가-힣]{2,}", folded)
        if len(token) > 1 and token not in _QUERY_STOP_TERMS
    }


def _document_relevance(query: str, document: Mapping[str, Any]) -> int:
    wanted = _evidence_tokens(query)
    if not wanted:
        return 0
    searchable = " ".join(
        str(document.get(field, ""))
        for field in (
            "document_id",
            "source_id",
            "title",
            "url",
            "topics",
            "content",
        )
    )
    return len(wanted & _evidence_tokens(searchable))


def _builtin_content_record(
    record: Mapping[str, Any],
    query: str,
) -> dict[str, Any] | None:
    source_id = str(
        record.get("source_id") or record.get("document_id") or ""
    ).strip()
    for document in BUILTIN_CORPUS:
        document_id = str(getattr(document, "document_id", "")).strip()
        if not source_id or source_id != document_id:
            continue
        candidate = {
            **document.public_metadata(),
            "source_id": source_id,
            "content": document.content,
            "evidence_origin": "official_reviewed_document",
        }
        if _document_relevance(query, candidate) > 0:
            return candidate
    return None


def _section_records(section: Any) -> list[Mapping[str, Any]]:
    if not isinstance(section, Mapping):
        return []
    rows: list[Mapping[str, Any]] = []
    for key in ("documents", "hits", "sources", "records", "results"):
        value = section.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))
    return rows


def _query_content_records(query_item: Mapping[str, Any]) -> list[dict[str, Any]]:
    query = str(query_item.get("query") or "")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section_name in ("external_rag", "code_rag", "project_rag"):
        section = query_item.get(section_name)
        for raw in _section_records(section):
            item = dict(raw)
            content = _source_body(item)
            if not content and section_name == "project_rag":
                materialized = _builtin_content_record(item, query)
                if materialized is not None:
                    item = materialized
                    content = _source_body(item)
            if not content:
                continue
            digest = str(item.get("content_sha256") or "").strip()
            if not digest:
                digest = "sha256:" + hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()
                item["content_sha256"] = digest
            if digest in seen:
                continue
            seen.add(digest)
            item["retrieval_section"] = section_name
            records.append(item)
    return records


def _domain_source_value(
    domain_id: str,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    result = {key: value for key, value in bundle.items() if key != "domains"}
    domains = bundle.get("domains")
    if isinstance(domains, list):
        selected = next(
            (
                item
                for item in domains
                if isinstance(item, Mapping)
                and str(item.get("domain_id") or "") == domain_id
            ),
            None,
        )
        if isinstance(selected, Mapping):
            result.update(dict(selected))
    return result


def _is_github_record(record: Mapping[str, Any]) -> bool:
    source_id = str(record.get("source_id") or "").casefold()
    url = str(record.get("url") or "").casefold()
    metadata = record.get("metadata")
    repository = (
        str(metadata.get("repository") or "")
        if isinstance(metadata, Mapping)
        else ""
    )
    return source_id.startswith("github:") or "github.com/" in url or bool(repository)


def _grounded_domain_evidence(
    domain_id: str,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _domain_source_value(domain_id, bundle)
    queries: list[dict[str, Any]] = []
    raw_queries = selected.get("queries")
    for item in raw_queries if isinstance(raw_queries, list) else []:
        if not isinstance(item, Mapping):
            continue
        records = _query_content_records(item)
        external = (
            item.get("external_rag")
            if isinstance(item.get("external_rag"), Mapping)
            else {}
        )
        github = (
            external.get("github_retrieval")
            if isinstance(external.get("github_retrieval"), Mapping)
            else {}
        )
        queries.append(
            {
                "query": str(item.get("query") or ""),
                "query_sha256": str(item.get("query_sha256") or ""),
                "evidence_records": records,
                "content_record_count": len(records),
                "github_record_count": sum(
                    1 for record in records if _is_github_record(record)
                ),
                "github_provider_status": str(
                    github.get("provider_status") or "not_requested"
                ),
                "github_saturation_reason": str(
                    github.get("saturation_reason") or ""
                ),
                "retrieval_errors": [
                    str(error) for error in external.get("errors", [])[:6]
                ]
                if isinstance(external.get("errors"), list)
                else [],
                "provider_receipts": dict(external.get("providers") or {})
                if isinstance(external.get("providers"), Mapping)
                else {},
            }
        )
    return {
        "schema_version": "mmm/pre-design-grounded-domain-evidence-v2",
        "domain_id": domain_id,
        "queries": queries,
    }


def _grounded_rag_receipt(bundle: Mapping[str, Any]) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    content_record_count = 0
    raw_domains = bundle.get("domains")
    for domain in raw_domains if isinstance(raw_domains, list) else []:
        if not isinstance(domain, Mapping):
            continue
        domain_id = str(domain.get("domain_id") or "")
        grounded = _grounded_domain_evidence(domain_id, bundle)
        query_receipts: list[dict[str, Any]] = []
        for query in grounded.get("queries", []):
            count = int(query.get("content_record_count") or 0)
            content_record_count += count
            query_receipts.append(
                {
                    "query": query.get("query"),
                    "query_sha256": query.get("query_sha256"),
                    "content_record_count": count,
                    "github_record_count": query.get("github_record_count"),
                    "github_provider_status": query.get("github_provider_status"),
                    "github_saturation_reason": query.get("github_saturation_reason"),
                    "provider_receipts": query.get("provider_receipts"),
                    "retrieval_errors": query.get("retrieval_errors"),
                }
            )
        domains.append({"domain_id": domain_id, "queries": query_receipts})
    return {
        "schema_version": "mmm/pre-design-grounded-rag-receipt-v3",
        "status": (
            "available" if content_record_count else "no_external_or_local_source_bodies"
        ),
        "research_sha256": bundle.get("research_sha256"),
        "domain_count": len(domains),
        "query_count": bundle.get("query_count"),
        "external_source_count": bundle.get("external_source_count"),
        "content_record_count": content_record_count,
        "domains": domains,
        "source_content_omitted": True,
    }


def _claim_refs(claim: Mapping[str, Any]) -> tuple[str, ...]:
    raw = (
        claim.get("evidence_refs")
        or claim.get("citations")
        or claim.get("source_refs")
        or ()
    )
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(str(value).strip() for value in raw if str(value).strip())
    )


def _validate_document_grounding(
    _legacy_agentic_owner: Any,
    project_rag: Any,
    note: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    domain_id: str,
) -> None:
    """Validate all note/card provenance against host-materialized evidence pages.

    The first positional argument is intentionally ignored so current callers can be
    migrated without resurrecting the deleted agentic private validator.
    """
    pages = project_rag._read_evidence_pages(document)
    allowed_refs = frozenset(
        str(page.get("page_ref") or "").strip()
        for page in pages
        if isinstance(page, Mapping) and str(page.get("page_ref") or "").strip()
    )

    for card in note.get("grounded_evidence_cards", ()):
        if not isinstance(card, Mapping):
            raise PreDesignResearchFailure(
                f"Pre-design evidence card is invalid for {domain_id!r}."
            )
        page_ref = str(card.get("page_ref") or "").strip()
        if not page_ref or page_ref not in allowed_refs:
            raise PreDesignResearchFailure(
                "Pre-design evidence card cited a page outside host-owned materialization "
                f"for {domain_id!r}: {page_ref!r}."
            )

    claims = note.get("claims")
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, Mapping):
            raise PreDesignResearchFailure(
                f"Pre-design claim is not an object for {domain_id!r}."
            )
        refs = _claim_refs(claim)
        if not refs:
            raise PreDesignResearchFailure(
                f"Pre-design claim has no host evidence reference for {domain_id!r}."
            )
        outside = [ref for ref in refs if ref not in allowed_refs]
        if outside:
            raise PreDesignResearchFailure(
                "Pre-design claim cited evidence outside host-owned pages for "
                f"{domain_id!r}: {outside}."
            )


def _domain_failure_reasons(note: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    checkpoint = note.get("checkpoint")
    status = (
        str(checkpoint.get("status") or "")
        if isinstance(checkpoint, Mapping)
        else ""
    )
    if status in {"terminal_gap", "failed"}:
        reasons.append(f"checkpoint.status={status}")
    failures = note.get("research_failures")
    if isinstance(failures, list) and failures:
        reasons.append("research_failures is non-empty")
    if note.get("sufficient") is not True:
        reasons.append("sufficient is not true")
    if note.get("fixed_point") is True:
        reasons.append("fixed_point=true")
    return reasons


def _validate_domain_result(note: Any, *, domain_id: str) -> dict[str, Any]:
    if not isinstance(note, Mapping):
        raise PreDesignResearchFailure(
            f"Pre-design research domain {domain_id!r} returned a non-object result."
        )
    value = dict(note)
    reasons = _domain_failure_reasons(value)
    _emit_research_diagnostic(
        "domain_result",
        domain_id=domain_id,
        sufficient=value.get("sufficient"),
        fixed_point=value.get("fixed_point"),
        source_body_count=value.get("source_body_count"),
        model_called=value.get("model_called"),
        failure_reasons=reasons,
    )
    if reasons:
        raise PreDesignResearchFailure(
            f"Pre-design research failed for domain {domain_id!r}: {'; '.join(reasons)}"
        )
    return value


def _target_frozen(knowledge_plan: Mapping[str, Any]) -> bool:
    policy = knowledge_plan.get("policy")
    return bool(policy.get("target_frozen")) if isinstance(policy, Mapping) else False


def _deferred_technology_receipt(
    knowledge_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "mmm/technology-radar-deferred-v1",
        "status": "deferred_until_target_freeze",
        "target_frozen": False,
        "reason": (
            "Exact Minecraft/Fabric target is intentionally not frozen during pre-design."
        ),
        "knowledge_plan_sha256": knowledge_plan.get("plan_sha256"),
    }


def collect_design_research(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from . import pre_design_grounded_rag as project_rag

    research_brief = _pre_design_brief(prompt)
    knowledge_plan = compile_minecraft_knowledge_plan(prompt)
    deterministic: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    _emit_research_diagnostic(
        "research_start",
        authored_requirement_count=len(_active_requirement_texts(prompt)),
        query_count=sum(
            len(domain.get("queries", []))
            for domain in research_brief.get("domains", [])
            if isinstance(domain, Mapping)
            and isinstance(domain.get("queries"), list)
        ),
    )

    try:
        grounded_bundle = forced_rag_bundle(project_rag, router, research_brief)
    except Exception as exc:
        diagnostic = _exception_payload(exc)
        _emit_research_diagnostic(
            "grounded_retrieval_host_failure",
            exception=diagnostic,
        )
        raise PreDesignResearchFailure(
            f"Host-owned grounded retrieval failed: {type(exc).__name__}: {exc}"
        ) from exc

    deterministic["grounded_rag"] = _grounded_rag_receipt(grounded_bundle)
    if _target_frozen(knowledge_plan):
        try:
            deterministic["technology_radar"] = collect_technology_radar(
                prompt,
                research_brief,
                page_size=50,
                page_builder=build_technology_radar,
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "technology_radar",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            deterministic["technology_radar"] = {
                "status": "unavailable",
                "failure": _exception_payload(exc),
            }
    else:
        deterministic["technology_radar"] = _deferred_technology_receipt(
            knowledge_plan
        )

    domain_notes: list[dict[str, Any]] = []
    for domain in research_brief.get("domains", []):
        if not isinstance(domain, Mapping):
            continue
        domain_id = str(domain.get("domain_id") or "").strip() or "unknown"
        grounded = _grounded_domain_evidence(domain_id, grounded_bundle)
        document = project_rag._materialize_domain_evidence_document(
            domain_id,
            {"grounded_rag": grounded},
        )
        with target_neutral_research_scope():
            raw_note = research_document_domain(
                None,
                project_rag,
                router,
                prompt=prompt,
                domain=domain,
                document=document,
                trace_metadata=trace_metadata,
            )
        _validate_document_grounding(
            None,
            project_rag,
            raw_note,
            document,
            domain_id=domain_id,
        )
        domain_notes.append(
            _validate_domain_result(raw_note, domain_id=domain_id)
        )

    payload: dict[str, Any] = {
        "schema_version": "mmm/agentic-pre-design-research-v2",
        "research_brief": research_brief,
        "deterministic": deterministic,
        "domain_notes": domain_notes,
        "errors": errors,
        "method": {
            "requirement_authority": "host_active_requirement_ledger",
            "provider_order": [
                "curseforge_if_configured",
                "modrinth",
                "official_docs",
                "project_rag",
                "github_exact_source_or_empty_catalog_fallback",
            ],
            "provider_failure_semantics": "isolated_per_provider_and_query",
            "zero_source_semantics": "retain_authored_requirement_and_skip_model",
            "model_role": "none_for_predesign_evidence_projection",
            "grounding_owner": "pre_design_research_pipeline",
        },
        "minecraft_knowledge_plan": knowledge_plan,
    }
    coverage = evaluate_route_coverage(knowledge_plan, payload)
    if coverage["status"] != "PASS":
        raise PreDesignResearchFailure(
            "Minecraft knowledge route coverage blocked pre-design research: "
            + ", ".join(coverage.get("blocking_requirement_refs", ()))
        )
    payload["minecraft_knowledge_route_coverage"] = coverage
    payload = attach_procedural_skillbank(router, prompt, payload)
    payload = compose_research_skillbank(router, prompt, payload)
    payload["research_sha256"] = _canonical_sha256(payload)
    _emit_research_diagnostic(
        "research_complete",
        research_sha256=payload["research_sha256"],
        domain_count=len(domain_notes),
        authored_requirement_count=len(_active_requirement_texts(prompt)),
        content_record_count=deterministic["grounded_rag"].get(
            "content_record_count"
        ),
        external_source_count=deterministic["grounded_rag"].get(
            "external_source_count"
        ),
    )
    return payload


__all__ = [
    "PreDesignResearchFailure",
    "_grounded_domain_evidence",
    "_validate_document_grounding",
    "collect_design_research",
]
