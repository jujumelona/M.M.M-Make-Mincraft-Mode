from __future__ import annotations

"""Single-owner pre-design evidence collection with fail-closed diagnostics.

Pre-design research is target-neutral by contract. Exact Minecraft/Fabric version,
mappings, dependency and API-signature research begins only after the host freezes the
platform target. Third-party donor discovery likewise belongs to the frozen-design reuse
phase. The host keeps complete evidence while model turns receive bounded receipts.
"""

import hashlib
import json
import re
import traceback
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .agent_capability_context import target_neutral_research_scope
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
    """Raised when pre-design evidence did not reach a usable completed state."""


def _emit_research_diagnostic(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(
        "PRE-DESIGN RESEARCH DIAGNOSTIC: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def _exception_payload(exc: BaseException) -> dict[str, Any]:
    chain: list[dict[str, str]] = []
    pending: BaseException | None = exc
    seen: set[int] = set()
    while pending is not None and id(pending) not in seen:
        seen.add(id(pending))
        chain.append(
            {
                "type": f"{type(pending).__module__}.{type(pending).__qualname__}",
                "message": str(pending),
            }
        )
        cause = getattr(pending, "__cause__", None)
        context = getattr(pending, "__context__", None)
        pending = cause if isinstance(cause, BaseException) else (
            context if isinstance(context, BaseException) else None
        )
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "chain": chain,
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _pre_design_brief(prompt: str) -> dict[str, Any]:
    candidate = {
        "summary": (
            "Design-critical target-neutral research only. Exact target compatibility, "
            "reusable donor selection, dependency closure and license validation are "
            "deferred until the detailed design and platform target are frozen."
        ),
        "domains": [
            {
                "domain_id": "request",
                "objective": (
                    "Resolve target-neutral Minecraft mechanics, architecture patterns and "
                    "existing local-project capabilities needed to design the authored request."
                ),
                "requirements": [prompt],
                "evidence_kinds": [
                    "minecraft_api",
                    "runtime_behavior",
                    "local_project",
                    "testing",
                ],
                "queries": [
                    prompt,
                    (
                        "Minecraft Fabric architecture items entities dimensions world "
                        "interaction networking persistence data components testing patterns"
                    ),
                ],
                "providers": ["official_docs", "project_rag", "external_mcp"],
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
    query_tokens = _evidence_tokens(query)
    if not query_tokens:
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
    return len(query_tokens & _evidence_tokens(searchable))


def _builtin_content_record(
    record: Mapping[str, Any], query: str
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
        }
        if _document_relevance(query, candidate) <= 0:
            return None
        candidate["evidence_origin"] = "official_reviewed_document"
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
    """Materialize only verified source bodies before any evidence document is written."""

    query = str(query_item.get("query", ""))
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
            digest = str(item.get("content_sha256", "")).strip()
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


def _is_github_record(record: Mapping[str, Any]) -> bool:
    source_id = str(record.get("source_id", "")).casefold()
    url = str(record.get("url", "")).casefold()
    metadata = record.get("metadata")
    repository = (
        str(metadata.get("repository", "")).strip()
        if isinstance(metadata, Mapping)
        else ""
    )
    return (
        source_id.startswith("github:")
        or "github.com/" in url
        or bool(repository)
    )


def _grounded_domain_evidence(
    agentic: Any, domain_id: str, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    selected = agentic._domain_source_value(domain_id, bundle)
    queries: list[dict[str, Any]] = []
    raw_queries = selected.get("queries", []) if isinstance(selected, Mapping) else []
    if isinstance(raw_queries, list):
        for item in raw_queries:
            if not isinstance(item, Mapping):
                continue
            records = _query_content_records(item)
            external = item.get("external_rag")
            external = external if isinstance(external, Mapping) else {}
            github = external.get("github_retrieval")
            github = github if isinstance(github, Mapping) else {}
            queries.append(
                {
                    "query": str(item.get("query", "")),
                    "query_sha256": str(item.get("query_sha256", "")),
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
                    "retrieval_errors": (
                        [str(error) for error in external.get("errors", ())[:3]]
                        if isinstance(external.get("errors"), list)
                        else []
                    ),
                }
            )
    return {
        "schema_version": "mmm/pre-design-grounded-domain-evidence",
        "domain_id": domain_id,
        "queries": queries,
    }


def _grounded_rag_receipt(bundle: Mapping[str, Any]) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    content_record_count = 0
    raw_domains = bundle.get("domains", [])
    for domain in raw_domains if isinstance(raw_domains, list) else []:
        if not isinstance(domain, Mapping):
            continue
        queries: list[dict[str, Any]] = []
        raw_queries = domain.get("queries", [])
        for item in raw_queries if isinstance(raw_queries, list) else []:
            if not isinstance(item, Mapping):
                continue
            records = _query_content_records(item)
            content_record_count += len(records)
            external = item.get("external_rag")
            external = external if isinstance(external, Mapping) else {}
            github = external.get("github_retrieval")
            github = github if isinstance(github, Mapping) else {}
            queries.append(
                {
                    "query": str(item.get("query", "")),
                    "query_sha256": str(item.get("query_sha256", "")),
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
                    "github_search_requests": int(
                        github.get("search_requests") or 0
                    ),
                    "github_source_requests": int(
                        github.get("source_requests") or 0
                    ),
                    "retrieval_errors": (
                        [str(error) for error in external.get("errors", ())[:3]]
                        if isinstance(external.get("errors"), list)
                        else []
                    ),
                }
            )
        domains.append(
            {"domain_id": str(domain.get("domain_id", "")), "queries": queries}
        )
    return {
        "schema_version": "mmm/pre-design-grounded-rag-receipt-v2",
        "status": "available" if content_record_count else "unavailable",
        "research_sha256": bundle.get("research_sha256"),
        "domain_count": len(domains),
        "content_record_count": content_record_count,
        "domains": domains,
        "source_content_omitted": True,
    }


def _validate_domain_provider_grounding(
    domain: Mapping[str, Any],
    grounded: Mapping[str, Any],
) -> None:
    # Ordinary target-neutral donor discovery is advisory.  An explicitly declared
    # required provider is different: the host promised that evidence source and must
    # fail closed if it is absent.
    required = {str(value).strip().casefold() for value in domain.get("required_providers", ()) if str(value).strip()}
    if not required:
        return
    available: set[str] = set()
    for row in grounded.get("queries", ()) if isinstance(grounded, Mapping) else ():
        if not isinstance(row, Mapping):
            continue
        for record in row.get("evidence_records", ()):
            if not isinstance(record, Mapping):
                continue
            if _is_github_record(record):
                available.add("github")
            source_type = str(record.get("source_type") or "").casefold()
            source_id = str(record.get("source_id") or "").casefold()
            if "modrinth" in source_type or source_id.startswith("modrinth:"):
                available.add("modrinth")
            section = str(record.get("retrieval_section") or "").casefold()
            if section == "project_rag" or source_id.startswith("project:"):
                available.add("project_rag")
    missing = sorted(required - available)
    if missing:
        raise PreDesignResearchFailure(
            "pre-design required provider evidence is missing: " + ", ".join(missing)
        )

def _domain_document_evidence(
    agentic: Any,
    domain_id: str,
    deterministic: Mapping[str, Any],
    *,
    grounded_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "grounded_rag": _grounded_domain_evidence(
            agentic, domain_id, grounded_bundle
        )
    }
    for source, value in deterministic.items():
        if source == "grounded_rag":
            continue
        if isinstance(value, Mapping) and str(value.get("status", "")) in {
            "deferred_until_target_freeze",
            "unavailable",
        }:
            continue
        result[str(source)] = agentic._domain_source_value(domain_id, value)
    return result


def _validate_document_grounding(
    agentic: Any,
    project_rag: Any,
    note: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    domain_id: str,
) -> None:
    pages = project_rag._read_evidence_pages(document)
    allowed_refs = frozenset(
        str(page.get("page_ref", "")).strip()
        for page in pages
        if str(page.get("page_ref", "")).strip()
    )
    if (
        note.get("research_mode") == "advisory_predesign"
        and note.get("sufficient") is True
        and not note.get("claims")
    ):
        return
    try:
        agentic._validate_sufficient_research(note, allowed_refs=allowed_refs)
    except agentic.SpecValidationError as exc:
        _emit_research_diagnostic(
            "domain_grounding_failure",
            domain_id=domain_id,
            allowed_page_refs=sorted(allowed_refs),
            note=note,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise PreDesignResearchFailure(
            "Pre-design research synthesis cited evidence outside the host-owned "
            f"lossless evidence pages for domain {domain_id!r}: {exc}"
        ) from exc


def _planner_config(router: Any) -> Any | None:
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    role = getattr(registry, "role", None)
    if registry is None or profile is None or not callable(role):
        return None
    return role(profile, "planner")


def _design_request_fits(
    agentic: Any,
    router: Any,
    prompt: str,
    research: Mapping[str, Any],
) -> bool:
    config = _planner_config(router)
    if config is None:
        return True

    from .model_context_budget import _canonical_size, request_message_budget
    from .model_router import (
        _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT,
        _inject_system_context,
    )

    budget = request_message_budget(config, ())
    for section_id, fields, _properties in agentic._SECTION_SPECS:
        for field in fields:
            messages = agentic._field_messages(
                prompt=prompt,
                section_id=section_id,
                field=field,
                research=research,
            )
            prepared = _inject_system_context(
                messages, _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT
            )
            if _canonical_size(prepared) > budget:
                return False
    return True


def _brief_model_index(
    research_brief: Any,
    *,
    full_research_sha256: str,
    domain_ids: list[str],
    agentic: Any,
) -> dict[str, Any]:
    summary = ""
    unresolved_count = 0
    if isinstance(research_brief, Mapping):
        summary = str(research_brief.get("summary", "")).strip()
        unresolved = research_brief.get("unresolved_questions", [])
        if isinstance(unresolved, list):
            unresolved_count = len(unresolved)

    return {
        "summary": summary,
        "domain_ids": domain_ids,
        "domain_count": len(domain_ids),
        "domain_ids_sha256": agentic._json_sha256(domain_ids),
        "unresolved_question_count": unresolved_count,
        "model_context_view": {
            "mode": "host_ledger_bounded_view",
            "full_research_sha256": full_research_sha256,
            "budget_authority": "model_context_budget.request_message_budget",
            "detail_selection": "source_order_until_live_request_budget",
        },
    }


def _note_summary(note: Any, *, agentic: Any) -> dict[str, Any]:
    if not isinstance(note, Mapping):
        return {
            "domain_id": "unknown",
            "note_sha256": agentic._json_sha256(note),
        }
    claims = note.get("claims", [])
    gaps = note.get("gaps", [])
    next_queries = note.get("next_queries", [])
    procedures = note.get("procedures", [])
    return {
        "domain_id": str(note.get("domain_id", "unknown") or "unknown"),
        "sufficient": bool(note.get("sufficient", False)),
        "fixed_point": bool(note.get("fixed_point", False)),
        "claim_count": len(claims) if isinstance(claims, list) else 0,
        "gap_count": len(gaps) if isinstance(gaps, list) else 0,
        "next_query_count": len(next_queries) if isinstance(next_queries, list) else 0,
        "procedure_count": len(procedures) if isinstance(procedures, list) else 0,
        "note_sha256": agentic._json_sha256(note),
    }


def _domain_ids(research_brief: Any, domain_notes: list[Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    if isinstance(research_brief, Mapping):
        domains = research_brief.get("domains", [])
        if isinstance(domains, list):
            for domain in domains:
                if not isinstance(domain, Mapping):
                    continue
                domain_id = str(domain.get("domain_id", "")).strip()
                if domain_id and domain_id not in seen:
                    seen.add(domain_id)
                    values.append(domain_id)
    for note in domain_notes:
        if not isinstance(note, Mapping):
            continue
        domain_id = str(note.get("domain_id", "")).strip()
        if domain_id and domain_id not in seen:
            seen.add(domain_id)
            values.append(domain_id)
    return values


def _enrich_note_detail(
    agentic: Any,
    router: Any,
    prompt: str,
    view: dict[str, Any],
    note_index: int,
    note: Mapping[str, Any],
) -> None:
    summary = dict(view["domain_notes"][note_index])
    base_detail = {
        **summary,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
    }
    previous = view["domain_notes"][note_index]
    view["domain_notes"][note_index] = base_detail
    if not _design_request_fits(agentic, router, prompt, view):
        view["domain_notes"][note_index] = previous
        return

    for field in ("claims", "gaps", "next_queries", "procedures"):
        raw_items = note.get(field, [])
        if not isinstance(raw_items, list):
            continue
        accepted = base_detail[field]
        for item in raw_items:
            accepted.append(deepcopy(item))
            if not _design_request_fits(agentic, router, prompt, view):
                accepted.pop()


def _bounded_model_view(
    agentic: Any,
    router: Any,
    prompt: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if _design_request_fits(agentic, router, prompt, payload):
        return payload

    full_research_sha256 = str(payload.get("research_sha256", ""))
    research_brief = payload.get("research_brief")
    raw_notes = payload.get("domain_notes", [])
    domain_notes = list(raw_notes) if isinstance(raw_notes, list) else []
    domain_ids = _domain_ids(research_brief, domain_notes)

    host_ledger = {
        "research_brief": deepcopy(research_brief),
        "domain_notes": deepcopy(domain_notes),
        "procedural_skillbank": deepcopy(payload.get("procedural_skillbank")),
        "minecraft_knowledge_plan": deepcopy(
            payload.get("minecraft_knowledge_plan")
        ),
        "minecraft_knowledge_route_coverage": deepcopy(
            payload.get("minecraft_knowledge_route_coverage")
        ),
        "research_sha256": full_research_sha256,
    }
    view = dict(payload)
    view["host_research_ledger"] = host_ledger
    view["research_brief"] = _brief_model_index(
        research_brief,
        full_research_sha256=full_research_sha256,
        domain_ids=domain_ids,
        agentic=agentic,
    )
    view["domain_notes"] = []
    view["model_view_sha256"] = ""

    indexed_notes: list[dict[str, Any]] = view["domain_notes"]
    for note in domain_notes:
        indexed_notes.append(_note_summary(note, agentic=agentic))
        if not _design_request_fits(agentic, router, prompt, view):
            indexed_notes.pop()
            break

    for index, note in enumerate(domain_notes[: len(indexed_notes)]):
        if isinstance(note, Mapping):
            _enrich_note_detail(agentic, router, prompt, view, index, note)

    view["model_view_sha256"] = agentic._json_sha256(
        {
            "research_brief": view.get("research_brief"),
            "domain_notes": view.get("domain_notes"),
            "deterministic": view.get("deterministic"),
            "procedural_skillbank": view.get("procedural_skillbank"),
            "errors": view.get("errors"),
        }
    )
    if not _design_request_fits(agentic, router, prompt, view):
        brief = dict(view["research_brief"])
        brief.pop("domain_ids", None)
        brief["summary"] = ""
        view["research_brief"] = brief
        view["domain_notes"] = []
        view["errors"] = []
        view["model_view_sha256"] = agentic._json_sha256(
            {
                "research_brief": brief,
                "deterministic": view.get("deterministic"),
                "procedural_skillbank": view.get("procedural_skillbank"),
            }
        )
    return view


def _domain_failure_reasons(note: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    checkpoint = note.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        status = str(checkpoint.get("status", "")).strip()
        if status in {"terminal_gap", "failed"}:
            reasons.append(f"checkpoint.status={status}")
    failures = note.get("research_failures")
    if isinstance(failures, list) and failures:
        reasons.append("research_failures is non-empty")
    if note.get("sufficient") is not True:
        reasons.append("sufficient is not true")
    if note.get("fixed_point") is True:
        reasons.append("fixed_point=true")
    return list(dict.fromkeys(reasons))


def _validate_domain_result(note: Any, *, domain_id: str) -> dict[str, Any]:
    if not isinstance(note, Mapping):
        _emit_research_diagnostic(
            "domain_result_invalid",
            domain_id=domain_id,
            result_type=type(note).__name__,
            result=note,
        )
        raise PreDesignResearchFailure(
            f"Pre-design research domain {domain_id!r} returned a non-object result."
        )

    value = dict(note)
    reasons = _domain_failure_reasons(value)
    _emit_research_diagnostic(
        "domain_result",
        domain_id=domain_id,
        status=(
            value.get("checkpoint", {}).get("status")
            if isinstance(value.get("checkpoint"), Mapping)
            else None
        ),
        sufficient=value.get("sufficient"),
        fixed_point=value.get("fixed_point"),
        failure_reasons=reasons,
        result=value,
    )
    if reasons:
        raise PreDesignResearchFailure(
            "Pre-design research failed closed for domain "
            f"{domain_id!r}: {'; '.join(reasons)}. Full domain result is printed above."
        )
    return value


def _target_frozen(knowledge_plan: Mapping[str, Any]) -> bool:
    policy = knowledge_plan.get("policy")
    return (
        bool(policy.get("target_frozen")) if isinstance(policy, Mapping) else False
    )


def _deferred_technology_receipt(
    knowledge_plan: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "mmm/technology-radar-deferred-v1",
        "status": "deferred_until_target_freeze",
        "target_frozen": False,
        "reason": (
            "Technology radar requires the exact host-selected Minecraft/Fabric target; "
            "pre-design intentionally runs before that target is frozen."
        ),
        "knowledge_plan_sha256": knowledge_plan.get("plan_sha256"),
    }


def collect_design_research(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from . import agentic_pre_design_rag as project_rag
    from . import agentic_research_game_design as agentic

    research_brief = _pre_design_brief(prompt)
    knowledge_plan = compile_minecraft_knowledge_plan(prompt)
    deterministic: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    _emit_research_diagnostic(
        "research_start",
        prompt=prompt,
        research_brief=research_brief,
        minecraft_knowledge_plan=knowledge_plan,
    )

    target_frozen = _target_frozen(knowledge_plan)
    try:
        grounded_bundle = project_rag._forced_rag_bundle(router, research_brief)
        deterministic["grounded_rag"] = _grounded_rag_receipt(grounded_bundle)
        _emit_research_diagnostic(
            "deterministic_stage_complete",
            stage="grounded_rag",
            result=deterministic["grounded_rag"],
        )
    except Exception as exc:
        diagnostic = _exception_payload(exc)
        _emit_research_diagnostic(
            "deterministic_stage_failure",
            stage="grounded_rag",
            exception=diagnostic,
        )
        raise PreDesignResearchFailure(
            "Unified pre-design grounded retrieval failed before evidence extraction: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if target_frozen:
        try:
            deterministic["technology_radar"] = collect_technology_radar(
                prompt,
                research_brief,
                page_size=50,
                page_builder=build_technology_radar,
            )
            _emit_research_diagnostic(
                "deterministic_stage_complete",
                stage="technology_radar",
                result=deterministic["technology_radar"],
            )
        except Exception as exc:
            diagnostic = _exception_payload(exc)
            _emit_research_diagnostic(
                "deterministic_stage_failure",
                stage="technology_radar",
                exception=diagnostic,
            )
            errors.append(
                {
                    "stage": "technology_radar",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            deterministic["technology_radar"] = {
                "status": "unavailable",
                "failure": diagnostic,
            }
    else:
        deterministic["technology_radar"] = _deferred_technology_receipt(
            knowledge_plan
        )
        _emit_research_diagnostic(
            "deterministic_stage_deferred",
            stage="technology_radar",
            result=deterministic["technology_radar"],
        )

    domain_notes: list[dict[str, Any]] = []
    for domain in research_brief.get("domains", []):
        if not isinstance(domain, dict):
            continue
        domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
        _emit_research_diagnostic(
            "domain_execution_start",
            domain_id=domain_id,
            domain=domain,
            deterministic_sources=list(deterministic),
        )
        try:
            domain_evidence = _domain_document_evidence(
                agentic,
                domain_id,
                deterministic,
                grounded_bundle=grounded_bundle,
            )
            _validate_domain_provider_grounding(
                domain,
                domain_evidence["grounded_rag"],
            )
            document = project_rag._materialize_domain_evidence_document(
                domain_id,
                domain_evidence,
            )
            with target_neutral_research_scope():
                raw_note = research_document_domain(
                    agentic,
                    project_rag,
                    router,
                    prompt=prompt,
                    domain=domain,
                    document=document,
                    trace_metadata=trace_metadata,
                )
            _validate_document_grounding(
                agentic,
                project_rag,
                raw_note,
                document,
                domain_id=domain_id,
            )
        except Exception as exc:
            diagnostic = _exception_payload(exc)
            _emit_research_diagnostic(
                "domain_execution_exception",
                domain_id=domain_id,
                exception=diagnostic,
            )
            raise
        domain_notes.append(
            _validate_domain_result(raw_note, domain_id=domain_id)
        )

    payload: dict[str, Any] = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": research_brief,
        "deterministic": deterministic,
        "domain_notes": domain_notes,
        "errors": errors,
        "method": {
            "reason_act": "target-neutral host evidence collection before design",
            "adaptive_retrieval": (
                "query-scoped claim-bearing evidence is paged to the model within "
                "bounded context"
            ),
            "corrective_retrieval": (
                "one grounded retrieval owner supplies local, official and external "
                "source evidence"
            ),
            "reflection": (
                "final sufficient claims must cite exact host-owned evidence page refs"
            ),
            "planning_search": (
                "pre-design source discovery is grounded; donor selection remains a "
                "frozen-design reuse decision"
            ),
            "minecraft_knowledge": (
                "version-sensitive routes are explicit deferred work until platform "
                "target freeze"
            ),
        },
    }
    payload["minecraft_knowledge_plan"] = knowledge_plan
    coverage = evaluate_route_coverage(knowledge_plan, payload)
    _emit_research_diagnostic(
        "minecraft_knowledge_route_coverage", coverage=coverage
    )
    if coverage["status"] != "PASS":
        raise PreDesignResearchFailure(
            "Minecraft knowledge route coverage blocked pre-design research: "
            + ", ".join(coverage.get("blocking_requirement_refs", ()))
        )
    payload["minecraft_knowledge_route_coverage"] = coverage
    payload = attach_procedural_skillbank(router, prompt, payload)
    payload = compose_research_skillbank(router, prompt, payload)
    payload["research_sha256"] = agentic._json_sha256(payload)
    model_view = _bounded_model_view(agentic, router, prompt, payload)
    _emit_research_diagnostic(
        "research_complete",
        research_sha256=payload["research_sha256"],
        domain_count=len(domain_notes),
        errors=errors,
        model_view_sha256=model_view.get("model_view_sha256"),
    )
    return model_view


__all__ = ["PreDesignResearchFailure", "collect_design_research"]
