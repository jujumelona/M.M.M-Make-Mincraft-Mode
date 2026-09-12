from __future__ import annotations

"""Resolve planning-state unknowns with route-appropriate grounded evidence.

The planning-state SSOT owns why information is needed before retrieval. Query compilation
and provider routing are deterministic host work: the small model and generic runtime
research wrappers never choose or augment queries, providers, or fallback paths.
Reference/world knowledge is target-neutral; Minecraft implementation research uses
catalog-first mod discovery followed by exact source/API/project evidence.
"""

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .catalog_first_grounded_rag import forced_rag_bundle
from .deadline_executor import (
    ParallelExecutionTimeout,
    ParallelTaskError,
    iter_completed_with_deadlines,
)
from .model_concurrency import router_native_model_parallelism
from .planning_candidate_evidence import (
    expansion_queries,
    fingerprint,
    global_grounded_pool,
    query_context,
    requirement_candidate_trace,
    requirement_for,
)
from .planning_mod_discovery import catalog_queries, discovery_receipt
from .planning_state_contract import validate_planning_state
from .pre_design_domain_research import research_document_domain
from .research_reuse_candidates import (
    merge_repository_candidates,
    project_repository_candidates,
)
from .root_cause_trace import emit_root_cause
from .spec import canonical_json

_DEFAULT_SCOPE_POLICY = (
    "When authored scope is unspecified, select only an externally evidenced, coherent "
    "end-to-end gameplay slice that preserves the reference's distinctive loop. Never "
    "claim a full clone and never invent unevidenced reference features."
)
# `web_sources` is deliberately NOT a reference source. Conflating the two previously
# routed arbitrary external-fact questions through Wikipedia and allowed unrelated pages
# to masquerade as game/reference evidence.
_REFERENCE_SOURCE_KINDS = frozenset({"reference_sources"})
_GENERIC_WEB_SOURCE_KINDS = frozenset({"web_sources"})
_PROVIDER_RECEIPT_FIELDS = (
    "provider",
    "status",
    "result_count",
    "provider_total",
    "search_requests",
    "source_requests",
    "authenticated",
    "policy",
    "languages",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _reference_names_for_research(
    state: Mapping[str, Any], research: Mapping[str, Any]
) -> list[str]:
    """Return only named references explicitly anchored to this research objective."""
    basis = " ".join(
        part
        for part in (
            _text(research.get("objective")),
            _text(research.get("information_needed")),
        )
        if part
    ).casefold()
    references = state.get("references")
    names: list[str] = []
    for item in references if isinstance(references, list) else []:
        if not isinstance(item, Mapping):
            continue
        name = _text(item.get("name"))
        if name and name.casefold() in basis:
            names.append(name)
    return list(dict.fromkeys(names))


def _query_context(state: Mapping[str, Any], research: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "objective": _text(research.get("objective")),
        "information_needed": _text(research.get("information_needed")),
        "source_kinds": [
            _text(item) for item in research.get("source_kinds", []) if _text(item)
        ] if isinstance(research.get("source_kinds"), list) else [],
        "reference_names": _reference_names_for_research(state, research),
    }


def _bounded_query(value: str) -> str:
    return " ".join(str(value or "").split()).strip()[:420]


def _compile_queries(
    _router: Any,
    state: Mapping[str, Any],
    research: Mapping[str, Any],
) -> list[str]:
    """Derive bounded queries from SSOT information needs; never ask a model to invent them."""
    context = _query_context(state, research)
    objective = context["objective"]
    needed = context["information_needed"]
    source_kinds = set(context["source_kinds"])
    reference_names = context["reference_names"]
    candidates: list[str] = []

    if source_kinds & _REFERENCE_SOURCE_KINDS:
        # Reference research must be identity-anchored. Do not silently borrow names from
        # unrelated references in the same request.
        for name in reference_names:
            candidates.append(f"{name} {needed or objective}")
            candidates.append(f"{name} documented systems behavior rules")
        if not reference_names:
            # Keep a deterministic query for diagnostics; `_research_brief` will mark the
            # route unsupported rather than issuing retrieval without an identity anchor.
            if objective:
                candidates.append(objective)
            if needed and needed != objective:
                candidates.append(needed)
    else:
        if objective:
            candidates.append(objective)
        if needed and needed != objective:
            candidates.append(needed)
        if source_kinds & {"minecraft_docs", "minecraft_source"}:
            basis = objective or needed
            if basis:
                candidates.append(f"Minecraft Fabric API source {basis}")
        if source_kinds & {"repository", "existing_mods"}:
            basis = objective or needed
            if basis:
                candidates.append(f"Minecraft mod {basis}")

    queries = list(
        dict.fromkeys(
            query for candidate in candidates if (query := _bounded_query(candidate))
        )
    )
    if not queries:
        raise ValueError("PLANNING_RESEARCH_QUERY: host information need produced no query")
    return queries


def _providers_for(source_kinds: Sequence[str]) -> list[str]:
    """Return exact HOST-owned providers; never cross-fallback between evidence classes."""
    kinds = set(source_kinds)
    if kinds & _REFERENCE_SOURCE_KINDS:
        return ["wikipedia"]
    if kinds and kinds <= _GENERIC_WEB_SOURCE_KINDS:
        # There is currently no dedicated generic-web evidence provider in this pipeline.
        # Empty means fail closed; never substitute Wikipedia or Minecraft RAG.
        return []

    providers: list[str] = []
    if kinds & {"repository", "existing_mods"}:
        providers.extend(["curseforge", "modrinth"])
    if kinds & {"minecraft_docs", "minecraft_source"}:
        providers.extend(["official_docs", "project_rag"])
    if "project_rag" in kinds:
        providers.append("project_rag")
    return list(dict.fromkeys(providers)) or ["project_rag", "official_docs"]


def _evidence_kinds_for(source_kinds: Sequence[str]) -> list[str]:
    kinds = set(source_kinds)
    output: list[str] = []
    if kinds & _REFERENCE_SOURCE_KINDS:
        output.append("gameplay_reference")
    elif kinds and kinds <= _GENERIC_WEB_SOURCE_KINDS:
        output.append("external_fact")
    if kinds & {"repository", "existing_mods"}:
        output.extend(["dependency", "source_code"])
    if kinds & {"minecraft_docs", "minecraft_source"}:
        output.append("minecraft_api")
    if "project_rag" in kinds:
        output.append("local_project")
    return list(dict.fromkeys(output)) or ["scholarly_reference"]


def _compile_pending_queries(router: Any, state: dict[str, Any]) -> None:
    for research in state.get("research_queue", []):
        if not isinstance(research, dict) or str(research.get("status") or "") != "pending":
            continue
        existing = research.get("queries")
        if isinstance(existing, list) and any(_text(item) for item in existing):
            research["queries"] = list(
                dict.fromkeys(_text(item) for item in existing if _text(item))
            )
            continue
        research["queries"] = _compile_queries(router, state, research)


def _research_brief(
    prompt: str, state: Mapping[str, Any]
) -> tuple[dict[str, Any], set[str], dict[str, str]]:
    """Build the planning research brief with explicit source-class isolation.

    The provider/source route is already a host-owned consequence of planning-state
    ``source_kinds``. Passing this through central runtime wrappers would create a second
    authority that can silently add providers after the state machine has made its choice.
    """
    domains: list[dict[str, Any]] = []
    reference_domain_ids: set[str] = set()
    unsupported_domains: dict[str, str] = {}
    for raw in state.get("research_queue", []) if isinstance(state.get("research_queue"), list) else []:
        if not isinstance(raw, Mapping) or str(raw.get("status") or "") != "pending":
            continue
        queries = [
            _text(item) for item in raw.get("queries", []) if _text(item)
        ] if isinstance(raw.get("queries"), list) else []
        if not queries:
            raise ValueError(
                f"PLANNING_RESEARCH_QUERY: pending research {raw.get('research_id')!r} has no compiled queries"
            )
        source_kinds = [str(item) for item in raw.get("source_kinds", [])]
        kinds = set(source_kinds)
        domain_id = str(raw.get("research_id") or "")
        required_anchor_terms = _reference_names_for_research(state, raw)
        if kinds & _REFERENCE_SOURCE_KINDS:
            if required_anchor_terms:
                reference_domain_ids.add(domain_id)
            else:
                unsupported_domains[domain_id] = "reference_identity_anchor_missing"
        elif kinds and kinds <= _GENERIC_WEB_SOURCE_KINDS:
            unsupported_domains[domain_id] = "generic_web_provider_unconfigured"
        domains.append(
            {
                "domain_id": domain_id,
                "objective": _text(raw.get("objective")),
                "requirements": [_text(raw.get("information_needed"))],
                "evidence_kinds": _evidence_kinds_for(source_kinds),
                "queries": queries,
                "providers": _providers_for(source_kinds),
                **({"catalog_queries": catalog_queries(state, raw, prompt=prompt),
                    "task_query_context": query_context(state, raw, prompt),
                    "requirement": dict(requirement_for(state, raw))}
                   if kinds & {"repository", "existing_mods"} and raw.get("requirement_ref") else {}),
                "required_anchor_terms": required_anchor_terms,
                "depends_on": [],
            }
        )

    brief: dict[str, Any] = {
        "schema_version": "mmm/planning-research-brief-v1",
        "summary": "Resolve only the open information needs in the prompt-first planning state.",
        "origin": "planning_state_host",
        "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "domains": domains,
        "unresolved_questions": [
            str(item.get("question") or "")
            for item in state.get("unresolved", [])
            if isinstance(item, Mapping) and item.get("status") == "open"
        ],
        "routing_policy": "planning_state_source_kinds_are_final_provider_authority",
    }
    brief["brief_sha256"] = "sha256:" + hashlib.sha256(
        canonical_json(brief).encode("utf-8")
    ).hexdigest()
    return brief, reference_domain_ids, unsupported_domains


def _domain_note_by_id(
    notes: Sequence[Mapping[str, Any]], domain_id: str
) -> Mapping[str, Any] | None:
    return next(
        (note for note in notes if str(note.get("domain_id") or "") == domain_id),
        None,
    )


def _evidence_refs(note: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    claims = note.get("claims")
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, Mapping):
            continue
        raw = claim.get("evidence_refs") or claim.get("citations") or claim.get("source_refs")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            refs.extend(_text(item) for item in raw if _text(item))
    return list(dict.fromkeys(refs))


def _provider_diagnostics(grounded: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep provider usage proof without persisting credentials or raw response bodies."""
    rows: list[dict[str, Any]] = []
    raw_queries = grounded.get("queries")
    for query in raw_queries if isinstance(raw_queries, list) else []:
        if not isinstance(query, Mapping):
            continue
        raw_receipts = query.get("provider_receipts")
        providers: dict[str, dict[str, Any]] = {}
        if isinstance(raw_receipts, Mapping):
            for name, raw in raw_receipts.items():
                if not isinstance(raw, Mapping):
                    continue
                receipt = {
                    field: deepcopy(raw[field])
                    for field in _PROVIDER_RECEIPT_FIELDS
                    if field in raw
                }
                providers[str(name)] = receipt
        rows.append(
            {
                "query_sha256": _text(query.get("query_sha256")),
                "content_record_count": int(query.get("content_record_count") or 0),
                "providers": providers,
                "retrieval_errors": deepcopy(query.get("retrieval_errors") or []),
            }
        )
    return rows


def _provider_statuses(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for row in rows:
        providers = row.get("providers")
        if not isinstance(providers, Mapping):
            continue
        for name, receipt in providers.items():
            if isinstance(receipt, Mapping):
                statuses[str(name)] = _text(receipt.get("status")) or "unknown"
    return statuses


def _apply_scope_policy(state: dict[str, Any], unresolved: dict[str, Any]) -> None:
    unresolved["status"] = "resolved"
    state["resolved"].append(
        {
            "unresolved_id": unresolved["unresolved_id"],
            "resolution": _DEFAULT_SCOPE_POLICY,
            "basis": "host_default_policy",
            "evidence_refs": [],
        }
    )


def _rehash(state: dict[str, Any]) -> dict[str, Any]:
    from .planning_state_contract import _hash_without

    state["state_sha256"] = ""
    state["state_sha256"] = _hash_without(state, "state_sha256")
    return state


def _grounded_reference_domain(domain: Mapping[str, Any]) -> dict[str, Any]:
    from .reference_source_research import retrieve_reference_grounded_evidence

    queries = [
        _text(item) for item in domain.get("queries", []) if _text(item)
    ] if isinstance(domain.get("queries"), list) else []
    return retrieve_reference_grounded_evidence(queries)


def _blocked_note(domain_id: str, reason: str) -> dict[str, Any]:
    """Represent a route-policy refusal as ordinary insufficient research evidence."""
    return {
        "domain_id": domain_id,
        "claims": [],
        "gaps": [reason],
        "next_queries": [],
        "procedures": [],
        "sufficient": False,
        "fixed_point": False,
        "checkpoint": {"status": "blocked", "reason": reason},
        "research_failures": [reason],
        "source_body_count": 0,
        "host_grounded_evidence_card_count": 0,
        "evidence_extraction_status": reason,
    }


def collect_planning_state_research(
    router: Any,
    prompt: str,
    state: Mapping[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every pending host research item using its declared source route.

    ``blocked`` and ``complete`` are terminal research states. Re-entry processes only
    rows that are still ``pending``; it never revives blocked work automatically.
    """
    from . import agentic_research_game_design as agentic
    from . import pre_design_grounded_rag as project_rag
    from .agent_capability_context import target_neutral_research_scope
    from .pre_design_research_pipeline import (
        _grounded_domain_evidence,
        _validate_document_grounding,
    )

    validate_planning_state(state, prompt=prompt)
    value = deepcopy(dict(state))
    value["repository_candidates"] = merge_repository_candidates(
        value.get("repository_candidates")
        if isinstance(value.get("repository_candidates"), list)
        else [],
        [],
    )
    unresolved_by_id = {
        str(item.get("unresolved_id") or ""): item
        for item in value.get("unresolved", [])
        if isinstance(item, dict)
    }
    for unresolved in unresolved_by_id.values():
        if (
            unresolved.get("status") == "open"
            and unresolved.get("resolution_route") == "default_policy"
        ):
            _apply_scope_policy(value, unresolved)

    _compile_pending_queries(router, value)
    brief, reference_domain_ids, unsupported_domains = _research_brief(prompt, value)
    if not brief.get("domains"):
        return _rehash(value)

    minecraft_domains = [
        dict(domain)
        for domain in brief.get("domains", [])
        if isinstance(domain, Mapping)
        and str(domain.get("domain_id") or "") not in reference_domain_ids
        and str(domain.get("domain_id") or "") not in unsupported_domains
    ]
    minecraft_bundle: dict[str, Any] | None = None
    if minecraft_domains:
        minecraft_bundle = forced_rag_bundle(
            project_rag,
            router,
            {**brief, "domains": minecraft_domains},
        )

    domains = [
        dict(domain)
        for domain in brief.get("domains", [])
        if isinstance(domain, Mapping)
    ]
    runnable_domains = [
        domain
        for domain in domains
        if str(domain.get("domain_id") or "") not in unsupported_domains
    ]

    grounded_by_domain = {
        str(domain["domain_id"]): deepcopy(_grounded_domain_evidence(str(domain["domain_id"]), minecraft_bundle))
        for domain in minecraft_domains
    }
    prior_pool = value.get("task_candidate_pool") or {"queries": []}
    pool = global_grounded_pool({"prior": prior_pool, **grounded_by_domain})
    from .planning_semantic_research import review_requirement_sources
    semantic_traces = {}
    for domain in minecraft_domains:
        if "catalog_queries" in domain:
            trace = requirement_candidate_trace(domain["requirement"], pool)
            review = review_requirement_sources(router, domain["requirement"], pool, trace)
            semantic_traces[domain["domain_id"]] = {**trace, "semantic_review": review,
                                                  "coverage_complete": review["complete"]}
    # A corrective pass must change the actual provider search space. No retry count,
    # reworded error, or model assertion can revive an identical exhausted query set.
    corrective_domains = []
    for domain in minecraft_domains:
        if "catalog_queries" not in domain:
            continue
        if semantic_traces[domain["domain_id"]]["coverage_complete"]:
            continue
        executed = {project_rag._query_terms(q).casefold()
                    for searched_domain in minecraft_domains
                    for q in searched_domain.get("catalog_queries", [])}
        fresh = [q for q in expansion_queries(domain["task_query_context"])
                 if project_rag._query_terms(q).casefold() not in executed]
        if fresh:
            corrective_domains.append({**domain, "catalog_queries": fresh, "queries": []})
            emit_root_cause(
                "planning_research_state", stage="planning_state",
                operation="collect_planning_state_research", result="RETRY",
                reason="material_search_space_expansion",
                details={"research_ref": domain["domain_id"], "new_queries": fresh,
                         "previous_queries_sha256": fingerprint(sorted(executed)),
                         "next_queries_sha256": fingerprint(sorted(executed | {q.casefold() for q in fresh}))},
            )
    if corrective_domains:
        corrective_bundle = forced_rag_bundle(project_rag, router, {**brief, "domains": corrective_domains})
        for domain in corrective_domains:
            domain_id = str(domain["domain_id"])
            correction = _grounded_domain_evidence(domain_id, corrective_bundle)
            grounded_by_domain[domain_id]["queries"].extend(correction.get("queries", []))
        pool = global_grounded_pool({"prior": prior_pool, **grounded_by_domain})
        for domain in minecraft_domains:
            if "catalog_queries" in domain:
                trace = requirement_candidate_trace(domain["requirement"], pool)
                review = review_requirement_sources(router, domain["requirement"], pool, trace)
                semantic_traces[domain["domain_id"]] = {**trace, "semantic_review": review,
                                                      "coverage_complete": review["complete"]}
    from .planning_semantic_research import validate_semantic_review
    retained_traces = []
    for research in value.get("research_queue", []):
        if research.get("status") != "complete" or not research.get("candidate_trace"):
            continue
        requirement = requirement_for(value, research)
        previous = validate_semantic_review(requirement, prior_pool,
                                            research["candidate_trace"].get("semantic_review") or {})
        if not previous["complete"]:
            raise ValueError("RESEARCH_RESTORE_STALE: previous completed review is invalid")
        rebound = validate_semantic_review(requirement, pool, {**previous, "pool_sha256": fingerprint(pool)})
        if not rebound["complete"]:
            raise ValueError("RESEARCH_RESTORE_STALE: accepted source bodies changed")
        retained = {**requirement_candidate_trace(requirement, pool),
                    "semantic_review": rebound, "coverage_complete": True}
        research["candidate_trace"] = retained
        retained_traces.append(deepcopy(retained))
        for evidence in value.get("evidence", []):
            if evidence.get("research_ref") == research.get("research_id"):
                evidence["candidate_trace"] = deepcopy(retained)
    value["task_candidate_pool"] = deepcopy(pool)
    value["candidate_requirement_trace"] = retained_traces

    def research_domain(domain: Mapping[str, Any]) -> dict[str, Any]:
        domain_id = str(domain.get("domain_id") or "")
        discovery: dict[str, Any] | None = None
        candidate_trace: dict[str, Any] | None = None
        repository_candidates: list[dict[str, Any]] = []
        if domain_id in reference_domain_ids:
            grounded = _grounded_reference_domain(domain)
        else:
            if minecraft_bundle is None:
                raise ValueError(
                    "PLANNING_RESEARCH_ROUTE: Minecraft RAG bundle is unexpectedly absent"
                )
            grounded = deepcopy(grounded_by_domain[domain_id])
            repository_candidates = project_repository_candidates(domain, grounded)
            if "catalog_queries" in domain:
                # Reassess sibling candidates against this requirement before deciding
                # completion. Preserve the domain's own API/document provider route.
                own_source_ids = {(str(record.get("source_id") or ""), str(record.get("content") or ""))
                                  for query in grounded.get("queries", [])
                                  for record in query.get("evidence_records", [])}
                for query in pool["queries"]:
                    records = [record for record in query["evidence_records"]
                               if (str(record.get("source_id") or ""), str(record.get("content") or "")) not in own_source_ids]
                    if records:
                        grounded["queries"].append({**query, "evidence_records": records})
                        own_source_ids.update((str(record.get("source_id") or ""), str(record.get("content") or ""))
                                              for record in records)
            if "catalog_queries" in domain:
                discovery = discovery_receipt(domain_id, grounded)
                # Raw bodies live once in task_candidate_pool, not in four discovery
                # receipts and again in each mandatory planner prompt.
                for candidate in discovery["candidates"]:
                    candidate.pop("description", None)
                candidate_trace = semantic_traces[domain_id]
                review = candidate_trace["semantic_review"]
                admitted = {proof["source_id"] for proof in review["accepted_proofs"]}
                for query in grounded["queries"]:
                    query["evidence_records"] = [record for record in query.get("evidence_records", [])
                        if str(record.get("source_id") or "").split(":", 1)[0]
                        not in {"modrinth", "curseforge", "github"} or record.get("source_id") in admitted]
        provider_diagnostics = _provider_diagnostics(grounded)
        document = project_rag._materialize_domain_evidence_document(
            domain_id,
            {"grounded_rag": grounded},
        )
        with target_neutral_research_scope():
            note = research_document_domain(
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
            note,
            document,
            domain_id=domain_id,
        )
        return {
            "domain_id": domain_id,
            "note": dict(note),
            "repository_candidates": repository_candidates,
            "discovery": discovery,
            "provider_diagnostics": provider_diagnostics,
            "candidate_trace": candidate_trace,
        }

    domain_results: dict[str, dict[str, Any]] = {}
    if runnable_domains:
        workers = max(
            1,
            min(len(runnable_domains), router_native_model_parallelism(router)),
        )
        try:
            completed_domains = iter_completed_with_deadlines(
                runnable_domains,
                research_domain,
                max_workers=workers,
                stage="planning-research-domain",
                sort_key=lambda domain: str(domain.get("domain_id") or ""),
            )
            for _domain, result in completed_domains:
                domain_results[result["domain_id"]] = result
        except ParallelExecutionTimeout as exc:
            domain = exc.item if isinstance(exc.item, Mapping) else {}
            domain_id = str(domain.get("domain_id") or "unknown")
            reason = (
                "PLANNING_RESEARCH_TIMEOUT: bounded research-domain deadline expired for "
                f"{domain_id}: {exc}"
            )
            emit_root_cause(
                "planning_research_timeout",
                stage="planning_state",
                operation="collect_planning_state_research",
                result="FAIL",
                reason=reason,
                details={
                    "domain_id": domain_id,
                    "deadline_kind": exc.deadline_kind,
                    "elapsed_seconds": exc.elapsed_seconds,
                    "work_unit_timeout_seconds": exc.work_unit_timeout_seconds,
                    "policy": "retryable_abort_current_stage",
                },
            )
            raise TimeoutError(reason) from exc
        except ParallelTaskError as exc:
            domain = exc.item if isinstance(exc.item, Mapping) else {}
            domain_id = str(domain.get("domain_id") or "unknown")
            cause = exc.cause
            if isinstance(cause, (TimeoutError, ConnectionError, InterruptedError)):
                reason = (
                    "PLANNING_RESEARCH_TRANSPORT_INTERRUPTED: retryable research-domain "
                    f"interruption for {domain_id}: {type(cause).__name__}: {cause}"
                )
                emit_root_cause(
                    "planning_research_transport_interrupted",
                    stage="planning_state",
                    operation="collect_planning_state_research",
                    result="FAIL",
                    reason=reason,
                    details={
                        "domain_id": domain_id,
                        "policy": "retryable_abort_current_stage",
                    },
                )
                raise TimeoutError(reason) from cause
            raise cause
    notes: list[dict[str, Any]] = []
    discovery_by_domain: dict[str, dict[str, Any]] = {}
    provider_diagnostics_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain in domains:
        domain_id = str(domain.get("domain_id") or "")
        unsupported_reason = unsupported_domains.get(domain_id)
        if unsupported_reason:
            provider_diagnostics_by_domain[domain_id] = []
            notes.append(_blocked_note(domain_id, unsupported_reason))
            continue
        result = domain_results.get(domain_id)
        if result is None:
            raise ValueError(
                f"PLANNING_RESEARCH_ROUTE: research domain {domain_id!r} produced no result"
            )
        provider_diagnostics_by_domain[domain_id] = list(
            result["provider_diagnostics"]
        )
        notes.append(dict(result["note"]))
        value["repository_candidates"] = merge_repository_candidates(
            value.get("repository_candidates", []),
            result["repository_candidates"],
        )
        discovery = result.get("discovery")
        if isinstance(discovery, dict):
            discovery_by_domain[domain_id] = discovery
            # Emit this shallowly so the console does not hide provider outcomes
            # behind the planning-state snapshot's depth limit.
            emit_root_cause(
                "planning_mod_discovery", stage="planning_state",
                operation="collect_planning_state_research", result="OBSERVED",
                reason=discovery["status"], details=discovery,
            )

    for research in value.get("research_queue", []):
        if not isinstance(research, dict) or research.get("status") != "pending":
            continue
        research_id = str(research.get("research_id") or "")
        note = _domain_note_by_id(notes, research_id)
        if note is None:
            research["status"] = "blocked"
            continue
        refs = _evidence_refs(note)
        sufficient = (
            note.get("sufficient") is True
            and bool(note.get("claims"))
            and bool(refs)
        )
        discovery = discovery_by_domain.get(research_id)
        if discovery is not None:
            research["mod_discovery"] = deepcopy(discovery)
            trace = domain_results[research_id]["candidate_trace"]
            value["candidate_requirement_trace"].append(deepcopy(trace))
            # A generic API excerpt cannot fill a gameplay evidence gap. Match only
            # requirement-local candidate evidence; never append every retrieved ID.
            sufficient = sufficient and discovery["complete"] and trace["coverage_complete"]
            matched_refs = [proof["source_id"] for proof in trace["semantic_review"]["accepted_proofs"]]
            refs = list(dict.fromkeys([*refs, *matched_refs]))
            research["research_state"] = "COMPLETE" if sufficient else "RESEARCH_BLOCKED"
            research["search_space_sha256"] = fingerprint([
                (q.get("query"), sorted((q.get("provider_receipts") or {}).keys()))
                for q in grounded_by_domain[research_id].get("queries", [])])
            research["candidate_trace"] = deepcopy(trace)
            emit_root_cause(
                "planning_research_state", stage="planning_state",
                operation="collect_planning_state_research", result=research["research_state"],
                reason="requirement_candidate_evidence" if sufficient else "candidate_evidence_missing",
                details={"research_ref": research_id, "candidate_count": len(trace["candidates"]),
                         "missing_facets": trace["missing_facets"],
                         "missing_obligations": trace["semantic_review"]["missing_obligation_indices"],
                         "pool_sha256": trace["pool_sha256"]},
            )
        research["status"] = "complete" if sufficient else "blocked"
        provider_diagnostics = provider_diagnostics_by_domain.get(research_id, [])
        provider_statuses = _provider_statuses(provider_diagnostics)
        value["evidence"].append(
            {
                "research_ref": research_id,
                "claims": deepcopy(note.get("claims") or []),
                "evidence_refs": refs,
                "sufficient": sufficient,
                "source": "grounded_materialized_pages",
                **({"mod_discovery": deepcopy(discovery)} if discovery is not None else {}),
                **({"candidate_trace": deepcopy(research["candidate_trace"])} if discovery is not None else {}),
                "diagnostics": {
                    "source_body_count": int(note.get("source_body_count") or 0),
                    "evidence_card_count": int(
                        note.get("host_grounded_evidence_card_count") or 0
                    ),
                    "evidence_extraction_status": _text(
                        note.get("evidence_extraction_status")
                    ),
                    "research_failures": deepcopy(
                        note.get("research_failures") or []
                    ),
                    "provider_statuses": provider_statuses,
                    "provider_receipts": provider_diagnostics,
                },
            }
        )
        for unresolved_id in research.get("resolves", []):
            unresolved = unresolved_by_id.get(str(unresolved_id))
            if unresolved is None:
                continue
            if sufficient:
                unresolved["status"] = "resolved"
                claim_texts = [
                    _text(c.get("claim") or c.get("statement") or c.get("text"))
                    for c in (note.get("claims") or [])
                    if isinstance(c, Mapping)
                ]
                claim_texts = [t for t in claim_texts if t]
                resolution_text = (
                    " ".join(dict.fromkeys(claim_texts))
                    if claim_texts
                    else "Evidenced by grounded research."
                )
                value["resolved"].append(
                    {
                        "unresolved_id": unresolved["unresolved_id"],
                        "resolution": resolution_text,
                        "basis": "grounded_research",
                        "evidence_refs": refs,
                    }
                )
            else:
                reason = (
                    (discovery["status"] if discovery is not None and not discovery["complete"] else "")
                    or ("requirement_candidate_evidence_missing" if discovery is not None
                        and not research["candidate_trace"]["coverage_complete"] else "")
                    or _text(note.get("evidence_extraction_status"))
                    or "insufficient_grounded_evidence"
                )
                value["blockers"].append(
                    {
                        "blocker_id": f"b_{len(value['blockers']) + 1:03d}",
                        "unresolved_id": unresolved["unresolved_id"],
                        "statement": (
                            "Grounded research blocked: "
                            f"{reason}; source_bodies={int(note.get('source_body_count') or 0)}; "
                            f"evidence_cards={int(note.get('host_grounded_evidence_card_count') or 0)}; "
                            f"providers={provider_statuses}."
                        ),
                        "provider_diagnostics": deepcopy(provider_diagnostics),
                    }
                )

    resolved_ids = {
        uid for uid, row in unresolved_by_id.items() if row.get("status") == "resolved"
    }
    value["blockers"] = [
        row
        for row in value["blockers"]
        if row.get("unresolved_id") not in resolved_ids
    ]
    for candidate in value["repository_candidates"]:
        source_ids = set(candidate.get("source_ids") or [])
        candidate["requirement_traces"] = [
            deepcopy(row) for trace in value["candidate_requirement_trace"]
            for row in trace["candidates"] if row["source_id"] in source_ids
        ]
    return _rehash(value)


__all__ = ["collect_planning_state_research"]
