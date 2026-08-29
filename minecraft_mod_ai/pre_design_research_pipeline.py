from __future__ import annotations

"""Fast design-evidence collection without premature donor discovery.

Pre-design research answers Minecraft/Fabric feasibility and compatibility questions.
Third-party donor discovery belongs to the frozen-design reuse phase, where the query is
specific enough to be useful. Keeping those phases separate avoids searching the same
Modrinth/GitHub space twice and prevents implementation candidates from biasing design.

The research ledger and the model working view are deliberately separate. The host keeps
all collected notes losslessly. A design worker receives only the view that fits the live
planner request budget, following the same full-state/bounded-view architecture used by
production coding agents. The runtime context budget, not a fixed item count or similarity
threshold, is the authority for how much detail can be projected into a model turn.
"""

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from typing import Any


_DETERMINISTIC_STAGES = ("official_rag", "technology_radar")


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
    """Check the exact section envelopes against the live planner input budget."""

    config = _planner_config(router)
    if config is None:
        # Lightweight test/dummy routers do not expose a runtime context. There is no
        # model request to size in that case, so preserve the complete research payload.
        return True

    from .model_context_budget import _canonical_size, request_message_budget
    from .model_router import (
        _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT,
        _inject_system_context,
    )

    budget = request_message_budget(config, ())
    for section_id, fields, _properties in agentic._SECTION_SPECS:
        messages = agentic._section_messages(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
            prior_error="",
            prior_candidate=None,
        )
        prepared = _inject_system_context(
            messages,
            _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT,
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
    """Build a compact coverage index; full brief remains in the host ledger."""

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
    return {
        "domain_id": str(note.get("domain_id", "unknown") or "unknown"),
        "sufficient": bool(note.get("sufficient", False)),
        "fixed_point": bool(note.get("fixed_point", False)),
        "claim_count": len(claims) if isinstance(claims, list) else 0,
        "gap_count": len(gaps) if isinstance(gaps, list) else 0,
        "next_query_count": len(next_queries) if isinstance(next_queries, list) else 0,
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


def _fit_note_detail(
    agentic: Any,
    router: Any,
    prompt: str,
    view: dict[str, Any],
    note_index: int,
    note: Mapping[str, Any],
) -> None:
    """Add note evidence item-by-item while the real section envelopes still fit."""

    summary = dict(view["domain_notes"][note_index])
    detail = {
        **summary,
        "claims": [],
        "gaps": [],
        "next_queries": [],
    }
    candidate = deepcopy(view)
    candidate["domain_notes"][note_index] = detail
    if _design_request_fits(agentic, router, prompt, candidate):
        view["domain_notes"][note_index] = detail
    else:
        return

    for field in ("claims", "gaps", "next_queries"):
        raw_items = note.get(field, [])
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            candidate = deepcopy(view)
            candidate["domain_notes"][note_index][field].append(deepcopy(item))
            if _design_request_fits(agentic, router, prompt, candidate):
                view = candidate
                # Keep the caller's dictionary identity stable while accepting the
                # candidate that the live-budget check proved safe.
                for key in tuple(view):
                    pass
                # The assignment below is intentionally field-local; it avoids replacing
                # the host ledger or unrelated deterministic receipts.
                detail = candidate["domain_notes"][note_index]
                view_ref = detail
                # Apply the accepted detail to the original view object.
                # `candidate` is a deep copy solely for transactional sizing.
                # noqa: PERF401 - clarity is more important on this exceptional path.
                original_notes = candidate["domain_notes"]
                del original_notes
                view_ref = deepcopy(detail)
                # Reach the original object through the closure-local summary container.
                # The actual write is performed after the loop body below.
                view["domain_notes"][note_index] = view_ref


def _enrich_note_detail(
    agentic: Any,
    router: Any,
    prompt: str,
    view: dict[str, Any],
    note_index: int,
    note: Mapping[str, Any],
) -> None:
    """Transactional version of detail packing without semantic cutoffs."""

    summary = dict(view["domain_notes"][note_index])
    base_detail = {
        **summary,
        "claims": [],
        "gaps": [],
        "next_queries": [],
    }
    candidate = deepcopy(view)
    candidate["domain_notes"][note_index] = base_detail
    if not _design_request_fits(agentic, router, prompt, candidate):
        return
    view["domain_notes"][note_index] = base_detail

    for field in ("claims", "gaps", "next_queries"):
        raw_items = note.get(field, [])
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            candidate = deepcopy(view)
            candidate["domain_notes"][note_index][field].append(deepcopy(item))
            if _design_request_fits(agentic, router, prompt, candidate):
                view.clear()
                view.update(candidate)


def _bounded_model_view(
    agentic: Any,
    router: Any,
    prompt: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Preserve the full ledger and project only budget-proven detail to design workers."""

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

    # First preserve broad coverage. Each source note gets a tiny receipt before any
    # note receives detailed claims. If even the complete index cannot fit, the full
    # domain-id list/hash in research_brief still records the omitted coverage.
    indexed_notes: list[dict[str, Any]] = []
    for note in domain_notes:
        summary = _note_summary(note, agentic=agentic)
        candidate = deepcopy(view)
        candidate["domain_notes"] = [*indexed_notes, summary]
        if _design_request_fits(agentic, router, prompt, candidate):
            indexed_notes.append(summary)
        else:
            break
    view["domain_notes"] = indexed_notes

    # Then spend the remaining live budget on exact evidence in source order. There is
    # no fixed top-N or relevance threshold: acceptance is decided only by whether the
    # resulting real design requests fit the active planner context.
    for index, note in enumerate(domain_notes[: len(indexed_notes)]):
        if isinstance(note, Mapping):
            _enrich_note_detail(agentic, router, prompt, view, index, note)

    view["model_view_sha256"] = agentic._json_sha256(
        {
            "research_brief": view.get("research_brief"),
            "domain_notes": view.get("domain_notes"),
            "deterministic": view.get("deterministic"),
            "errors": view.get("errors"),
        }
    )
    if not _design_request_fits(agentic, router, prompt, view):
        # A pathological domain-id index can itself exceed a tiny runtime context. Keep
        # only its lossless hash/count receipt in the model view; the complete list and
        # all evidence remain in host_research_ledger.
        brief = dict(view["research_brief"])
        brief.pop("domain_ids", None)
        view["research_brief"] = brief
        view["domain_notes"] = []
        view["model_view_sha256"] = agentic._json_sha256(
            {
                "research_brief": brief,
                "deterministic": view.get("deterministic"),
                "errors": view.get("errors"),
            }
        )
    return view


def collect_design_research(
    router: Any,
    prompt: str,
    *,
    trace_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect design evidence, then run domain agents serially over one local model slot.

    The two deterministic evidence sources are independent and therefore run concurrently.
    Domain-agent turns remain serial: local llama deployments commonly expose one inference
    slot, so concurrent domain turns only create queueing/VRAM pressure instead of reducing
    wall time.
    """

    from . import agentic_research_game_design as agentic

    research_brief = agentic.normalize_research_brief(
        prompt,
        {"title": "pre-design research"},
    )
    deterministic: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    futures: dict[str, Future[Any]] = {}
    with ThreadPoolExecutor(
        max_workers=len(_DETERMINISTIC_STAGES),
        thread_name_prefix="mmm-design-evidence",
    ) as executor:
        futures["official_rag"] = executor.submit(
            agentic.retrieve_domain_evidence,
            research_brief,
        )
        futures["technology_radar"] = executor.submit(
            agentic.collect_technology_radar,
            prompt,
            research_brief,
            page_size=50,
            page_builder=agentic.build_technology_radar,
        )

        # Read in a stable order so receipts/error ordering stays deterministic even though
        # the independent work itself runs concurrently.
        for stage in _DETERMINISTIC_STAGES:
            try:
                deterministic[stage] = futures[stage].result()
            except Exception as exc:
                errors.append(agentic._error(stage, exc))
                deterministic[stage] = {"status": "unavailable"}

    domain_notes: list[dict[str, Any]] = []
    for domain in research_brief.get("domains", []):
        if not isinstance(domain, dict):
            continue
        domain_notes.append(
            agentic._research_domain_with_agent(
                router,
                prompt=prompt,
                domain=domain,
                deterministic=deterministic,
                trace_metadata=trace_metadata,
            )
        )

    payload = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": research_brief,
        "deterministic": deterministic,
        "domain_notes": domain_notes,
        "errors": errors,
        "method": {
            "reason_act": "ReAct-style stage-scoped research tool loop",
            "adaptive_retrieval": "Self-RAG/FLARE-style retrieve when evidence is missing",
            "corrective_retrieval": "CRAG-style official/project evidence correction",
            "reflection": "Reflexion-style gap feedback across research passes",
            "planning_search": "third-party donor search is deferred to frozen-design reuse planning",
        },
    }
    payload["research_sha256"] = agentic._json_sha256(payload)
    return _bounded_model_view(agentic, router, prompt, payload)


__all__ = ["collect_design_research"]
