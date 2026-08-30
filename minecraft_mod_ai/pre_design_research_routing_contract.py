from __future__ import annotations

"""Canonical pre-design research routing and grounded-RAG API compatibility.

Target-neutral design research uses the official corpus and the current-project index.
Explicit public source routes remain available through the grounded external retriever,
but donor/reuse selection stays outside this pre-design owner.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_PIPELINE_MARKER = "__mmm_canonical_pre_design_routing_v1__"
_AUGMENT_MARKER = "__mmm_augment_default_query_semantics_v1__"
_INSTALLED = False


def _route_names(research_brief: Any) -> set[str]:
    routes: set[str] = set()
    if not isinstance(research_brief, Mapping):
        return routes
    domains = research_brief.get("domains", ())
    if not isinstance(domains, Sequence) or isinstance(domains, (str, bytes, bytearray)):
        return routes
    for domain in domains:
        if not isinstance(domain, Mapping):
            continue
        for key in ("providers", "required_providers"):
            values = domain.get(key, ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                continue
            routes.update(str(value).strip().casefold() for value in values if str(value).strip())
    return routes


def _uses_explicit_public_source_route(research_brief: Any) -> bool:
    return bool(_route_names(research_brief) & {"github", "modrinth"})


def _payload_queries(payload: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    domains = payload.get("domains", ())
    if not isinstance(domains, Sequence) or isinstance(domains, (str, bytes, bytearray)):
        return ()
    for domain in domains:
        if not isinstance(domain, Mapping):
            continue
        queries = domain.get("queries", ())
        if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes, bytearray)):
            continue
        for raw in queries:
            value = raw.get("query") if isinstance(raw, Mapping) else raw
            text = " ".join(str(value or "").split()).strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                found.append(text)
    return tuple(found)


def _install_augment_default_semantics(grounded: Any) -> None:
    current = grounded._augment_bundle
    if getattr(current, _AUGMENT_MARKER, False):
        return

    @wraps(current)
    def augment_bundle(
        agentic_module: Any,
        payload: Mapping[str, Any],
        *,
        versions: Sequence[str],
        local_index: Mapping[str, Any],
        external_queries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selected = _payload_queries(payload) if external_queries is None else tuple(external_queries)
        return current(
            agentic_module,
            payload,
            versions=versions,
            local_index=local_index,
            external_queries=selected,
        )

    setattr(augment_bundle, _AUGMENT_MARKER, True)
    grounded._augment_bundle = augment_bundle


def _canonical_target_neutral_collect(
    pipeline: Any,
    router: Any,
    prompt: str,
    research_brief: Mapping[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from . import agentic_pre_design_rag as project_rag
    from . import agentic_research_game_design as agentic

    knowledge_plan = pipeline.compile_minecraft_knowledge_plan(prompt)
    deterministic: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    pipeline._emit_research_diagnostic(
        "research_start",
        prompt=prompt,
        research_brief=research_brief,
        minecraft_knowledge_plan=knowledge_plan,
    )

    deterministic["official_rag"] = pipeline._target_neutral_official_evidence(research_brief)
    pipeline._emit_research_diagnostic(
        "deterministic_stage_complete",
        stage="official_rag",
        result=deterministic["official_rag"],
    )

    deterministic["forced_project_rag"] = pipeline.collect_local_project_evidence(research_brief)
    pipeline._emit_research_diagnostic(
        "deterministic_stage_complete",
        stage="forced_project_rag",
        result=deterministic["forced_project_rag"],
    )

    target_frozen = pipeline._target_frozen(knowledge_plan)
    if target_frozen:
        try:
            deterministic["technology_radar"] = pipeline.collect_technology_radar(
                prompt,
                research_brief,
                page_size=50,
                page_builder=pipeline.build_technology_radar,
            )
            pipeline._emit_research_diagnostic(
                "deterministic_stage_complete",
                stage="technology_radar",
                result=deterministic["technology_radar"],
            )
        except Exception as exc:  # noqa: BLE001 - provider failure remains visible in the ledger
            diagnostic = pipeline._exception_payload(exc)
            errors.append(
                {"stage": "technology_radar", "error": f"{type(exc).__name__}: {exc}"}
            )
            deterministic["technology_radar"] = {
                "status": "unavailable",
                "failure": diagnostic,
            }
            pipeline._emit_research_diagnostic(
                "deterministic_stage_failure",
                stage="technology_radar",
                exception=diagnostic,
            )
    else:
        deterministic["technology_radar"] = pipeline._deferred_technology_receipt(knowledge_plan)
        pipeline._emit_research_diagnostic(
            "deterministic_stage_deferred",
            stage="technology_radar",
            result=deterministic["technology_radar"],
        )

    domain_notes: list[dict[str, Any]] = []
    domains = research_brief.get("domains", [])
    for domain in domains if isinstance(domains, list) else []:
        if not isinstance(domain, dict):
            continue
        domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
        domain_evidence = {
            source: agentic._domain_source_value(domain_id, value)
            for source, value in deterministic.items()
        }
        document = project_rag._materialize_domain_evidence_document(
            domain_id,
            domain_evidence,
        )
        with pipeline.target_neutral_research_scope():
            raw_note = pipeline.research_document_domain(
                agentic,
                project_rag,
                router,
                prompt=prompt,
                domain=domain,
                document=document,
                trace_metadata=trace_metadata,
            )
        pipeline._validate_document_grounding(
            agentic,
            project_rag,
            raw_note,
            document,
            domain_id=domain_id,
        )
        domain_notes.append(pipeline._validate_domain_result(raw_note, domain_id=domain_id))

    payload: dict[str, Any] = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": dict(research_brief),
        "deterministic": deterministic,
        "domain_notes": domain_notes,
        "errors": errors,
        "method": {
            "reason_act": "target-neutral host evidence collection before design",
            "adaptive_retrieval": "official and current-project evidence are collected once before design synthesis",
            "corrective_retrieval": "explicit public-source routes use the grounded external owner instead of the local-project lane",
            "reflection": "final sufficient claims must cite exact host-owned evidence page refs",
            "planning_search": "donor selection remains a frozen-design reuse decision",
            "minecraft_knowledge": "version-sensitive routes remain deferred until platform target freeze",
        },
    }
    payload["minecraft_knowledge_plan"] = knowledge_plan
    coverage = pipeline.evaluate_route_coverage(knowledge_plan, payload)
    pipeline._emit_research_diagnostic("minecraft_knowledge_route_coverage", coverage=coverage)
    if coverage["status"] != "PASS":
        raise pipeline.PreDesignResearchFailure(
            "Minecraft knowledge route coverage blocked pre-design research: "
            + ", ".join(coverage.get("blocking_requirement_refs", ()))
        )
    payload["minecraft_knowledge_route_coverage"] = coverage
    payload = pipeline.attach_procedural_skillbank(router, prompt, payload)
    payload = pipeline.compose_research_skillbank(router, prompt, payload)
    payload["research_sha256"] = agentic._json_sha256(payload)
    model_view = pipeline._bounded_model_view(agentic, router, prompt, payload)
    pipeline._emit_research_diagnostic(
        "research_complete",
        research_sha256=payload["research_sha256"],
        domain_count=len(domain_notes),
        errors=errors,
        model_view_sha256=model_view.get("model_view_sha256"),
    )
    return model_view


def install() -> None:
    """Install one phase-aware pre-design owner and preserve the grounded public API."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import pre_design_research_pipeline as pipeline
    from . import research_grounded_rag_contract as grounded
    from .pre_design_local_project_evidence import collect_local_project_evidence

    _install_augment_default_semantics(grounded)
    pipeline.collect_local_project_evidence = collect_local_project_evidence

    current = pipeline.collect_design_research
    if not getattr(current, _PIPELINE_MARKER, False):

        @wraps(current)
        def collect_design_research(
            router: Any,
            prompt: str,
            *,
            trace_metadata: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            research_brief = pipeline._pre_design_brief(prompt)
            if _uses_explicit_public_source_route(research_brief):
                return current(
                    router,
                    prompt,
                    trace_metadata=trace_metadata,
                )
            return _canonical_target_neutral_collect(
                pipeline,
                router,
                prompt,
                research_brief,
                trace_metadata=trace_metadata,
            )

        setattr(collect_design_research, _PIPELINE_MARKER, True)
        pipeline.collect_design_research = collect_design_research

    _INSTALLED = True


__all__ = ["install"]
