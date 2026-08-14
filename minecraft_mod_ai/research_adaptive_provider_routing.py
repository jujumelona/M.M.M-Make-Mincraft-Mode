from __future__ import annotations

"""Late adaptive routing for pre-design research providers.

The normalized research brief is already the host-owned routing graph. Keep the
planning critical path small: avoid duplicate authoritative retrieval, skip the
technology radar when the request has no technology capability, and defer external
ecosystem discovery until a research-domain agent actually has an evidence gap.
"""

import hashlib
import json
from functools import wraps
from typing import Any, Mapping


_PROVIDER_MARKER = "_mmm_adaptive_research_provider_routing_v1"
_FORCED_ROUTE_MARKER = "_mmm_official_owner_forced_rag_route_v1"

_EXTERNAL_PROVIDERS = frozenset(
    {
        "modrinth",
        "github",
        "openverse_images",
        "openverse_audio",
        "wikipedia",
        "huggingface_models",
        "openalex_works",
        "crossref_works",
    }
)
_TECHNOLOGY_EVIDENCE_KINDS = frozenset(
    {
        "ai_inference",
        "agent_tool_use",
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_conversion",
        "translation",
        "model_runtime",
        "model_license",
        "dataset_provenance",
        "consent_privacy",
        "latency_budget",
    }
)
_CODE_EVIDENCE_KINDS = frozenset({"source_code", "local_project"})


def _strings(value: Any) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item).strip() for item in value if str(item).strip())


def _domains(research_brief: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(research_brief, Mapping):
        return ()
    raw = research_brief.get("domains")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _needs_technology_radar(research_brief: Mapping[str, Any] | None) -> bool:
    return any(
        bool(_strings(domain.get("evidence_kinds")) & _TECHNOLOGY_EVIDENCE_KINDS)
        for domain in _domains(research_brief)
    )


def _external_route_count(research_brief: Mapping[str, Any] | None) -> int:
    seen: set[tuple[str, str, str]] = set()
    for domain in _domains(research_brief):
        domain_id = str(domain.get("domain_id", "")).strip()
        providers = _strings(domain.get("providers")) & _EXTERNAL_PROVIDERS
        queries = domain.get("queries")
        if not isinstance(queries, list):
            continue
        for provider in providers:
            for query in queries:
                query_text = str(query).strip()
                if query_text:
                    seen.add((domain_id, provider, query_text))
    return len(seen)


def _sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _technology_not_required(
    prompt: str,
    research_brief: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "mmm/technology-radar-not-required-v1",
        "status": "not_required",
        "requirements": [],
        "errors": [],
        "routing": {
            "policy": "run_only_for_technology_capability_evidence",
            "technology_evidence_kinds": sorted(_TECHNOLOGY_EVIDENCE_KINDS),
        },
        "request_sha256": _sha256(prompt),
        "brief_sha256": str(
            research_brief.get("brief_sha256", "")
            if isinstance(research_brief, Mapping)
            else ""
        ),
    }
    payload["radar_sha256"] = _sha256(payload)
    return payload


def _ecosystem_deferred(
    prompt: str,
    research_brief: Mapping[str, Any] | None,
) -> dict[str, Any]:
    route_count = _external_route_count(research_brief)
    status = "deferred" if route_count else "not_required"
    route_basis = {
        "brief_sha256": str(
            research_brief.get("brief_sha256", "")
            if isinstance(research_brief, Mapping)
            else ""
        ),
        "route_count": route_count,
    }
    payload: dict[str, Any] = {
        "schema_version": "mmm/ecosystem-planning-deferred-v1",
        "status": status,
        "phase": "planning",
        "candidate_count": 0,
        "processed_route_count": 0,
        "remaining_route_count": route_count,
        "routes_complete": route_count == 0,
        "route_count": route_count,
        "route_sha256": _sha256(route_basis),
        "query_sha256": _sha256(prompt),
        "pages": [],
        "errors": [],
        "coverage": (
            "external discovery is deferred until a domain agent reports a relevant "
            "evidence gap or validation requires a specialist lookup"
        ),
        "routing": {
            "policy": "defer_external_provider_io_until_gap",
            "external_provider_count": len(_EXTERNAL_PROVIDERS),
        },
    }
    return payload


def harden(agentic_module: Any, adaptive_rag_module: Any) -> None:
    """Install provider-level routing without replacing the research agent loop."""

    current_route = adaptive_rag_module._route
    if not getattr(current_route, _FORCED_ROUTE_MARKER, False):

        @wraps(current_route)
        def owner_aware_route(
            domain: Mapping[str, Any],
            *,
            code_index_available: bool,
        ) -> dict[str, Any]:
            routed = dict(
                current_route(domain, code_index_available=code_index_available)
            )
            providers = _strings(domain.get("providers"))
            if "official_docs" not in providers or not routed.get("catalog"):
                return routed

            original_kinds = _strings(domain.get("evidence_kinds"))
            code_kinds = original_kinds & _CODE_EVIDENCE_KINDS
            routed["catalog"] = False
            # The forced-RAG lane should expose only evidence it still owns. This
            # also prevents its CRAG correction branch from re-querying the project
            # catalog after an explicit code-RAG miss when official_docs already owns
            # authoritative API/reference evidence for the same domain.
            routed["evidence_kinds"] = code_kinds
            routed["reason"] = (
                "official_docs_owns_catalog_code_routed"
                if routed.get("code")
                else "official_docs_owns_catalog"
            )
            return routed

        setattr(owner_aware_route, _FORCED_ROUTE_MARKER, True)
        owner_aware_route.__wrapped__ = current_route
        adaptive_rag_module._route = owner_aware_route

    current_technology = agentic_module.collect_technology_radar
    if not getattr(current_technology, _PROVIDER_MARKER, False):

        @wraps(current_technology)
        def adaptive_technology(
            prompt: str,
            research_brief: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if research_brief is None or _needs_technology_radar(research_brief):
                return current_technology(prompt, research_brief, **kwargs)
            return _technology_not_required(prompt, research_brief)

        setattr(adaptive_technology, _PROVIDER_MARKER, True)
        adaptive_technology.__wrapped__ = current_technology
        agentic_module.collect_technology_radar = adaptive_technology

    current_ecosystem = agentic_module.collect_ecosystem_seed_bundle
    if not getattr(current_ecosystem, _PROVIDER_MARKER, False):

        @wraps(current_ecosystem)
        def adaptive_ecosystem(
            prompt: str,
            game_design: dict[str, Any],
            *,
            research_brief: dict[str, Any] | None = None,
            planning_seed_only: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if planning_seed_only and research_brief is not None:
                return _ecosystem_deferred(prompt, research_brief)
            return current_ecosystem(
                prompt,
                game_design,
                research_brief=research_brief,
                planning_seed_only=planning_seed_only,
                **kwargs,
            )

        setattr(adaptive_ecosystem, _PROVIDER_MARKER, True)
        adaptive_ecosystem.__wrapped__ = current_ecosystem
        agentic_module.collect_ecosystem_seed_bundle = adaptive_ecosystem


__all__ = ["harden"]
