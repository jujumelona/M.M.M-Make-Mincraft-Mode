from __future__ import annotations

import hashlib
import json
from functools import wraps
from typing import Any, Mapping, Sequence


_EVIDENCE_FIELDS = (
    "module_ids",
    "asset_ids",
    "audio_ids",
    "acceptance_tests",
)
_PRODUCTION_KEYS = frozenset(
    {
        "modules",
        "assets",
        "audio",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
)
_AGENTIC_RISK_MARKERS = (
    "networking",
    "multiplayer",
    "custom_java",
    "integration",
    "dimension",
    "world_event",
    "ai_inference",
    "agent_tool_use",
    "speech",
    "migration",
    "persistence",
)
_QUERY_BOUNDARIES = frozenset("\n\r\t .!?;:,。！？；，、")


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _recent_catalog_ids(request: Any, field: str) -> set[str]:
    if not isinstance(request, Mapping):
        return set()
    receipt = request.get(field)
    if not isinstance(receipt, Mapping):
        return set()
    return _string_set(receipt.get("recent_ids", ()))


def _dependency_export_ids(request: Any) -> set[str]:
    if not isinstance(request, Mapping):
        return set()
    exports = request.get("dependency_exports")
    if not isinstance(exports, Mapping):
        return set()
    result: set[str] = set()
    for values in exports.values():
        result.update(_string_set(values))
    return result


def _produced_ids(page: Mapping[str, Any], field: str, id_field: str) -> set[str]:
    result: set[str] = set()
    values = page.get(field)
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get(id_field, "")).strip()
        if value:
            result.add(value)
    return result


def _evidence_has_declared_reference(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(_string_set(value.get(field, ())) for field in _EVIDENCE_FIELDS)


def _evidence_is_grounded(
    value: Any,
    *,
    module_ids: set[str],
    asset_ids: set[str],
    audio_ids: set[str],
    acceptance_tests: set[str],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        (_string_set(value.get("module_ids", ())) & module_ids)
        or (_string_set(value.get("asset_ids", ())) & asset_ids)
        or (_string_set(value.get("audio_ids", ())) & audio_ids)
        or (_string_set(value.get("acceptance_tests", ())) & acceptance_tests)
    )


def _sanitize_production_page(
    page: Mapping[str, Any],
    request: dict[str, Any] | str,
) -> dict[str, Any]:
    """Validate explicit evidence without weakening legacy host-owned bookkeeping."""
    result = dict(page)
    evidence_value = result.get("deliverable_evidence")
    if not isinstance(evidence_value, Mapping):
        return result

    completed = result.get("completed_deliverables")
    if not isinstance(completed, list):
        completed = []

    evidence = {
        str(key): dict(value)
        for key, value in evidence_value.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }
    result["deliverable_evidence"] = evidence

    current_modules = _produced_ids(result, "modules", "module_id")
    current_assets = _produced_ids(result, "assets", "asset_id")
    current_audio = _produced_ids(result, "audio", "sound_id")
    current_tests = _string_set(result.get("acceptance_tests", ()))

    valid_modules = (
        current_modules
        | _recent_catalog_ids(request, "known_module_catalog")
        | _dependency_export_ids(request)
    )
    valid_assets = current_assets | _recent_catalog_ids(
        request, "known_asset_catalog"
    )
    valid_audio = current_audio | _recent_catalog_ids(
        request, "known_audio_catalog"
    )

    supported: list[str] = []
    for raw in completed:
        deliverable = str(raw).strip()
        if not deliverable:
            continue
        if _evidence_is_grounded(
            evidence.get(deliverable),
            module_ids=valid_modules,
            asset_ids=valid_assets,
            audio_ids=valid_audio,
            acceptance_tests=current_tests,
        ):
            supported.append(deliverable)
    result["completed_deliverables"] = supported
    return result


def _is_production_decode(
    request: dict[str, Any] | str,
    expected_contracts: Sequence[frozenset[str]],
) -> bool:
    if not isinstance(request, Mapping) or "remaining_deliverables" not in request:
        return False
    return any(set(contract) == set(_PRODUCTION_KEYS) for contract in expected_contracts)


def _install_evidence_contract(complete_planner_module: Any) -> None:
    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_small_model_evidence_guard", False):
        return

    @wraps(current)
    def generate_with_evidence_guard(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        page = current(
            router,
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )
        if not _is_production_decode(request, expected_contracts):
            return page
        if "deliverable_evidence" not in page:
            return page
        sanitized = _sanitize_production_page(page, request)
        sanitized.pop("deliverable_evidence", None)
        return sanitized

    generate_with_evidence_guard._mmm_small_model_evidence_guard = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_evidence_guard


def _install_evidence_aware_scoring(agentic_module: Any) -> None:
    current = agentic_module._score_plan_page
    if getattr(current, "_mmm_evidence_aware_plan_score", False):
        return

    @wraps(current)
    def score_with_evidence(page: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        base_score, verifier = current(page)
        if "deliverable_evidence" not in page:
            return base_score, dict(verifier)

        completed = page.get("completed_deliverables")
        completed_values = (
            [str(item).strip() for item in completed if str(item).strip()]
            if isinstance(completed, list)
            else []
        )
        evidence = page.get("deliverable_evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}

        current_modules = _produced_ids(page, "modules", "module_id")
        current_assets = _produced_ids(page, "assets", "asset_id")
        current_audio = _produced_ids(page, "audio", "sound_id")
        current_tests = _string_set(page.get("acceptance_tests", ()))

        declared = 0
        grounded = 0
        unsupported = 0
        for deliverable in completed_values:
            item = evidence.get(deliverable)
            if _evidence_has_declared_reference(item):
                declared += 1
            else:
                unsupported += 1
            if _evidence_is_grounded(
                item,
                module_ids=current_modules,
                asset_ids=current_assets,
                audio_ids=current_audio,
                acceptance_tests=current_tests,
            ):
                grounded += 1

        score = base_score + 24.0 * grounded - 36.0 * unsupported
        details = {
            **dict(verifier),
            "declared_completion_evidence": declared,
            "grounded_completion_evidence": grounded,
            "unsupported_completion_claims": unsupported,
        }
        return score, details

    score_with_evidence._mmm_evidence_aware_plan_score = True  # type: ignore[attr-defined]
    agentic_module._score_plan_page = score_with_evidence


def _semantic_digest(value: Any) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        rendered = repr(value)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _semantic_planner_key(prompt: str, research_brief: Any) -> tuple[str, str]:
    return _semantic_digest(prompt), _semantic_digest(research_brief)


def _semantic_ecosystem_key(
    prompt: str,
    game_design: Any,
    research_brief: Any,
) -> tuple[str, str, str]:
    return (
        _semantic_digest(prompt),
        _semantic_digest(game_design),
        _semantic_digest(research_brief),
    )


def _install_semantic_keys(parallel_module: Any) -> None:
    """Keep stable keys for diagnostics without joining planner futures."""
    parallel_module._planner_key = _semantic_planner_key
    parallel_module._ecosystem_key = _semantic_ecosystem_key


def _maximal_planner_risk(request: Any, stage: str) -> bool:
    rendered = (
        request
        if isinstance(request, str)
        else json.dumps(request, ensure_ascii=False, sort_keys=True, default=str)
    )
    size_risk = len(rendered.encode("utf-8")) >= 12 * 1024
    lowered = (stage + "\n" + rendered).casefold()
    domain_risk = any(marker in lowered for marker in _AGENTIC_RISK_MARKERS)
    target_risk = False
    if isinstance(request, Mapping):
        targets = request.get("current_target_deliverables", ())
        target_risk = (
            isinstance(targets, Sequence)
            and not isinstance(targets, (str, bytes))
            and len(targets) >= 3
        )
    return bool(size_risk and domain_risk and target_risk)


def _install_trace_adaptive_search(agentic_module: Any) -> None:
    current = agentic_module._planner_candidate_count
    if getattr(current, "_mmm_trace_adaptive_width", False):
        return

    @wraps(current)
    def candidate_count(request: Any, stage: str) -> int:
        base = int(current(request, stage))
        if agentic_module._mode() != "auto":
            return base
        width = agentic_module._env_int("MMM_PLAN_SEARCH_WIDTH", 2, maximum=3)
        try:
            from .small_model_agent_policy import planner_search_width_hint

            hint = planner_search_width_hint(request, stage, maximum=width)
        except Exception:
            hint = None
        if hint is None:
            return base
        if hint > base:
            return int(hint)
        if hint < base and not _maximal_planner_risk(request, stage):
            return int(hint)
        return base

    candidate_count._mmm_trace_adaptive_width = True  # type: ignore[attr-defined]
    candidate_count.__wrapped__ = current  # type: ignore[attr-defined]
    agentic_module._planner_candidate_count = candidate_count


def _lossless_utf8_pages(text: str, max_bytes: int) -> tuple[str, ...]:
    """Split text into bounded semantic pages while preserving every code point."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not text:
        return ()
    if len(text.encode("utf-8")) <= max_bytes:
        return (text,)

    pages: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in text:
        size = len(character.encode("utf-8"))
        if size > max_bytes:
            raise ValueError("A UTF-8 code point exceeds the page byte budget.")
        if current and current_bytes + size > max_bytes:
            minimum = max_bytes // 2
            consumed = 0
            boundary = 0
            for index, value in enumerate(current, start=1):
                consumed += len(value.encode("utf-8"))
                if consumed >= minimum and value in _QUERY_BOUNDARIES:
                    boundary = index
            cut = boundary or len(current)
            pages.append("".join(current[:cut]))
            current = current[cut:]
            current_bytes = len("".join(current).encode("utf-8"))
            if current and current_bytes + size > max_bytes:
                pages.append("".join(current))
                current = []
                current_bytes = 0
        current.append(character)
        current_bytes += size
    if current:
        pages.append("".join(current))
    if "".join(pages) != text:
        raise RuntimeError("Lossless research query paging changed source text.")
    if any(len(page.encode("utf-8")) > max_bytes for page in pages):
        raise RuntimeError("Lossless research query page exceeded its byte budget.")
    return tuple(pages)


def _install_lossless_research_input(central_module: Any, ecosystem_module: Any) -> None:
    """Preserve authoritative research text and shard only execution-sized queries."""
    if not getattr(central_module._bounded_text, "_mmm_lossless_text", False):
        def full_text(value: str, *, field: str = "research text") -> str:
            del field
            return value

        full_text._mmm_lossless_text = True  # type: ignore[attr-defined]
        central_module._bounded_text = full_text

    current_domain = central_module._research_domain
    if not getattr(current_domain, "_mmm_lossless_query_pages", False):
        @wraps(current_domain)
        def domain_with_query_pages(value: Any):
            normalized = value
            if isinstance(value, dict) and isinstance(value.get("queries"), list):
                queries: list[Any] = []
                for raw in value["queries"]:
                    if (
                        isinstance(raw, str)
                        and raw.strip()
                        and len(raw.encode("utf-8")) > central_module._MAX_QUERY_BYTES
                    ):
                        queries.extend(
                            page
                            for page in _lossless_utf8_pages(
                                raw,
                                central_module._MAX_QUERY_BYTES,
                            )
                            if page.strip()
                        )
                    else:
                        queries.append(raw)
                normalized = {**value, "queries": queries}
            return current_domain(normalized)

        domain_with_query_pages._mmm_lossless_query_pages = True  # type: ignore[attr-defined]
        central_module._research_domain = domain_with_query_pages

    def full_seed_query(prompt: str, game_design: dict[str, Any]) -> str:
        parts = [
            prompt,
            str(game_design.get("title", "")),
            str(game_design.get("pitch", "")),
        ]
        for item in game_design.get("modules", []):
            if isinstance(item, dict):
                parts.append(str(item.get("reason") or item.get("name") or ""))
        for item in game_design.get("assets", []):
            if isinstance(item, dict):
                parts.append(str(item.get("brief") or ""))
        return " ".join(part.strip() for part in parts if part.strip())

    full_seed_query._mmm_lossless_seed_query = True  # type: ignore[attr-defined]
    ecosystem_module._seed_query = full_seed_query

    current_discover = ecosystem_module.discover_seed_bundle
    if getattr(current_discover, "_mmm_lossless_seed_routes", False):
        return

    @wraps(current_discover)
    def discover_seed_lossless(
        prompt: str,
        game_design: dict[str, Any],
        *,
        research_brief: dict[str, Any] | None = None,
        client: Any = None,
        route_cursor: str = "",
        route_limit: int = 12,
    ) -> dict[str, Any]:
        if research_brief is not None:
            return current_discover(
                prompt,
                game_design,
                research_brief=research_brief,
                client=client,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )

        query = ecosystem_module._seed_query(prompt, game_design)
        pages = _lossless_utf8_pages(query, central_module._MAX_QUERY_BYTES)
        if len(pages) <= 1:
            return current_discover(
                prompt,
                game_design,
                research_brief=None,
                client=client,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )

        providers = ["modrinth", "openverse_images", "openverse_audio"]
        if (client is not None and client.github_token) or __import__("os").environ.get("GITHUB_TOKEN"):
            providers.append("github")
        domains = [
            {
                "domain_id": f"request_page_{index + 1}",
                "objective": "Search one lossless page of the complete request.",
                "requirements": [page],
                "evidence_kinds": ["gameplay_reference"],
                "queries": [page],
                "providers": providers,
                "depends_on": [],
            }
            for index, page in enumerate(pages)
        ]
        synthetic_brief = {
            "schema_version": "mmm/central-research-brief-v1",
            "summary": "Lossless direct ecosystem query pages.",
            "origin": "lossless_direct_seed",
            "domains": domains,
            "unresolved_questions": [],
        }
        result = current_discover(
            prompt,
            game_design,
            research_brief=synthetic_brief,
            client=client,
            route_cursor=route_cursor,
            route_limit=route_limit,
        )
        result = dict(result)
        result["request_query_ingestion"] = {
            "schema_version": "mmm/ecosystem-query-ingestion-v1",
            "query_sha256": ecosystem_module._sha256_text(query),
            "query_byte_length": len(query.encode("utf-8")),
            "page_count": len(pages),
            "lossless": "".join(pages) == query,
            "pages": [
                {
                    "page_index": index,
                    "byte_length": len(page.encode("utf-8")),
                    "sha256": ecosystem_module._sha256_text(page),
                }
                for index, page in enumerate(pages)
            ],
        }
        return result

    discover_seed_lossless._mmm_lossless_seed_routes = True  # type: ignore[attr-defined]
    ecosystem_module.discover_seed_bundle = discover_seed_lossless


def _install_nonblocking_planner_research(
    complete_planner_module: Any,
    research_coordinator_module: Any,
    central_module: Any,
) -> None:
    """Remove planner-level future joins; inner provider/RAG work may remain parallel."""
    complete_planner_module.collect_technology_radar = (
        research_coordinator_module.collect_technology_radar
    )

    retrieve = complete_planner_module.retrieve_domain_evidence

    def implementation_evidence(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or complete_planner_module.normalize_research_brief(
            prompt,
            game_design,
        )
        return retrieve(brief)

    implementation_evidence._mmm_nonblocking_planner_research = True  # type: ignore[attr-defined]
    complete_planner_module._retrieve_implementation_evidence = implementation_evidence

    def ecosystem_seed(
        prompt: str,
        game_design: dict[str, Any],
        *,
        research_brief: dict[str, Any] | None = None,
        client: Any = None,
        route_limit: int = 12,
        page_builder: Any = None,
        planning_seed_only: bool = False,
    ) -> dict[str, Any]:
        if planning_seed_only and isinstance(research_brief, dict):
            routes = central_module.external_discovery_routes(research_brief)
            receipts = [
                {
                    "domain_id": str(route.get("domain_id", "")),
                    "provider": str(route.get("provider", "")),
                    "target_profile": str(route.get("target_profile", "")),
                    "query_sha256": central_module._sha256(str(route.get("query", ""))),
                }
                for route in routes
            ]
            return {
                "schema_version": "mmm/ecosystem-planning-deferred-v1",
                "status": "deferred",
                "brief_sha256": str(research_brief.get("brief_sha256", "")),
                "route_sha256": central_module._sha256(
                    central_module.canonical_json(receipts)
                ),
                "route_count": len(routes),
                "processed_route_count": 0,
                "remaining_route_count": len(routes),
                "routes_complete": not routes,
                "candidate_count": 0,
                "pages": [],
                "errors": [],
                "route_receipts": receipts,
                "coverage": "full route graph retained; public provider I/O deferred",
                "authorization": "none",
                "download_performed": False,
                "planning_critical_path": False,
            }
        builder = page_builder or research_coordinator_module.discover_seed_bundle
        return research_coordinator_module.collect_ecosystem_seed_bundle(
            prompt,
            game_design,
            research_brief=research_brief,
            client=client,
            route_limit=route_limit,
            page_builder=builder,
            planning_seed_only=False,
        )

    ecosystem_seed._mmm_nonblocking_planner_research = True  # type: ignore[attr-defined]
    complete_planner_module.collect_ecosystem_seed_bundle = ecosystem_seed


def install() -> None:
    """Bind research-derived small-model amplification to the fully composed runtime."""
    from . import (
        agentic_optimization_contract,
        central_research,
        complete_planner,
        ecosystem_discovery,
        parallel_runtime_contract,
        research_coordinator,
        scheduler_parallel_safety_contract,
        work_graph,
    )
    from .max_efficiency_runtime_contract import enhance_runtime
    from .small_model_agent_policy import enhance_planner

    _install_lossless_research_input(central_research, ecosystem_discovery)
    _install_nonblocking_planner_research(
        complete_planner,
        research_coordinator,
        central_research,
    )
    _install_evidence_aware_scoring(agentic_optimization_contract)
    _install_evidence_contract(complete_planner)
    _install_semantic_keys(parallel_runtime_contract)
    _install_trace_adaptive_search(agentic_optimization_contract)
    enhance_planner(complete_planner)

    enhance_runtime(
        work_graph_module=work_graph,
        scheduler_module=scheduler_parallel_safety_contract,
    )


__all__ = ["install"]
