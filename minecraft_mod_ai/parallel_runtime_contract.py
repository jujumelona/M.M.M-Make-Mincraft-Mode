from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar


_T = TypeVar("_T")
_R = TypeVar("_R")

_PREFETCH_LOCK = threading.RLock()
_PREFETCH_FUTURES: dict[tuple[str, str], Future[str]] = {}
_PREFETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mmm_model_prefetch",
)
_PLANNER_AUX_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mmm_planner_aux",
)
_PLANNER_STATE = threading.local()
_MODEL_RESOLVER: Callable[[Any], str] | None = None


def _env_workers(name: str, default: int, *, maximum: int = 32) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(1, min(maximum, default))
    try:
        value = int(raw)
    except ValueError:
        return max(1, min(maximum, default))
    return max(1, min(maximum, value))


def _ordered_parallel_map(
    function: Callable[[_T], _R],
    values: Iterable[_T],
    *,
    workers: int,
) -> list[_R]:
    """Run independent work concurrently while preserving deterministic order."""

    items = list(values)
    if len(items) <= 1 or workers <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(workers, len(items)),
        thread_name_prefix="mmm_parallel_io",
    ) as pool:
        futures = [pool.submit(function, item) for item in items]
        return [future.result() for future in futures]


def _colab_setup_active() -> bool:
    return bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())


def _model_key(config: Any) -> tuple[str, str]:
    model_id = str(getattr(config, "model_id", "")).strip()
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", "")).strip()
        if isinstance(extra, dict)
        else ""
    )
    return model_id, filename


def _eligible_for_model_prefetch(config: Any) -> bool:
    if not _colab_setup_active():
        return False
    if str(getattr(config, "provider", "local")) != "local":
        return False
    if str(getattr(config, "adapter", "")) not in {"llama_cpp", "vllm"}:
        return False
    model_id, _filename = _model_key(config)
    return bool(model_id)


def _prefetch_model_worker(config: Any) -> str:
    resolver = _MODEL_RESOLVER
    if resolver is None:
        raise RuntimeError("GGUF prefetch resolver is not installed.")
    return resolver(config)


def _ensure_model_prefetch(config: Any) -> Future[str] | None:
    """Start exactly one asynchronous GGUF resolution for a local Colab model."""

    if not _eligible_for_model_prefetch(config):
        return None
    key = _model_key(config)
    with _PREFETCH_LOCK:
        future = _PREFETCH_FUTURES.get(key)
        if future is not None:
            return future
        future = _PREFETCH_EXECUTOR.submit(_prefetch_model_worker, config)
        _PREFETCH_FUTURES[key] = future
    label = key[1] or Path(key[0]).name or key[0]
    print("GGUF prefetch: started", label, flush=True)
    return future


def _resolve_prefetched_model(config: Any) -> str:
    """Reuse an already-running prefetch, otherwise resolve synchronously."""

    resolver = _MODEL_RESOLVER
    if resolver is None:
        raise RuntimeError("GGUF model resolver is not installed.")
    future = _ensure_model_prefetch(config)
    if future is None:
        return resolver(config)
    return future.result()


_resolve_prefetched_model._mmm_parallel_prefetch_resolver = True  # type: ignore[attr-defined]


def _install_model_prefetch(
    *,
    model_registry_module: Any,
    llama_server_autotune_module: Any,
) -> None:
    """Bind asynchronous GGUF resolution to the single native server resolver."""

    global _MODEL_RESOLVER

    current_resolver = llama_server_autotune_module._resolve_model_path
    if not getattr(current_resolver, "_mmm_parallel_prefetch_resolver", False):
        _MODEL_RESOLVER = current_resolver
        llama_server_autotune_module._resolve_model_path = _resolve_prefetched_model
    elif _MODEL_RESOLVER is None:
        return

    registry_cls = model_registry_module.ModelRegistry
    current_load_profile = registry_cls.load_profile
    if getattr(current_load_profile, "_mmm_parallel_model_prefetch", False):
        return

    @wraps(current_load_profile)
    def load_profile_with_prefetch(self: Any, name: str):
        profile = current_load_profile(self, name)
        config = profile.roles.get("planner")
        if config is not None:
            _ensure_model_prefetch(config)
        return profile

    load_profile_with_prefetch._mmm_parallel_model_prefetch = True  # type: ignore[attr-defined]
    registry_cls.load_profile = load_profile_with_prefetch


def _parallel_discover_seed_bundle_factory(
    ecosystem_module: Any,
    original_discover: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(original_discover)
    def discover_seed_bundle_parallel(
        prompt: str,
        game_design: dict[str, Any],
        *,
        research_brief: dict[str, Any] | None = None,
        client: Any = None,
        route_cursor: str = "",
        route_limit: int = 12,
    ) -> dict[str, Any]:
        mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
        if mode not in {"auto", "on", "off"}:
            raise ecosystem_module.SpecValidationError(
                "MMM_ECOSYSTEM_DISCOVERY must be auto, on or off."
            )
        if type(route_limit) is not int or not 1 <= route_limit <= 100:
            raise ecosystem_module.SpecValidationError(
                "route_limit must be between 1 and 100."
            )

        query = ecosystem_module._seed_query(prompt, game_design)
        if research_brief is None:
            routes = [
                {
                    "domain_id": "request",
                    "provider": provider,
                    "query": query,
                    "target_profile": (
                        "media"
                        if provider in {"openverse_images", "openverse_audio"}
                        else "minecraft_mod"
                    ),
                }
                for provider in (
                    "modrinth",
                    "openverse_images",
                    "openverse_audio",
                )
            ]
            if (client is not None and client.github_token) or os.environ.get(
                "GITHUB_TOKEN"
            ):
                routes.append(
                    {
                        "domain_id": "request",
                        "provider": "github",
                        "query": query,
                        "target_profile": "minecraft_mod",
                    }
                )
        else:
            routes = list(ecosystem_module.external_discovery_routes(research_brief))

        route_receipt = ecosystem_module._sha256_text(
            ecosystem_module.canonical_json(routes)
        )
        route_offset = ecosystem_module._decode_seed_route_cursor(
            route_cursor,
            route_sha256=route_receipt,
            route_limit=route_limit,
        )
        if route_offset > len(routes):
            raise ecosystem_module.SpecValidationError(
                "Seed route cursor is beyond the route catalog."
            )
        selected_routes = routes[route_offset : route_offset + route_limit]
        next_route_offset = route_offset + len(selected_routes)
        next_route_cursor = (
            ecosystem_module._encode_seed_route_cursor(
                next_route_offset,
                route_sha256=route_receipt,
                route_limit=route_limit,
            )
            if next_route_offset < len(routes)
            else ""
        )

        if mode == "off":
            return {
                "schema_version": "mmm/ecosystem-seed-bundle-v1",
                "status": "disabled",
                "query_sha256": ecosystem_module._sha256_text(query),
                "route_sha256": route_receipt,
                "route_count": len(routes),
                "route_offset": route_offset,
                "processed_route_count": 0,
                "remaining_route_count": len(routes) - route_offset,
                "next_route_cursor": next_route_cursor,
                "routes_complete": not next_route_cursor,
                "candidate_count": 0,
                "pages": [],
                "errors": [],
                "coverage": "specialist discovery still required per production batch",
                "authorization": "none",
                "download_performed": False,
            }

        discovery = client or ecosystem_module.EcosystemDiscoveryClient()

        def search_route(route: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            provider = route["provider"]
            provider_query = route["query"]
            if provider == "openverse_audio":
                provider_query += " ambience sound effects music"
            elif provider == "openverse_images":
                provider_query += " visual reference texture architecture objects"
            try:
                page = discovery.search(
                    provider,
                    provider_query,
                    limit=10,
                    target_profile=str(
                        route.get("target_profile", "minecraft_mod")
                    ),
                )
                return (
                    "page",
                    {
                        **page,
                        "research_domain_id": route["domain_id"],
                        "route_query_sha256": ecosystem_module._sha256_text(
                            route["query"]
                        ),
                    },
                )
            except ecosystem_module.EcosystemDiscoveryUnavailable as exc:
                return (
                    "error",
                    {
                        "domain_id": route["domain_id"],
                        "provider": provider,
                        "query_sha256": ecosystem_module._sha256_text(
                            route["query"]
                        ),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )

        workers = (
            1
            if client is not None
            else _env_workers("MMM_DISCOVERY_WORKERS", 8, maximum=32)
        )
        outcomes = _ordered_parallel_map(
            search_route,
            selected_routes,
            workers=workers,
        )
        pages = [payload for kind, payload in outcomes if kind == "page"]
        errors = [payload for kind, payload in outcomes if kind == "error"]
        candidate_count = sum(
            int(page.get("returned", 0))
            for page in pages
            if isinstance(page, dict)
        )
        status = (
            "available"
            if candidate_count
            else "empty"
            if pages
            else "unavailable"
        )
        if mode == "on" and not pages:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                "Required ecosystem discovery providers were unavailable."
            )
        return {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "status": status,
            "query_sha256": ecosystem_module._sha256_text(query),
            "route_sha256": route_receipt,
            "route_count": len(routes),
            "route_offset": route_offset,
            "processed_route_count": len(selected_routes),
            "remaining_route_count": len(routes) - next_route_offset,
            "next_route_cursor": next_route_cursor,
            "routes_complete": not next_route_cursor,
            "candidate_count": candidate_count,
            "pages": pages,
            "errors": errors,
            "coverage": (
                "seed pages only; continue each provider cursor and run exact project "
                "inspection for every dependency or third-party asset considered"
            ),
            "authorization": "none",
            "download_performed": False,
        }

    discover_seed_bundle_parallel._mmm_parallel_routes = True  # type: ignore[attr-defined]
    return discover_seed_bundle_parallel


def _parallel_retrieve_domain_evidence_factory(
    central_module: Any,
    original_retrieve_graph: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    original_default_retrieve = central_module.retrieve_official_evidence

    @wraps(original_retrieve_graph)
    def retrieve_domain_evidence_parallel(
        research_brief: dict[str, Any],
        *,
        retrieve: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        selected_retrieve = retrieve or original_default_retrieve
        if retrieve is not None and retrieve is not original_default_retrieve:
            return original_retrieve_graph(research_brief, retrieve=retrieve)

        raw_domains = research_brief.get("domains")
        if not isinstance(raw_domains, list) or not raw_domains:
            raise central_module.SpecValidationError(
                "Central research brief has no domains."
            )
        domains = [central_module._research_domain(raw) for raw in raw_domains]
        ordered_jobs: list[tuple[int, int, str]] = []
        for domain_index, domain in enumerate(domains):
            if "official_docs" not in domain.providers:
                continue
            for query_index, query in enumerate(domain.queries):
                ordered_jobs.append((domain_index, query_index, query))

        if len(ordered_jobs) <= 1:
            return original_retrieve_graph(
                research_brief,
                retrieve=selected_retrieve,
            )

        workers = _env_workers("MMM_RESEARCH_WORKERS", 8, maximum=32)
        primary_results: dict[tuple[int, int], Any] = {}
        correction_futures: dict[tuple[int, int, int], Future[Any]] = {}
        with ThreadPoolExecutor(
            max_workers=min(workers, len(ordered_jobs)),
            thread_name_prefix="mmm_official_rag",
        ) as pool:
            future_to_job = {
                pool.submit(
                    selected_retrieve,
                    query,
                    minecraft_version="1.20.1",
                    loader="fabric",
                    mappings="yarn-1.20.1+build.1",
                    limit=8,
                ): (domain_index, query_index, query)
                for domain_index, query_index, query in ordered_jobs
            }
            for future in as_completed(future_to_job):
                domain_index, query_index, _query = future_to_job[future]
                primary = future.result()
                primary_results[(domain_index, query_index)] = primary
                for correction_index, correction_query in enumerate(
                    primary.correction_queries
                ):
                    correction_futures[
                        (domain_index, query_index, correction_index)
                    ] = pool.submit(
                        selected_retrieve,
                        correction_query,
                        minecraft_version="1.20.1",
                        loader="fabric",
                        mappings="yarn-1.20.1+build.1",
                        limit=4,
                    )

            results: list[dict[str, Any]] = []
            unresolved: list[str] = []
            for domain_index, domain in enumerate(domains):
                if "official_docs" not in domain.providers:
                    results.append(
                        {
                            "domain_id": domain.domain_id,
                            "strategy": "routed_to_other_providers",
                            "queries": [],
                        }
                    )
                    continue
                query_results: list[dict[str, Any]] = []
                has_hits = False
                for query_index, query in enumerate(domain.queries):
                    primary = primary_results[(domain_index, query_index)]
                    corrections: list[dict[str, Any]] = []
                    for correction_index, _correction_query in enumerate(
                        primary.correction_queries
                    ):
                        correction = correction_futures[
                            (domain_index, query_index, correction_index)
                        ].result()
                        corrections.append(correction.to_dict())
                        has_hits = has_hits or bool(correction.hits)
                    has_hits = has_hits or bool(primary.hits)
                    query_results.append(
                        {
                            "query_sha256": central_module._sha256(query),
                            "strategy": (
                                "single"
                                if not primary.correction_required
                                else "corrective_multi_hop"
                            ),
                            "primary": primary.to_dict(),
                            "corrections": corrections,
                        }
                    )
                if not has_hits:
                    unresolved.append(domain.domain_id)
                results.append(
                    {
                        "domain_id": domain.domain_id,
                        "strategy": "adaptive_per_query",
                        "queries": query_results,
                    }
                )

        payload = {
            "schema_version": "mmm/central-evidence-graph-v1",
            "brief_sha256": research_brief.get("brief_sha256", ""),
            "domains": results,
            "unresolved_official_domains": unresolved,
            "authorization": "none",
            "retrieval_is_authority": False,
        }
        payload["evidence_sha256"] = central_module._sha256(
            central_module.canonical_json(payload)
        )
        return payload

    retrieve_domain_evidence_parallel._mmm_parallel_rag = True  # type: ignore[attr-defined]
    return retrieve_domain_evidence_parallel


def _planner_key(prompt: str, research_brief: Any) -> tuple[str, int]:
    return prompt, id(research_brief)


def _ecosystem_key(
    prompt: str,
    game_design: Any,
    research_brief: Any,
) -> tuple[str, int, int]:
    return prompt, id(game_design), id(research_brief)


def _install_planner_overlap(
    *,
    complete_planner_module: Any,
    parallel_retrieve: Callable[..., dict[str, Any]],
    parallel_discover: Callable[..., dict[str, Any]],
) -> None:
    complete_planner_module.retrieve_domain_evidence = parallel_retrieve
    complete_planner_module.discover_seed_bundle = parallel_discover

    current_radar = complete_planner_module.collect_technology_radar
    current_impl = complete_planner_module._retrieve_implementation_evidence
    current_ecosystem = complete_planner_module.collect_ecosystem_seed_bundle
    if getattr(current_radar, "_mmm_parallel_planner_overlap", False):
        return

    @wraps(current_radar)
    def radar_with_prefetch(
        prompt: str,
        research_brief: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(research_brief, dict):
            key = _planner_key(prompt, research_brief)
            existing = getattr(_PLANNER_STATE, "evidence", None)
            if not existing or existing[0] != key:
                _PLANNER_STATE.evidence = (
                    key,
                    _PLANNER_AUX_EXECUTOR.submit(parallel_retrieve, research_brief),
                )
        return current_radar(prompt, research_brief, *args, **kwargs)

    radar_with_prefetch._mmm_parallel_planner_overlap = True  # type: ignore[attr-defined]

    @wraps(current_impl)
    def implementation_evidence_with_overlap(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or complete_planner_module.normalize_research_brief(
            prompt,
            game_design,
        )
        eco_key = _ecosystem_key(prompt, game_design, brief)
        existing_ecosystem = getattr(_PLANNER_STATE, "ecosystem", None)
        if not existing_ecosystem or existing_ecosystem[0] != eco_key:
            original_collect = getattr(
                current_ecosystem,
                "__wrapped__",
                current_ecosystem,
            )
            _PLANNER_STATE.ecosystem = (
                eco_key,
                _PLANNER_AUX_EXECUTOR.submit(
                    original_collect,
                    prompt,
                    game_design,
                    research_brief=brief,
                    page_builder=parallel_discover,
                    allow_legacy_terminal=True,
                ),
            )

        evidence_key = _planner_key(prompt, brief)
        existing_evidence = getattr(_PLANNER_STATE, "evidence", None)
        if existing_evidence and existing_evidence[0] == evidence_key:
            return existing_evidence[1].result()
        return parallel_retrieve(brief)

    @wraps(current_ecosystem)
    def ecosystem_with_prefetch(
        prompt: str,
        game_design: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        brief = kwargs.get("research_brief")
        if isinstance(brief, dict):
            key = _ecosystem_key(prompt, game_design, brief)
            existing = getattr(_PLANNER_STATE, "ecosystem", None)
            if existing and existing[0] == key:
                try:
                    return existing[1].result()
                finally:
                    _PLANNER_STATE.ecosystem = None
        return current_ecosystem(prompt, game_design, *args, **kwargs)

    complete_planner_module.collect_technology_radar = radar_with_prefetch
    complete_planner_module._retrieve_implementation_evidence = (
        implementation_evidence_with_overlap
    )
    complete_planner_module.collect_ecosystem_seed_bundle = ecosystem_with_prefetch


def install(
    *,
    complete_planner_module: Any,
    model_registry_module: Any,
    llama_server_autotune_module: Any,
) -> None:
    """Overlap only independent work while preserving deterministic result order."""

    from . import central_research as central_module
    from . import ecosystem_discovery as ecosystem_module

    _install_model_prefetch(
        model_registry_module=model_registry_module,
        llama_server_autotune_module=llama_server_autotune_module,
    )

    original_discover = ecosystem_module.discover_seed_bundle
    if getattr(original_discover, "_mmm_parallel_routes", False):
        parallel_discover = original_discover
    else:
        parallel_discover = _parallel_discover_seed_bundle_factory(
            ecosystem_module,
            original_discover,
        )
        ecosystem_module.discover_seed_bundle = parallel_discover

    original_retrieve = central_module.retrieve_domain_evidence
    if getattr(original_retrieve, "_mmm_parallel_rag", False):
        parallel_retrieve = original_retrieve
    else:
        parallel_retrieve = _parallel_retrieve_domain_evidence_factory(
            central_module,
            original_retrieve,
        )
        central_module.retrieve_domain_evidence = parallel_retrieve

    _install_planner_overlap(
        complete_planner_module=complete_planner_module,
        parallel_retrieve=parallel_retrieve,
        parallel_discover=parallel_discover,
    )


__all__ = ["install"]
