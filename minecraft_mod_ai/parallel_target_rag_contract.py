from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable


def _target_tuple(research_brief: dict[str, Any]) -> tuple[str, str, str] | None:
    target = research_brief.get("_mmm_platform_target")
    if not isinstance(target, dict):
        return None
    minecraft_version = str(target.get("minecraft_version", "")).strip()
    loader = str(target.get("loader", "fabric")).strip().lower()
    mappings = str(target.get("mappings", "")).strip()
    if not minecraft_version or not loader or not mappings:
        return None
    return minecraft_version, loader, mappings


def _target_parallel_retrieve_factory(
    *,
    central_module: Any,
    parallel_module: Any,
    legacy_parallel_retrieve: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(legacy_parallel_retrieve)
    def retrieve_domain_evidence(
        research_brief: dict[str, Any],
        *,
        retrieve: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        target = _target_tuple(research_brief)
        if target is None:
            if retrieve is None:
                return legacy_parallel_retrieve(research_brief)
            return legacy_parallel_retrieve(research_brief, retrieve=retrieve)

        minecraft_version, loader, mappings = target
        selected_retrieve = retrieve or central_module.retrieve_official_evidence
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

        primary_results: dict[tuple[int, int], Any] = {}
        correction_results: dict[tuple[int, int, int], Any] = {}
        if ordered_jobs:
            workers = parallel_module._env_workers(
                "MMM_RESEARCH_WORKERS",
                8,
                maximum=32,
            )
            with ThreadPoolExecutor(
                max_workers=min(workers, len(ordered_jobs)),
                thread_name_prefix="mmm_target_rag",
            ) as pool:
                future_to_job: dict[Future[Any], tuple[int, int, str]] = {
                    pool.submit(
                        selected_retrieve,
                        query,
                        minecraft_version=minecraft_version,
                        loader=loader,
                        mappings=mappings,
                        limit=8,
                    ): (domain_index, query_index, query)
                    for domain_index, query_index, query in ordered_jobs
                }
                correction_futures: dict[
                    tuple[int, int, int], Future[Any]
                ] = {}
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
                            minecraft_version=minecraft_version,
                            loader=loader,
                            mappings=mappings,
                            limit=4,
                        )
                for key, future in correction_futures.items():
                    correction_results[key] = future.result()

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
                    correction = correction_results[
                        (domain_index, query_index, correction_index)
                    ]
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
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
            },
            "domains": results,
            "unresolved_official_domains": unresolved,
            "authorization": "none",
            "retrieval_is_authority": False,
        }
        payload["evidence_sha256"] = central_module._sha256(
            central_module.canonical_json(payload)
        )
        return payload

    retrieve_domain_evidence._mmm_parallel_target_rag = True
    return retrieve_domain_evidence


def install(
    *,
    complete_planner_module: Any,
    central_module: Any,
    parallel_module: Any,
) -> None:
    current_parallel = central_module.retrieve_domain_evidence
    if getattr(current_parallel, "_mmm_parallel_target_rag", False):
        return

    target_parallel = _target_parallel_retrieve_factory(
        central_module=central_module,
        parallel_module=parallel_module,
        legacy_parallel_retrieve=current_parallel,
    )
    central_module.retrieve_domain_evidence = target_parallel
    complete_planner_module.retrieve_domain_evidence = target_parallel

    current_radar = complete_planner_module.collect_technology_radar
    current_impl = complete_planner_module._retrieve_implementation_evidence

    @wraps(current_radar)
    def radar_with_target_prefetch(
        prompt: str,
        research_brief: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(research_brief, dict) and _target_tuple(research_brief):
            key = parallel_module._planner_key(prompt, research_brief)
            existing = getattr(parallel_module._PLANNER_STATE, "evidence", None)
            if not existing or existing[0] != key:
                parallel_module._PLANNER_STATE.evidence = (
                    key,
                    parallel_module._PLANNER_AUX_EXECUTOR.submit(
                        target_parallel,
                        research_brief,
                    ),
                )
        # The existing overlap wrapper sees the matching future and therefore does
        # not enqueue its historical 1.20.1-only prefetch.
        return current_radar(prompt, research_brief, *args, **kwargs)

    radar_with_target_prefetch._mmm_parallel_target_rag = True
    complete_planner_module.collect_technology_radar = radar_with_target_prefetch

    @wraps(current_impl)
    def implementation_with_target_prefetch(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or complete_planner_module.normalize_research_brief(
            prompt,
            game_design,
        )
        if _target_tuple(brief):
            key = parallel_module._planner_key(prompt, brief)
            existing = getattr(parallel_module._PLANNER_STATE, "evidence", None)
            if not existing or existing[0] != key:
                parallel_module._PLANNER_STATE.evidence = (
                    key,
                    parallel_module._PLANNER_AUX_EXECUTOR.submit(
                        target_parallel,
                        brief,
                    ),
                )
        # Preserve ecosystem overlap and deterministic result ordering implemented
        # by the existing wrapper; only its evidence future is replaced.
        return current_impl(
            prompt,
            game_design,
            research_brief=brief,
        )

    implementation_with_target_prefetch._mmm_parallel_target_rag = True
    complete_planner_module._retrieve_implementation_evidence = (
        implementation_with_target_prefetch
    )
