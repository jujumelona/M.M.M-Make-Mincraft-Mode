from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable

from .platform_catalog import adapter_for_target


def _adapter_from_brief(research_brief: dict[str, Any]) -> Any | None:
    target = research_brief.get("_mmm_platform_target")
    if not isinstance(target, dict):
        return None
    minecraft_version = str(target.get("minecraft_version", "")).strip()
    loader = str(target.get("loader", "fabric")).strip() or "fabric"
    if not minecraft_version:
        return None
    return adapter_for_target(minecraft_version, loader)


def _target_parallel_retrieve_factory(
    *,
    central_module: Any,
    retrieval_module: Any,
    legacy_retrieve: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def retrieve_domain_evidence_target_parallel(
        research_brief: dict[str, Any],
        *,
        retrieve: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        adapter = _adapter_from_brief(research_brief)
        if adapter is None:
            # Pre-target classification may still use the legacy conceptual lane.
            # Once a host target exists, this branch is never allowed to relabel it.
            return legacy_retrieve(research_brief, retrieve=retrieve) if retrieve else legacy_retrieve(research_brief)

        selected_retrieve = retrieve or retrieval_module.retrieve_official_evidence
        raw_domains = research_brief.get("domains")
        if not isinstance(raw_domains, list) or not raw_domains:
            raise central_module.SpecValidationError("Central research brief has no domains.")
        domains = [central_module._research_domain(raw) for raw in raw_domains]
        jobs: list[tuple[int, int, str]] = []
        for domain_index, domain in enumerate(domains):
            if "official_docs" not in domain.providers:
                continue
            for query_index, query in enumerate(domain.queries):
                jobs.append((domain_index, query_index, query))

        if not jobs:
            payload = {
                "schema_version": "mmm/central-evidence-graph-v1",
                "brief_sha256": research_brief.get("brief_sha256", ""),
                "target": {
                    "minecraft_version": adapter.minecraft_version,
                    "loader": adapter.loader,
                    "mappings": adapter.yarn_mappings,
                },
                "domains": [
                    {
                        "domain_id": domain.domain_id,
                        "strategy": "routed_to_other_providers",
                        "queries": [],
                    }
                    for domain in domains
                ],
                "unresolved_official_domains": [],
                "authorization": "none",
                "retrieval_is_authority": False,
            }
            payload["evidence_sha256"] = central_module._sha256(
                central_module.canonical_json(payload)
            )
            return payload

        from .parallel_runtime_contract import _env_workers

        workers = _env_workers("MMM_RESEARCH_WORKERS", 8, maximum=32)
        primary_results: dict[tuple[int, int], Any] = {}
        correction_results: dict[tuple[int, int], list[Any]] = {}

        def fetch_primary(job: tuple[int, int, str]) -> tuple[int, int, Any]:
            domain_index, query_index, query = job
            receipt = selected_retrieve(
                query,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
                limit=8,
            )
            return domain_index, query_index, receipt

        with ThreadPoolExecutor(
            max_workers=min(workers, len(jobs)),
            thread_name_prefix="mmm_target_rag",
        ) as pool:
            primary_futures = [pool.submit(fetch_primary, job) for job in jobs]
            for future in as_completed(primary_futures):
                domain_index, query_index, primary = future.result()
                primary_results[(domain_index, query_index)] = primary

            correction_futures: dict[Future[Any], tuple[int, int, int]] = {}
            for (domain_index, query_index), primary in primary_results.items():
                correction_results[(domain_index, query_index)] = [
                    None for _ in primary.correction_queries
                ]
                for correction_index, correction_query in enumerate(primary.correction_queries):
                    future = pool.submit(
                        selected_retrieve,
                        correction_query,
                        minecraft_version=adapter.minecraft_version,
                        loader=adapter.loader,
                        mappings=adapter.yarn_mappings,
                        limit=4,
                    )
                    correction_futures[future] = (
                        domain_index,
                        query_index,
                        correction_index,
                    )
            for future in as_completed(correction_futures):
                domain_index, query_index, correction_index = correction_futures[future]
                correction_results[(domain_index, query_index)][correction_index] = future.result()

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
                corrections = [
                    item.to_dict()
                    for item in correction_results[(domain_index, query_index)]
                    if item is not None
                ]
                has_hits = has_hits or bool(primary.hits) or any(
                    bool(item.get("hits")) for item in corrections
                )
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
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
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

    retrieve_domain_evidence_target_parallel._mmm_parallel_target_rag = True
    return retrieve_domain_evidence_target_parallel


def install(*, complete_planner_module: Any, central_module: Any, retrieval_module: Any) -> None:
    """Repair the late parallel overlay without sacrificing research concurrency."""

    from . import parallel_runtime_contract as parallel_module
    from . import ecosystem_discovery as ecosystem_module

    current_central = central_module.retrieve_domain_evidence
    if getattr(current_central, "_mmm_parallel_target_rag", False):
        target_retrieve = current_central
    else:
        target_retrieve = _target_parallel_retrieve_factory(
            central_module=central_module,
            retrieval_module=retrieval_module,
            legacy_retrieve=current_central,
        )
        central_module.retrieve_domain_evidence = target_retrieve

    complete_planner_module.retrieve_domain_evidence = target_retrieve

    current_radar = complete_planner_module.collect_technology_radar
    base_radar = getattr(current_radar, "__wrapped__", current_radar)

    @wraps(base_radar)
    def radar_with_target_prefetch(
        prompt: str,
        research_brief: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(research_brief, dict):
            key = parallel_module._planner_key(prompt, research_brief)
            existing = getattr(parallel_module._PLANNER_STATE, "evidence", None)
            if not existing or existing[0] != key:
                parallel_module._PLANNER_STATE.evidence = (
                    key,
                    parallel_module._PLANNER_AUX_EXECUTOR.submit(
                        target_retrieve,
                        research_brief,
                    ),
                )
        return base_radar(prompt, research_brief, *args, **kwargs)

    radar_with_target_prefetch._mmm_parallel_target_rag = True
    complete_planner_module.collect_technology_radar = radar_with_target_prefetch

    current_ecosystem = complete_planner_module.collect_ecosystem_seed_bundle
    base_ecosystem = getattr(current_ecosystem, "__wrapped__", current_ecosystem)

    def implementation_evidence_with_target_overlap(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or complete_planner_module.normalize_research_brief(
            prompt,
            game_design,
        )
        if _adapter_from_brief(brief) is None:
            selection = game_design.get("_platform_selection")
            if isinstance(selection, dict) and isinstance(selection.get("target"), dict):
                brief = {**brief, "_mmm_platform_target": dict(selection["target"])}

        ecosystem_key = parallel_module._ecosystem_key(prompt, game_design, brief)
        existing_ecosystem = getattr(parallel_module._PLANNER_STATE, "ecosystem", None)
        if not existing_ecosystem or existing_ecosystem[0] != ecosystem_key:
            parallel_module._PLANNER_STATE.ecosystem = (
                ecosystem_key,
                parallel_module._PLANNER_AUX_EXECUTOR.submit(
                    base_ecosystem,
                    prompt,
                    game_design,
                    research_brief=brief,
                    page_builder=ecosystem_module.discover_seed_bundle,
                    allow_legacy_terminal=True,
                ),
            )

        evidence_key = parallel_module._planner_key(prompt, brief)
        existing_evidence = getattr(parallel_module._PLANNER_STATE, "evidence", None)
        if existing_evidence and existing_evidence[0] == evidence_key:
            return existing_evidence[1].result()
        return target_retrieve(brief)

    implementation_evidence_with_target_overlap._mmm_parallel_target_rag = True
    complete_planner_module._retrieve_implementation_evidence = implementation_evidence_with_target_overlap


__all__ = ["install"]
