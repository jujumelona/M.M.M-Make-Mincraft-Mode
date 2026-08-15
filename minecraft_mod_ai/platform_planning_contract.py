from __future__ import annotations

"""Platform-neutral planning consumers after host target selection.

Target selection has exactly one owner:
``platform_central_ai_contract`` -> ``platform_resolver`` -> ``platform_optimizer``.
This module only keeps prompts target-neutral and retrieves implementation evidence
for the already selected executable target.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any

from .platform_catalog import adapter_for_target


def install(
    *,
    game_design_module: Any,
    complete_planner_module: Any,
    central_research_module: Any,
    retrieval_module: Any,
) -> None:
    _install_target_neutral_prompts(game_design_module)
    _install_selected_target_evidence(
        complete_planner_module,
        central_research_module,
        retrieval_module,
    )


def _install_target_neutral_prompts(module: Any) -> None:
    original_system = module._system_prompt
    if not getattr(original_system, "_mmm_target_neutral_prompt", False):

        @wraps(original_system)
        def system_prompt() -> str:
            text = original_system().replace(
                "GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.",
                "GameDesignPlanner for a Minecraft Java mod production system.",
            )
            return text + (
                "\n\nPlatform rule: describe capabilities only. Do not choose or assume a "
                "Minecraft version, loader, mappings, Java, build tool, or package coordinate. "
                "The host resolves and verifies the target after semantic design."
            )

        system_prompt._mmm_target_neutral_prompt = True
        module._system_prompt = system_prompt

    original_sharded = module._sharded_design_system_prompt
    if not getattr(original_sharded, "_mmm_target_neutral_prompt", False):

        @wraps(original_sharded)
        def sharded_prompt() -> str:
            text = original_sharded().replace(
                "request for a Minecraft Java 1.20.1 Fabric mod.",
                "request for a Minecraft Java mod.",
            )
            return text + (
                "\nReturn semantic requirements only; exact platform coordinates are host-owned."
            )

        sharded_prompt._mmm_target_neutral_prompt = True
        module._sharded_design_system_prompt = sharded_prompt


def _install_selected_target_evidence(module: Any, central: Any, retrieval: Any) -> None:
    def retrieve_implementation_evidence(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or central.normalize_research_brief(prompt, game_design)
        adapter = _target_from_design(module, game_design)
        return _retrieve_domain_evidence_target(
            central,
            retrieval,
            brief,
            adapter=adapter,
        )

    retrieve_implementation_evidence._mmm_selected_target_evidence = True
    module._retrieve_implementation_evidence = retrieve_implementation_evidence

    original_normalize = module.normalize_research_brief
    if not getattr(original_normalize, "_mmm_selected_target_evidence", False):

        @wraps(original_normalize)
        def normalize_brief(prompt: str, game_design: dict[str, Any], candidate=None):
            brief = original_normalize(prompt, game_design, candidate)
            selection = game_design.get("_platform_selection")
            if isinstance(selection, dict) and isinstance(selection.get("target"), dict):
                brief = {**brief, "_mmm_platform_target": dict(selection["target"])}
            return brief

        normalize_brief._mmm_selected_target_evidence = True
        module.normalize_research_brief = normalize_brief


def _target_from_design(module: Any, game_design: dict[str, Any]):
    selection = game_design.get("_platform_selection")
    if not isinstance(selection, dict):
        raise module.SpecValidationError("Planning target selection is missing.")
    target = selection.get("target")
    if not isinstance(target, dict):
        raise module.SpecValidationError("Planning target payload is missing.")
    try:
        return adapter_for_target(
            str(target.get("minecraft_version", "")),
            str(target.get("loader", "")),
        )
    except ValueError as exc:
        raise module.SpecValidationError(str(exc)) from exc


def _retrieve_domain_evidence_target(
    central: Any,
    retrieval: Any,
    research_brief: dict[str, Any],
    *,
    adapter: Any,
) -> dict[str, Any]:
    domains = research_brief.get("domains")
    if not isinstance(domains, list) or not domains:
        raise central.SpecValidationError("Central research brief has no domains.")

    official_jobs: list[tuple[str, str]] = []
    routed: list[dict[str, Any]] = []
    for raw_domain in domains:
        domain = central._research_domain(raw_domain)
        if "official_docs" not in domain.providers:
            routed.append(
                {
                    "domain_id": domain.domain_id,
                    "strategy": "routed_to_other_providers",
                    "queries": [],
                }
            )
            continue
        official_jobs.extend((domain.domain_id, query) for query in domain.queries)

    by_domain: dict[str, list[dict[str, Any]]] = {}
    workers = min(8, max(1, len(official_jobs)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-target-rag") as pool:
        futures = {
            pool.submit(_target_query, central, retrieval, query, adapter): (domain_id, query)
            for domain_id, query in official_jobs
        }
        for future in as_completed(futures):
            domain_id, query = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = {
                    "query_sha256": central._sha256(query),
                    "strategy": "failed_closed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "primary": None,
                    "corrections": [],
                }
            by_domain.setdefault(domain_id, []).append(value)

    results = list(routed)
    unresolved: list[str] = []
    for raw_domain in domains:
        domain = central._research_domain(raw_domain)
        if "official_docs" not in domain.providers:
            continue
        values = sorted(
            by_domain.get(domain.domain_id, []),
            key=lambda item: item["query_sha256"],
        )
        has_hits = any(
            isinstance(item.get("primary"), dict) and bool(item["primary"].get("hits"))
            for item in values
        )
        if not has_hits:
            unresolved.append(domain.domain_id)
        results.append(
            {
                "domain_id": domain.domain_id,
                "strategy": "parallel_target_multi_path",
                "queries": values,
            }
        )

    payload = {
        "schema_version": "mmm/central-evidence-graph-v2",
        "brief_sha256": research_brief.get("brief_sha256", ""),
        "target": {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        },
        "domains": sorted(results, key=lambda item: item["domain_id"]),
        "unresolved_official_domains": sorted(set(unresolved)),
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = central._sha256(central.canonical_json(payload))
    return payload


def _target_query(central: Any, retrieval: Any, query: str, adapter: Any) -> dict[str, Any]:
    primary = _target_retrieve(retrieval, query, adapter=adapter, limit=8)
    corrections = [
        _target_retrieve(retrieval, correction, adapter=adapter, limit=4).to_dict()
        for correction in primary.correction_queries
    ]
    return {
        "query_sha256": central._sha256(query),
        "strategy": "single"
        if not primary.correction_required
        else "evolving_corrective_multi_hop",
        "primary": primary.to_dict(),
        "corrections": corrections,
    }


def _target_retrieve(retrieval: Any, query: str, *, adapter: Any, limit: int):
    with retrieval.OfficialCorpusIndex(documents=retrieval.BUILTIN_CORPUS) as index:
        return index.retrieve(
            query,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            mappings=adapter.yarn_mappings,
            limit=limit,
        )
