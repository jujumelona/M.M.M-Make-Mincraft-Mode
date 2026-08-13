from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from .agentic_research_fusion import retrieve_target_agentic_evidence
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
            # Keep explicit injected retrievers on the legacy path for deterministic
            # contract tests and callers that intentionally control call ordering.
            if retrieve is not None:
                return legacy_retrieve(research_brief, retrieve=retrieve)
            # Normal pre-design research is still a real runtime lane. Run the reviewed
            # conceptual 1.20.1 corpus through the same parallel agentic fusion layer
            # instead of silently falling back to serial retrieval.
            return retrieve_target_agentic_evidence(
                research_brief,
                central_module=central_module,
                retrieve=retrieval_module.retrieve_official_evidence,
                minecraft_version="1.20.1",
                loader="fabric",
                mappings="yarn-1.20.1+build.1",
                include_target=False,
            )

        selected_retrieve = retrieve or retrieval_module.retrieve_official_evidence
        return retrieve_target_agentic_evidence(
            research_brief,
            central_module=central_module,
            retrieve=selected_retrieve,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            mappings=adapter.yarn_mappings,
            include_target=True,
        )

    retrieve_domain_evidence_target_parallel._mmm_parallel_target_rag = True
    retrieve_domain_evidence_target_parallel._mmm_agentic_rag_fusion = True
    return retrieve_domain_evidence_target_parallel


def install(*, complete_planner_module: Any, central_module: Any, retrieval_module: Any) -> None:
    """Bind target RAG to parallel adaptive retrieval and overlap independent research."""

    from . import ecosystem_discovery as ecosystem_module
    from . import parallel_runtime_contract as parallel_module

    current_central = central_module.retrieve_domain_evidence
    if getattr(current_central, "_mmm_agentic_rag_fusion", False):
        target_retrieve = current_central
    else:
        target_retrieve = _target_parallel_retrieve_factory(
            central_module=central_module,
            retrieval_module=retrieval_module,
            legacy_retrieve=current_central,
        )
        central_module.retrieve_domain_evidence = target_retrieve

    complete_planner_module.retrieve_domain_evidence = target_retrieve

    # platform_prompt_contract imports the research-first game-design helper before
    # this late target-RAG installer runs. Repair that captured function reference so
    # pre-design research cannot bypass the parallel/fusion owner.
    try:
        from . import agentic_research_game_design as research_design_module
    except ImportError:
        research_design_module = None
    if research_design_module is not None:
        research_design_module.retrieve_domain_evidence = target_retrieve

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
    radar_with_target_prefetch._mmm_agentic_rag_fusion = True
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
    implementation_evidence_with_target_overlap._mmm_agentic_rag_fusion = True
    complete_planner_module._retrieve_implementation_evidence = (
        implementation_evidence_with_target_overlap
    )


__all__ = ["install"]
