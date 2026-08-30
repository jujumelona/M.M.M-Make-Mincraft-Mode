from __future__ import annotations

"""Preserve the approved authored requirement graph across pre-design boundaries.

The semantic authority already owns request decomposition and the evidence-obligation
contract already owns atomic research DAG construction.  This contract fixes the two
remaining authority leaks:

1. pre_design_research_pipeline held a stale import-by-value of normalize_research_brief,
   so the evidence-obligation wrapper never ran there;
2. minecraft_knowledge_contract rebuilt a weaker router=None request catalog after the
   approved graph had already been frozen.

No new semantic model, retry, repair, or fallback planner is introduced here.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_INSTALLED = False
_MARKER = "_mmm_approved_scope_downstream_authority_v1"


def _active_catalog(prompt: str) -> dict[str, Any] | None:
    from . import evidence_request_guard as guard

    active = guard._ACTIVE_REQUEST_CATALOG.get()
    if active is None or active[0] != prompt:
        return None
    catalog = active[1]
    return deepcopy(catalog) if isinstance(catalog, Mapping) else None


def _approved_research_normalize(
    obligation_module: Any,
    previous_normalize: Any,
    prompt: str,
    game_design: dict[str, Any],
    candidate: Any | None = None,
) -> dict[str, Any]:
    """Preserve the pre-design phase; expand the approved graph only after design."""

    # The owning pre-design pipeline supplies this exact internal seed and a single
    # request candidate. Replacing it with one GitHub obligation per authored
    # requirement launches donor search before there is a frozen design to search for.
    if (
        candidate is not None
        and isinstance(game_design, Mapping)
        and set(game_design) == {"title"}
        and game_design.get("title") == "pre-design research"
    ):
        return previous_normalize(prompt, game_design, candidate)

    catalog = obligation_module._catalog_for(prompt)
    if catalog is None:
        catalog = _active_catalog(prompt)
    if catalog is not None:
        return obligation_module.build_evidence_obligation_brief(
            prompt,
            catalog,
            game_design,
        )
    return previous_normalize(prompt, game_design, candidate)


def _compile_knowledge_plan_with_active_catalog(
    knowledge_module: Any,
    previous_compile: Any,
    prompt: str,
    game_design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile technical hints around the frozen catalog instead of reconstructing scope."""

    catalog = _active_catalog(prompt)
    if catalog is None:
        return previous_compile(prompt, game_design)

    design = dict(game_design or {})
    plan = dict(knowledge_module._base_compile_minecraft_knowledge_plan(prompt, design))
    plan["authored_request_catalog"] = deepcopy(catalog)
    plan["authored_requirements"] = knowledge_module._authored_requirement_lifecycle(catalog)

    requirements = catalog.get("requirements")
    routes: list[dict[str, Any]] = []
    if isinstance(requirements, list):
        from .canonical_capability_ontology import search_queries_for_capability

        for raw in requirements:
            if not isinstance(raw, Mapping):
                continue
            capability = str(raw.get("capability") or "").strip()
            source = raw.get("source_span")
            source_text = (
                str(source.get("text") or "").strip()
                if isinstance(source, Mapping)
                else ""
            )
            semantic = str(
                raw.get("semantic_statement") or raw.get("statement") or source_text
            ).strip()
            routes.append(
                {
                    "requirement_id": str(raw.get("requirement_id") or ""),
                    "capability": capability,
                    "source_text": source_text,
                    "semantic_statement": semantic,
                    "research_queries": list(search_queries_for_capability(capability))
                    if capability
                    else [],
                }
            )
    plan["authored_capability_routes"] = routes

    policy = dict(plan.get("policy", {}))
    policy.update(
        {
            "request_completeness_owner": "evidence_request_catalog",
            "feature_detection_role": "routing_hint_only",
            "authored_requirements_may_be_dropped": False,
            "unknown_authored_requirements": "preserve_for_research",
            "authored_requirement_routing_owner": "approved_requirement_graph",
            "catalog_rebuild_after_freeze": False,
        }
    )
    plan["policy"] = policy
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = knowledge_module._nodes._sha({**plan, "plan_sha256": ""})
    knowledge_module.validate_plan(plan)
    return plan


def install() -> None:
    """Bind the already-approved graph into both pre-design consumers exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_research_game_design as agentic
    from . import central_research as central
    from . import evidence_obligation_contract as obligations
    from . import minecraft_knowledge_contract as knowledge
    from . import pre_design_research_pipeline as pipeline

    current_normalize = central.normalize_research_brief
    if not getattr(current_normalize, _MARKER, False):

        def normalize(
            prompt: str,
            game_design: dict[str, Any],
            candidate: Any | None = None,
        ) -> dict[str, Any]:
            return _approved_research_normalize(
                obligations,
                current_normalize,
                prompt,
                game_design,
                candidate,
            )

        setattr(normalize, _MARKER, True)
        normalize.__wrapped__ = current_normalize  # type: ignore[attr-defined]
        central.normalize_research_brief = normalize
        agentic.normalize_research_brief = normalize
        # This module imported normalize_research_brief by value before late contracts ran.
        pipeline.normalize_research_brief = normalize
    else:
        # Even if another owner already installed the wrapper, repair the stale import edge.
        pipeline.normalize_research_brief = current_normalize

    current_compile = knowledge.compile_minecraft_knowledge_plan
    if not getattr(current_compile, _MARKER, False):

        def compile_plan(
            prompt: str,
            game_design: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            return _compile_knowledge_plan_with_active_catalog(
                knowledge,
                current_compile,
                prompt,
                game_design,
            )

        setattr(compile_plan, _MARKER, True)
        compile_plan.__wrapped__ = current_compile  # type: ignore[attr-defined]
        knowledge.compile_minecraft_knowledge_plan = compile_plan
        # Same import-by-value issue exists in pre_design_research_pipeline.
        pipeline.compile_minecraft_knowledge_plan = compile_plan
    else:
        pipeline.compile_minecraft_knowledge_plan = current_compile

    _INSTALLED = True


__all__ = [
    "_active_catalog",
    "_approved_research_normalize",
    "_compile_knowledge_plan_with_active_catalog",
    "install",
]
