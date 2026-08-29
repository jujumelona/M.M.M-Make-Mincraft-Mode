from __future__ import annotations

"""Bind research-first game design to leaf-granular production execution.

The authored request catalog remains immutable.  Game-design subsystems are derived
implementation obligations beneath those authored requirements: they may refine donor
search and coder task granularity, but they cannot become new public requirements.

This contract also restores the intended pre-design agentic research path without
replacing ``GameDesignPlanner.plan``.  The host-owned plan method continues to own
inventory binding, request freezing, pre-retrieval planning, target/reuse selection,
and proposal construction; only its bounded design-generation primitive is replaced.
"""

import hashlib
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any

from . import evidence_first_planning as _evidence

_INSTALLED = False
_GENERATOR_MARKER = "__mmm_research_first_design_generator__"
_SECTION_MARKER = "__mmm_deep_design_section_prompt__"
_STEPS_MARKER = "__mmm_design_leaf_semantic_steps__"
_TASKS_MARKER = "__mmm_design_leaf_reuse_binding__"
_COMPILE_MARKER = "__mmm_design_leaf_evidence_plan__"
_VALIDATE_MARKER = "__mmm_design_leaf_plan_validation__"

_ACTIVE_DESIGN_EXECUTION: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "mmm_active_design_execution_facets",
    default=(),
)


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(
        dict.fromkeys(
            text
            for item in value
            if (text := str(item or "").strip())
        )
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_section_messages(
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    research: Mapping[str, Any],
    prior_error: str,
    prior_candidate: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    original = _deep_section_messages.__wrapped__
    messages = original(
        prompt=prompt,
        section_id=section_id,
        fields=fields,
        research=research,
        prior_error=prior_error,
        prior_candidate=prior_candidate,
    )
    if not messages:
        return messages
    output = [dict(message) for message in messages]
    system = output[0]
    system["content"] = (
        str(system.get("content") or "")
        + "\n\nPRODUCTION DEPTH: finish the game/mod design before implementation search. "
        "Decompose every requested mechanic into the smallest meaningful subsystems "
        "that can be independently implemented, tested, and searched for reuse. Split "
        "different player verbs, resources, state transitions, purchase/assembly steps, "
        "upgrade gates, travel phases, encounters, combat outcomes, world interactions, "
        "persistence-visible state, networking/client surfaces, and integration rules when "
        "they can fail independently. The modules array is the implementation-leaf index: "
        "every implementation-bearing core-loop/progression/combat/mod-context behavior "
        "must have a concrete modules entry with a stable snake_case plugin_id and a reason "
        "that states its owned behavior. Do not collapse an epic such as planet interaction, "
        "ship construction, trading, or progression into one generic module. Use as many "
        "leaf modules as the authored design genuinely needs; never add unrelated features. "
        "Use the supplied research evidence for Minecraft/Fabric facts and unresolved "
        "assumptions, but do not claim a third-party donor was selected here: donor/reuse "
        "selection happens only after this design is frozen."
    )
    return output


def _research_first_generate_once(
    router: Any,
    *,
    authoritative_prompt: str,
    media_paths: Sequence[Any],
    system_prompt: str,
    fallback_prompt: str | None = None,
) -> dict[str, Any]:
    original = _research_first_generate_once.__wrapped__
    from . import agentic_research_game_design as agentic
    from . import game_design as game_design

    if not agentic.supports_agentic_research_router(router):
        return original(
            router,
            authoritative_prompt=authoritative_prompt,
            media_paths=media_paths,
            system_prompt=system_prompt,
            fallback_prompt=fallback_prompt,
        )

    # In a sharded request the host JSON envelope is provenance, while fallback_prompt
    # is the exact lossless user page. Research and design must consume the latter.
    design_prompt = str(fallback_prompt or authoritative_prompt)
    research = agentic.collect_pre_design_research(router, design_prompt)
    design = agentic.generate_sectioned_game_design(
        game_design,
        router,
        design_prompt,
        media_paths=media_paths,
        research=research,
    )
    canonical = game_design._canonical_game_design(design)
    return {
        **canonical,
        "_pre_design_research": research,
        "_research_brief": dict(research.get("research_brief") or {}),
    }


def _reuse_payload(
    game_design: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(explicit, Mapping):
        return dict(explicit)
    return _evidence._reuse_payload(game_design)


def _execution_context(
    game_design: Mapping[str, Any],
    reuse_plan: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    pre = _mapping(game_design.get("_pre_retrieval_plan"))
    bindings = pre.get("design_retrieval_facets")
    if not isinstance(bindings, list) or not bindings:
        return ()

    request = _mapping(game_design.get("_evidence_request_catalog"))
    requirements = request.get("requirements")
    if not isinstance(requirements, list):
        return ()
    requirement_by_id = {
        str(item.get("requirement_id") or ""): item
        for item in requirements
        if isinstance(item, Mapping) and str(item.get("requirement_id") or "")
    }

    from . import planner_graph_integrity_contract as graph_contract

    facet_details = {
        str(item.get("capability") or ""): dict(item)
        for item in graph_contract._design_facets(game_design)
        if isinstance(item, Mapping) and str(item.get("capability") or "")
    }
    reuse = _reuse_payload(game_design, reuse_plan)
    raw_reuse = {
        str(item.get("capability") or ""): dict(item)
        for item in reuse.get("capabilities", ())
        if isinstance(item, Mapping) and str(item.get("capability") or "")
    }

    candidates: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        requirement_ref = str(binding.get("requirement_ref") or "")
        requirement = requirement_by_id.get(requirement_ref)
        capability = str(binding.get("capability") or "")
        detail = facet_details.get(capability, {})
        if requirement is None or not capability or not detail:
            continue
        parent_capability = str(requirement.get("capability") or "").strip()
        if not parent_capability:
            continue
        raw = raw_reuse.get(capability, {})
        refs: list[str] = []
        mode = str(raw.get("mode") or "fresh").strip().casefold()
        if mode != "fresh":
            refs.extend(_strings(raw.get("component_refs")))
            source_id = str(raw.get("source_id") or "").strip()
            if source_id:
                refs.append(source_id)
        candidates.append(
            {
                "requirement_ref": requirement_ref,
                "parent_capability": parent_capability,
                "capability": capability,
                "detail": str(detail.get("detail") or capability).strip(),
                "source": str(detail.get("source") or binding.get("source") or "").strip(),
                "reuse_refs": list(dict.fromkeys(refs)),
                "reuse_mode": mode,
                "proof_level": str(raw.get("proof_level") or "").strip(),
            }
        )

    # modules[] is the implementation-leaf index. Core-loop/progression/combat facets
    # remain donor-search evidence, but once concrete module leaves exist for a parent we
    # do not generate duplicate coder classes for its narrative descriptions.
    module_parents = {
        item["requirement_ref"]
        for item in candidates
        if item["source"].startswith("game_design.modules[")
    }
    selected = [
        item
        for item in candidates
        if item["requirement_ref"] not in module_parents
        or item["source"].startswith("game_design.modules[")
    ]

    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for item in selected:
        key = (item["requirement_ref"], item["capability"])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return tuple(output)


def _context_groups(
    context: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in context:
        item = dict(raw)
        parent = str(item.get("parent_capability") or "")
        if parent:
            grouped.setdefault(parent, []).append(item)
    return grouped


def _leaf_outcome(item: Mapping[str, Any]) -> str:
    return (
        "Implement and independently verify design leaf "
        f"{item.get('capability')}: {item.get('detail')}"
    )


def _semantic_steps_with_design_leaves(
    capability: str,
    branches: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    original = _semantic_steps_with_design_leaves.__wrapped__
    facets = _context_groups(_ACTIVE_DESIGN_EXECUTION.get()).get(str(capability), [])
    if not facets:
        return original(capability, branches)

    leaf_provides: list[str] = []
    steps: list[Any] = []
    for index, item in enumerate(facets):
        digest = hashlib.sha256(
            f"{capability}\0{item['capability']}".encode("utf-8")
        ).hexdigest()[:16]
        provided = f"design_leaf:{digest}"
        leaf_provides.append(provided)
        folded = f"{item['capability']} {item['detail']}".casefold()
        anchor_kinds = (
            ("symbol", "resource", "test")
            if any(
                term in folded
                for term in (
                    "gui",
                    "screen",
                    "render",
                    "texture",
                    "model",
                    "recipe",
                    "loot",
                    "tag",
                    "worldgen",
                    "biome",
                    "resource",
                )
            )
            else ("symbol", "test")
        )
        steps.append(
            _evidence._Step(
                f"design_leaf_{index + 1}",
                _leaf_outcome(item),
                ("target:frozen",),
                (provided,),
                anchor_kinds,
            )
        )

    steps.append(
        _evidence._Step(
            "design_integration",
            f"Integrate and verify all planned design leaves for {capability}",
            tuple(leaf_provides),
            (capability,),
            ("test",),
        )
    )
    return tuple(steps)


def _compile_tasks_with_design_reuse(
    gaps: Sequence[Mapping[str, Any]],
    reuse: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    original = _compile_tasks_with_design_reuse.__wrapped__
    tasks = original(gaps, reuse, target, branches, ownership)
    if not tasks:
        return tasks

    refs_by_outcome = {
        _leaf_outcome(item): list(_strings(item.get("reuse_refs")))
        for item in _ACTIVE_DESIGN_EXECUTION.get()
        if _strings(item.get("reuse_refs"))
    }
    if not refs_by_outcome:
        return tasks

    output: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        refs = refs_by_outcome.get(str(task.get("semantic_outcome") or ""), [])
        if refs:
            task["reuse_refs"] = list(
                dict.fromkeys([*_strings(task.get("reuse_refs")), *refs])
            )
            task["task_sha256"] = ""
            task["task_sha256"] = _evidence._hash_without(task, "task_sha256")
        output.append(task)
    return tuple(output)


def _context_from_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = plan.get("design_execution_facets")
    if not isinstance(raw, list):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _validate_plan_with_design_context(
    plan: Mapping[str, Any],
    *,
    prompt: str | None = None,
) -> None:
    original = _validate_plan_with_design_context.__wrapped__
    context = _context_from_plan(plan)
    if not context:
        return original(plan, prompt=prompt)
    token = _ACTIVE_DESIGN_EXECUTION.set(context)
    try:
        original(plan, prompt=prompt)
    finally:
        _ACTIVE_DESIGN_EXECUTION.reset(token)


def _compile_plan_with_design_context(
    prompt: str,
    game_design: Mapping[str, Any],
    *,
    component_catalog: Any = None,
    reuse_plan: Mapping[str, Any] | None = None,
    target_decision: Mapping[str, Any] | None = None,
    semantic_router: Any | None = None,
) -> dict[str, Any]:
    original = _compile_plan_with_design_context.__wrapped__
    context = _execution_context(game_design, reuse_plan)
    if not context:
        return original(
            prompt,
            game_design,
            component_catalog=component_catalog,
            reuse_plan=reuse_plan,
            target_decision=target_decision,
            semantic_router=semantic_router,
        )

    token = _ACTIVE_DESIGN_EXECUTION.set(context)
    try:
        plan = original(
            prompt,
            game_design,
            component_catalog=component_catalog,
            reuse_plan=reuse_plan,
            target_decision=target_decision,
            semantic_router=semantic_router,
        )
    finally:
        _ACTIVE_DESIGN_EXECUTION.reset(token)

    enriched = dict(plan)
    enriched["design_execution_facets"] = [dict(item) for item in context]
    enriched["plan_sha256"] = ""
    enriched["plan_sha256"] = _evidence._hash_without(enriched, "plan_sha256")
    _evidence.validate_evidence_first_plan(enriched, prompt=prompt)
    return enriched


def install() -> None:
    """Install research-first design and leaf-granular execution exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_research_game_design as agentic
    from . import game_design

    original_messages = agentic._section_messages
    if not getattr(original_messages, _SECTION_MARKER, False):
        _deep_section_messages.__wrapped__ = original_messages  # type: ignore[attr-defined]
        setattr(_deep_section_messages, _SECTION_MARKER, True)
        agentic._section_messages = _deep_section_messages

    original_generator = game_design._generate_game_design_once
    if not getattr(original_generator, _GENERATOR_MARKER, False):
        _research_first_generate_once.__wrapped__ = original_generator  # type: ignore[attr-defined]
        setattr(_research_first_generate_once, _GENERATOR_MARKER, True)
        game_design._generate_game_design_once = _research_first_generate_once

    original_steps = _evidence._semantic_steps
    if not getattr(original_steps, _STEPS_MARKER, False):
        _semantic_steps_with_design_leaves.__wrapped__ = original_steps  # type: ignore[attr-defined]
        setattr(_semantic_steps_with_design_leaves, _STEPS_MARKER, True)
        _evidence._semantic_steps = _semantic_steps_with_design_leaves

    original_tasks = _evidence._compile_tasks
    if not getattr(original_tasks, _TASKS_MARKER, False):
        _compile_tasks_with_design_reuse.__wrapped__ = original_tasks  # type: ignore[attr-defined]
        setattr(_compile_tasks_with_design_reuse, _TASKS_MARKER, True)
        _evidence._compile_tasks = _compile_tasks_with_design_reuse

    original_validate = _evidence.validate_evidence_first_plan
    if not getattr(original_validate, _VALIDATE_MARKER, False):
        _validate_plan_with_design_context.__wrapped__ = original_validate  # type: ignore[attr-defined]
        setattr(_validate_plan_with_design_context, _VALIDATE_MARKER, True)
        _evidence.validate_evidence_first_plan = _validate_plan_with_design_context

    original_compile = _evidence.compile_evidence_first_plan
    if not getattr(original_compile, _COMPILE_MARKER, False):
        _compile_plan_with_design_context.__wrapped__ = original_compile  # type: ignore[attr-defined]
        setattr(_compile_plan_with_design_context, _COMPILE_MARKER, True)
        _evidence.compile_evidence_first_plan = _compile_plan_with_design_context

    _INSTALLED = True


__all__ = ["install"]
