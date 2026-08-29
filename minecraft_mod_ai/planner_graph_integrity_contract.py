from __future__ import annotations

"""Restore production-depth planning without moving semantic authority.

The authored request remains owned by ``semantic_requirement_authority``. This contract
adds four late planning policies after that authority is installed:

* semantic model calls must decompose compound clauses into independently testable leaves;
* game design must be completed at reusable subsystem granularity before ecosystem search;
* design subsystems become retrieval/reuse facets under the already-approved requirement
  identities, never new public requirements; and
* cross-system ontology prerequisites are bound into the implementation task DAG.

This keeps source provenance and public acceptance stable while preventing a broad request
from collapsing into a handful of giant custom Java tasks.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from . import evidence_first_planning as _evidence
from .canonical_capability_ontology import atomic_capability_definitions

_INSTALLED = False
_MARKER = "__mmm_cross_system_dependencies__"
_SEMANTIC_MARKER = "__mmm_deep_semantic_leaf_planning__"
_DESIGN_MARKER = "__mmm_production_depth_game_design__"
_REUSE_MARKER = "__mmm_design_facet_retrieval__"
_TOKEN = re.compile(r"[\w]{2,}", re.UNICODE)


def _task_capability(gap: Mapping[str, Any]) -> str:
    return (
        str(gap.get("capability") or "")
        .strip()
        .casefold()
        .removeprefix("capability:")
    )


def _would_create_capability_cycle(
    edges: Mapping[str, Sequence[str]],
    consumer: str,
    provider: str,
) -> bool:
    if consumer == provider:
        return True
    stack = [provider]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == consumer:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(str(item) for item in edges.get(current, ()))
    return False


def _compile_tasks_with_cross_system_dependencies(
    gaps: Sequence[Mapping[str, Any]],
    reuse: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Add prerequisite consumes to the first task of each unresolved capability."""

    original = _compile_tasks_with_cross_system_dependencies.__wrapped__
    tasks = original(gaps, reuse, target, branches, ownership)
    if not tasks or not gaps:
        return tasks

    definitions = atomic_capability_definitions()
    gap_by_capability: dict[str, Mapping[str, Any]] = {}
    first_task_by_gap: dict[str, str] = {}
    required_provide_by_capability: dict[str, str] = {}

    for gap in gaps:
        capability = _task_capability(gap)
        if not capability or capability in gap_by_capability:
            continue
        gap_by_capability[capability] = gap
        missing = [str(item) for item in gap.get("missing_provides", ()) if str(item)]
        if missing:
            required_provide_by_capability[capability] = missing[0]

    mutable = [dict(task) for task in tasks]
    for task in mutable:
        for gap_ref in task.get("gap_refs", ()):
            first_task_by_gap.setdefault(str(gap_ref), str(task["task_id"]))
    task_by_id = {str(task["task_id"]): task for task in mutable}

    accepted_edges: dict[str, list[str]] = {
        capability: [] for capability in gap_by_capability
    }
    for capability, gap in gap_by_capability.items():
        definition = definitions.get(capability)
        if definition is None:
            continue
        consumer_task_id = first_task_by_gap.get(str(gap.get("gap_id") or ""))
        if not consumer_task_id:
            continue
        consumer_task = task_by_id[consumer_task_id]

        for raw_dependency in definition.default_dependencies:
            dependency = str(raw_dependency).strip().casefold()
            if dependency not in gap_by_capability:
                continue
            if _would_create_capability_cycle(
                accepted_edges,
                capability,
                dependency,
            ):
                continue
            required_provide = required_provide_by_capability.get(dependency)
            if not required_provide:
                continue

            consumes = [
                str(item) for item in consumer_task.get("consumes", ()) if str(item)
            ]
            if required_provide not in consumes:
                consumes.append(required_provide)
                consumer_task["consumes"] = consumes
            accepted_edges.setdefault(capability, []).append(dependency)

    return _evidence._bind_consumes_dependencies(
        mutable,
        root_provides={"target:frozen"},
    )


def _semantic_model_with_leaf_decomposition(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
    *,
    repair_diagnostics: Sequence[Mapping[str, Any]] = (),
) -> Any:
    """Run the canonical semantic call without an arbitrary per-clause item ceiling."""

    from . import semantic_requirement_authority as semantic

    max_clause_index = max(int(clause["clause_index"]) for clause in clauses)
    parameters = semantic._semantic_schema(max_clause_index, max(1, len(clauses) * 64))
    requirements = parameters.get("properties", {}).get("requirements")
    if isinstance(requirements, dict):
        # The request itself determines cardinality. Small-model schemas used to cap one
        # clause at eight leaves, which silently forced unrelated mechanics back together.
        requirements.pop("maxItems", None)

    messages = semantic._model_messages(
        clauses,
        repair_diagnostics=repair_diagnostics,
    )
    if messages and isinstance(messages[0], dict):
        messages = [dict(item) for item in messages]
        messages[0]["content"] = (
            str(messages[0].get("content") or "")
            + "\n\nDECOMPOSITION DEPTH CONTRACT: A clause may contain many independent "
            "behaviors even when the author used no punctuation. Split conjunctions, "
            "sequences, lists, resource flows, state transitions, purchases, assembly "
            "steps, upgrades, travel phases, combat outcomes, world interactions, and "
            "persistence-visible outcomes whenever each can be implemented and observed "
            "independently. Never compress several verbs or several player-visible "
            "outcomes into one broad umbrella requirement merely to keep the response "
            "short. Emit as many leaf requirements as the authored clause actually "
            "contains; there is no fixed item target. Each leaf must have one concrete "
            "Given/When/Then behavior. Do not invent unrelated features, implementation "
            "classes, APIs, networking, storage, UI, or third-party mods that the request "
            "does not require."
        )

    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        return native(
            "planner",
            messages,
            tool_name="compile_semantic_requirements",
            parameters=parameters,
            description=(
                "Compile every independently observable authored behavior into semantic "
                "leaf requirements. The host owns exact source grounding."
            ),
        )
    raw = router.generate_text(
        "planner",
        messages,
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    return semantic._parse_json(raw)


def _production_depth_game_design_prompt() -> str:
    """Require a design artifact that is useful before any donor/reuse search."""

    original = _production_depth_game_design_prompt.__wrapped__
    return (
        original()
        + "\n\nPRODUCTION-DEPTH DESIGN CONTRACT:\n"
        "- Complete the gameplay/mod design before choosing any third-party implementation "
        "or ecosystem donor. Search and reuse happen only after this design is frozen.\n"
        "- Preserve the authored scope, but expand every requested mechanic until the "
        "modules array names the smallest meaningful subsystems that can be independently "
        "implemented, tested, and searched for reuse. Separate different verbs, states, "
        "resources, progression gates, interactions, upgrades, destinations, combat "
        "behaviors, and outcomes instead of hiding them under one epic module.\n"
        "- modules[].plugin_id is a stable English snake_case leaf-system identifier. "
        "Use as many module entries as the design needs; there is no arbitrary module "
        "count. modules[].reason must state the concrete behavior/state transition owned "
        "by that subsystem, not merely repeat the theme.\n"
        "- core_loop and progression must describe concrete player actions, prerequisites, "
        "state changes, rewards/costs, and unlock transitions in executable order. combat "
        "and mod_context must expose distinct systems rather than one generic summary.\n"
        "- acceptance_tests must cover the independent player-visible mechanics and their "
        "important transitions. A single broad end-to-end sentence is not enough when "
        "several mechanics can fail independently.\n"
        "- Record design-critical assumptions precisely enough that downstream reviewed "
        "Skills/MCP research can verify Minecraft/Fabric behavior, exact APIs, versions, "
        "licenses, and reusable implementations. Do not fabricate API symbols or claim a "
        "library/mod has been selected before that evidence exists.\n"
        "- Do not add unrelated features just to make the design longer."
    )


def _facet_identifier(namespace: str, text: str) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", text.casefold()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    if clean and clean[0].isalpha():
        suffix = clean[:52]
    else:
        suffix = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"design.{namespace}.{suffix}"


def _append_design_facet(
    output: list[dict[str, str]],
    seen: set[str],
    *,
    capability: str,
    label: str,
    detail: str,
    source: str,
) -> None:
    normalized_detail = " ".join(str(detail or "").split())
    normalized_label = " ".join(str(label or "").split())
    if not normalized_detail:
        normalized_detail = normalized_label
    if not normalized_detail or capability in seen:
        return
    seen.add(capability)
    output.append(
        {
            "capability": capability,
            "label": normalized_label or capability,
            "detail": normalized_detail,
            "source": source,
        }
    )


def _design_facets(design: Mapping[str, Any]) -> list[dict[str, str]]:
    """Extract reusable implementation/search facets without changing public requirements."""

    output: list[dict[str, str]] = []
    seen: set[str] = set()

    raw_modules = design.get("modules")
    if isinstance(raw_modules, Sequence) and not isinstance(
        raw_modules, (str, bytes, bytearray)
    ):
        for index, raw in enumerate(raw_modules):
            if not isinstance(raw, Mapping):
                continue
            plugin_id = re.sub(
                r"[^a-z0-9_]+",
                "_",
                str(raw.get("plugin_id") or "").strip().casefold(),
            ).strip("_")
            if not plugin_id:
                continue
            reason = str(raw.get("reason") or plugin_id)
            _append_design_facet(
                output,
                seen,
                capability=f"design.module.{plugin_id[:63]}",
                label=plugin_id.replace("_", " "),
                detail=reason,
                source=f"game_design.modules[{index}]",
            )

    for field in ("core_loop", "progression"):
        values = design.get(field)
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            continue
        for index, raw in enumerate(values):
            text = " ".join(str(raw or "").split())
            if not text:
                continue
            capability = _facet_identifier(field, text)
            _append_design_facet(
                output,
                seen,
                capability=capability,
                label=field.replace("_", " "),
                detail=text,
                source=f"game_design.{field}[{index}]",
            )

    for field in ("combat", "mod_context"):
        mapping = design.get(field)
        if not isinstance(mapping, Mapping):
            continue
        for key, raw_values in mapping.items():
            values: Sequence[Any]
            if isinstance(raw_values, Sequence) and not isinstance(
                raw_values, (str, bytes, bytearray)
            ):
                values = raw_values
            else:
                values = (raw_values,)
            for index, raw in enumerate(values):
                text = " ".join(str(raw or "").split())
                if not text:
                    continue
                label = f"{key} {text}"
                capability = _facet_identifier(field, label)
                _append_design_facet(
                    output,
                    seen,
                    capability=capability,
                    label=str(key),
                    detail=text,
                    source=f"game_design.{field}.{key}[{index}]",
                )
    return output


def _text_tokens(value: Any) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(str(value or ""))}


def _work_text(work: Mapping[str, Any]) -> str:
    parts = [
        str(work.get("objective") or ""),
        " ".join(str(item) for item in work.get("capabilities", ()) if str(item)),
        " ".join(str(item) for item in work.get("acceptance", ()) if str(item)),
    ]
    return " ".join(parts)


def _facet_work_index(
    facet: Mapping[str, str],
    work: Sequence[Mapping[str, Any]],
    facet_index: int,
    facet_count: int,
) -> int:
    """Bind a design facet to the closest authored requirement, preserving order as fallback."""

    facet_text = f"{facet.get('label', '')} {facet.get('detail', '')}"
    facet_tokens = _text_tokens(facet_text)
    best_index = 0
    best_score = -1.0
    for index, candidate in enumerate(work):
        candidate_text = _work_text(candidate)
        candidate_tokens = _text_tokens(candidate_text)
        overlap = len(facet_tokens & candidate_tokens)
        lexical = overlap / max(1, len(facet_tokens))
        ratio = SequenceMatcher(
            None,
            facet_text.casefold(),
            candidate_text.casefold(),
            autojunk=False,
        ).ratio()
        score = lexical * 4.0 + ratio
        if score > best_score:
            best_score = score
            best_index = index

    if best_score <= 0.15 and work:
        # When languages/wording differ, narrative order is a safer deterministic fallback
        # than attaching every facet to the first broad requirement.
        best_index = min(
            len(work) - 1,
            (facet_index * len(work)) // max(1, facet_count),
        )
    return best_index


def _compile_pre_retrieval_plan_with_design_facets(
    prompt: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Add game-design leaf systems to donor/reuse search after semantic scope is frozen."""

    from . import reuse_planner as reuse

    original = _compile_pre_retrieval_plan_with_design_facets.__wrapped__
    plan = original(prompt, design)
    facets = _design_facets(design)
    if not facets:
        return plan

    mutable = dict(plan)
    raw_work = plan.get("planned_work")
    raw_graph = plan.get("capability_graph")
    if not isinstance(raw_work, list) or not isinstance(raw_graph, Mapping):
        return plan

    work = [dict(item) for item in raw_work if isinstance(item, Mapping)]
    if not work:
        return plan
    graph = dict(raw_graph)
    nodes = [str(item) for item in graph.get("nodes", ()) if str(item)]
    edges = [
        dict(item)
        for item in graph.get("edges", ())
        if isinstance(item, Mapping)
    ]
    sources = [
        dict(item)
        for item in graph.get("sources", ())
        if isinstance(item, Mapping)
    ]
    search_terms = [
        dict(item)
        for item in graph.get("search_terms", ())
        if isinstance(item, Mapping)
    ]
    known_nodes = set(nodes)
    facet_bindings: list[dict[str, str]] = []

    for facet_index, facet in enumerate(facets):
        capability = facet["capability"]
        if capability in known_nodes:
            continue
        work_index = _facet_work_index(facet, work, facet_index, len(facets))
        owner = work[work_index]
        owner_capabilities = [
            str(item) for item in owner.get("capabilities", ()) if str(item)
        ]
        parent_capability = next(
            (item for item in owner_capabilities if not item.startswith("design.")),
            owner_capabilities[0] if owner_capabilities else "",
        )
        owner_capabilities.append(capability)
        owner["capabilities"] = list(dict.fromkeys(owner_capabilities))

        nodes.append(capability)
        known_nodes.add(capability)
        sources.append(
            {
                "capability": capability,
                "source": facet["source"],
            }
        )
        label = facet["label"].replace("_", " ")
        detail = facet["detail"]
        search_terms.append(
            {
                "capability": capability,
                "terms": list(
                    dict.fromkeys(
                        (
                            f"{label} Minecraft mod implementation {detail}",
                            f"{label} Fabric mod source code",
                            f"{detail} reusable Minecraft mod",
                            f"{label} Modrinth GitHub Fabric",
                        )
                    )
                ),
            }
        )
        if parent_capability and parent_capability != capability:
            edge = {"from": parent_capability, "to": capability}
            if edge not in edges:
                edges.append(edge)
        facet_bindings.append(
            {
                "capability": capability,
                "work_id": str(owner.get("work_id") or ""),
                "requirement_ref": str(owner.get("requirement_ref") or ""),
                "source": facet["source"],
            }
        )

    if not facet_bindings:
        return plan

    graph["nodes"] = nodes
    graph["edges"] = edges
    graph["sources"] = sources
    graph["search_terms"] = search_terms
    mutable["planned_work"] = work
    mutable["capability_graph"] = graph
    mutable["design_retrieval_facets"] = facet_bindings
    mutable["plan_sha256"] = ""
    mutable["plan_sha256"] = reuse._plan_hash(mutable)
    reuse.validate_pre_retrieval_plan(mutable, prompt=prompt, design=design)
    return mutable


def install() -> None:
    """Install the late planning policies exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import game_design as game_design
    from . import reuse_planner as reuse
    from . import semantic_requirement_authority as semantic

    original_compile = _evidence._compile_tasks
    if not getattr(original_compile, _MARKER, False):
        _compile_tasks_with_cross_system_dependencies.__wrapped__ = original_compile  # type: ignore[attr-defined]
        setattr(_compile_tasks_with_cross_system_dependencies, _MARKER, True)
        _evidence._compile_tasks = _compile_tasks_with_cross_system_dependencies

    original_semantic_call = semantic._call_semantic_model
    if not getattr(original_semantic_call, _SEMANTIC_MARKER, False):
        _semantic_model_with_leaf_decomposition.__wrapped__ = original_semantic_call  # type: ignore[attr-defined]
        setattr(_semantic_model_with_leaf_decomposition, _SEMANTIC_MARKER, True)
        semantic._call_semantic_model = _semantic_model_with_leaf_decomposition

    original_design_prompt = game_design._system_prompt
    if not getattr(original_design_prompt, _DESIGN_MARKER, False):
        _production_depth_game_design_prompt.__wrapped__ = original_design_prompt  # type: ignore[attr-defined]
        setattr(_production_depth_game_design_prompt, _DESIGN_MARKER, True)
        game_design._system_prompt = _production_depth_game_design_prompt

    original_pre_retrieval = reuse.compile_pre_retrieval_plan
    if not getattr(original_pre_retrieval, _REUSE_MARKER, False):
        _compile_pre_retrieval_plan_with_design_facets.__wrapped__ = original_pre_retrieval  # type: ignore[attr-defined]
        setattr(_compile_pre_retrieval_plan_with_design_facets, _REUSE_MARKER, True)
        reuse.compile_pre_retrieval_plan = _compile_pre_retrieval_plan_with_design_facets

    _INSTALLED = True


__all__ = [
    "_compile_pre_retrieval_plan_with_design_facets",
    "_design_facets",
    "_facet_work_index",
    "_production_depth_game_design_prompt",
    "_semantic_model_with_leaf_decomposition",
    "install",
]
