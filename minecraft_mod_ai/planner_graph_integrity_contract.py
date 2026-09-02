from __future__ import annotations

"""Production-depth planning policies installed after semantic authority is frozen."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from . import evidence_first_planning as _evidence
from .canonical_capability_ontology import atomic_capability_definitions

_INSTALLED = False
_TOKEN = re.compile(r"[\w]{2,}", re.UNICODE)


def _task_capability(gap: Mapping[str, Any]) -> str:
    return str(gap.get("capability") or "").strip().casefold().removeprefix("capability:")


def _would_create_capability_cycle(
    edges: Mapping[str, Sequence[str]], consumer: str, provider: str
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
    """Bind producer prerequisites to consumers before the host freezes the task DAG."""

    original = _compile_tasks_with_cross_system_dependencies.__wrapped__
    tasks = original(gaps, reuse, target, branches, ownership)
    if not tasks or not gaps:
        return tasks

    definitions = atomic_capability_definitions()
    gaps_by_capability: dict[str, Mapping[str, Any]] = {}
    first_task_by_gap: dict[str, str] = {}
    provide_by_capability: dict[str, str] = {}

    for gap in gaps:
        capability = _task_capability(gap)
        if not capability or capability in gaps_by_capability:
            continue
        gaps_by_capability[capability] = gap
        missing = [str(item) for item in gap.get("missing_provides", ()) if str(item)]
        if missing:
            provide_by_capability[capability] = missing[0]

    mutable = [dict(task) for task in tasks]
    for task in mutable:
        for gap_ref in task.get("gap_refs", ()):
            first_task_by_gap.setdefault(str(gap_ref), str(task["task_id"]))
    by_id = {str(task["task_id"]): task for task in mutable}
    accepted: dict[str, list[str]] = {capability: [] for capability in gaps_by_capability}

    for capability, gap in gaps_by_capability.items():
        definition = definitions.get(capability)
        if definition is None:
            continue
        consumer_id = first_task_by_gap.get(str(gap.get("gap_id") or ""))
        if not consumer_id:
            continue
        consumer = by_id[consumer_id]
        for raw_dependency in definition.default_dependencies:
            dependency = str(raw_dependency).strip().casefold()
            if dependency not in gaps_by_capability:
                continue
            if _would_create_capability_cycle(accepted, capability, dependency):
                continue
            provided = provide_by_capability.get(dependency)
            if not provided:
                continue
            consumes = [str(item) for item in consumer.get("consumes", ()) if str(item)]
            if provided not in consumes:
                consumes.append(provided)
                consumer["consumes"] = consumes
            accepted.setdefault(capability, []).append(dependency)

    # This binder derives task dependencies from producer/consumer artifacts. It is
    # generic: no space/ship/economy feature names are hard-coded into the DAG policy.
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
    from . import semantic_requirement_authority as semantic

    max_clause_index = max(int(clause["clause_index"]) for clause in clauses)
    parameters = semantic._semantic_schema(max_clause_index, max(1, len(clauses) * 64))
    requirements = parameters.get("properties", {}).get("requirements")
    if isinstance(requirements, dict):
        requirements.pop("maxItems", None)

    messages = semantic._model_messages(clauses, repair_diagnostics=repair_diagnostics)
    if messages and isinstance(messages[0], dict):
        messages = [dict(item) for item in messages]
        messages[0]["content"] = (
            str(messages[0].get("content") or "")
            + "\n\nDECOMPOSITION DEPTH CONTRACT: A clause may contain many independent behaviors "
            "even without punctuation. Never compress multiple verbs or independently "
            "observable outcomes into one umbrella requirement. Split resource flows, "
            "state transitions, purchases, assembly, upgrades, travel phases, combat "
            "outcomes, world interactions, and persistent outcomes when each can fail "
            "independently. Emit every authored leaf: there is no fixed leaf target or "
            "per-clause leaf ceiling. Each leaf gets one concrete Given/When/Then behavior. "
            "Do not invent unrelated features, APIs, storage, networking, UI, or third-party "
            "mods that the authored request does not require."
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
    original = _production_depth_game_design_prompt.__wrapped__
    return (
        original()
        + "\n\nPRODUCTION-DEPTH DESIGN CONTRACT:\n"
        "- Complete the gameplay/mod design before choosing any third-party implementation "
        "or ecosystem donor. Search and reuse happen only after this design is frozen.\n"
        "- Preserve authored scope, but expand requested mechanics into the smallest meaningful "
        "subsystems that can be independently implemented, tested, and searched for reuse.\n"
        "- Use as many module entries as the design needs; there is no arbitrary module count.\n"
        "- core_loop and progression must expose prerequisites, state changes, costs/rewards, "
        "and unlock transitions in executable order.\n"
        "- Acceptance tests must cover mechanics that can fail independently.\n"
        "- Record assumptions precisely enough for downstream reviewed Skills/MCP research "
        "to verify exact Minecraft/Fabric APIs, versions, licenses, and reusable implementations.\n"
        "- Do not add unrelated features merely to make the design longer."
    )


def _facet_identifier(namespace: str, text: str) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", text.casefold()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    suffix = clean[:52] if clean and clean[0].isalpha() else hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:12]
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
    detail = " ".join(str(detail or "").split()) or " ".join(str(label or "").split())
    label = " ".join(str(label or "").split())
    if not detail or capability in seen:
        return
    seen.add(capability)
    output.append(
        {
            "capability": capability,
            "label": label or capability,
            "detail": detail,
            "source": source,
        }
    )


def _design_facets(design: Mapping[str, Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    modules = design.get("modules")
    if isinstance(modules, Sequence) and not isinstance(modules, (str, bytes, bytearray)):
        for index, raw in enumerate(modules):
            if not isinstance(raw, Mapping):
                continue
            plugin_id = re.sub(
                r"[^a-z0-9_]+", "_", str(raw.get("plugin_id") or "").strip().casefold()
            ).strip("_")
            if plugin_id:
                _append_design_facet(
                    output,
                    seen,
                    capability=f"design.module.{plugin_id[:63]}",
                    label=plugin_id.replace("_", " "),
                    detail=str(raw.get("reason") or plugin_id),
                    source=f"game_design.modules[{index}]",
                )

    for field in ("core_loop", "progression"):
        values = design.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        for index, raw in enumerate(values):
            text = " ".join(str(raw or "").split())
            if text:
                _append_design_facet(
                    output,
                    seen,
                    capability=_facet_identifier(field, text),
                    label=field.replace("_", " "),
                    detail=text,
                    source=f"game_design.{field}[{index}]",
                )

    for field in ("combat", "mod_context"):
        mapping = design.get(field)
        if not isinstance(mapping, Mapping):
            continue
        for key, raw_values in mapping.items():
            values = (
                raw_values
                if isinstance(raw_values, Sequence)
                and not isinstance(raw_values, (str, bytes, bytearray))
                else (raw_values,)
            )
            for index, raw in enumerate(values):
                text = " ".join(str(raw or "").split())
                if text:
                    _append_design_facet(
                        output,
                        seen,
                        capability=_facet_identifier(field, f"{key} {text}"),
                        label=str(key),
                        detail=text,
                        source=f"game_design.{field}.{key}[{index}]",
                    )
    return output


def _text_tokens(value: Any) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(str(value or ""))}


def _work_text(work: Mapping[str, Any]) -> str:
    return " ".join(
        (
            str(work.get("objective") or ""),
            " ".join(str(item) for item in work.get("capabilities", ()) if str(item)),
            " ".join(str(item) for item in work.get("acceptance", ()) if str(item)),
        )
    )


def _facet_work_index(
    facet: Mapping[str, str],
    work: Sequence[Mapping[str, Any]],
    facet_index: int,
    facet_count: int,
) -> int:
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
            None, facet_text.casefold(), candidate_text.casefold(), autojunk=False
        ).ratio()
        score = lexical * 4.0 + ratio
        if score > best_score:
            best_score = score
            best_index = index
    if best_score <= 0.15 and work:
        return min(
            len(work) - 1,
            (facet_index * len(work)) // max(1, facet_count),
        )
    return best_index


def _compile_pre_retrieval_plan_with_design_facets(
    prompt: str, design: Mapping[str, Any]
) -> dict[str, Any]:
    from . import reuse_planner as reuse

    original = _compile_pre_retrieval_plan_with_design_facets.__wrapped__
    plan = original(prompt, design)
    facets = _design_facets(design)
    raw_work = plan.get("planned_work")
    raw_graph = plan.get("capability_graph")
    if not facets or not isinstance(raw_work, list) or not isinstance(raw_graph, Mapping):
        return plan

    work = [dict(item) for item in raw_work if isinstance(item, Mapping)]
    if not work:
        return plan
    graph = dict(raw_graph)
    nodes = [str(item) for item in graph.get("nodes", ()) if str(item)]
    edges = [dict(item) for item in graph.get("edges", ()) if isinstance(item, Mapping)]
    sources = [dict(item) for item in graph.get("sources", ()) if isinstance(item, Mapping)]
    search_terms = [
        dict(item) for item in graph.get("search_terms", ()) if isinstance(item, Mapping)
    ]
    known = set(nodes)
    bindings: list[dict[str, str]] = []

    for facet_index, facet in enumerate(facets):
        capability = facet["capability"]
        if capability in known:
            continue
        owner = work[_facet_work_index(facet, work, facet_index, len(facets))]
        owner_caps = [str(item) for item in owner.get("capabilities", ()) if str(item)]
        parent = next(
            (item for item in owner_caps if not item.startswith("design.")),
            owner_caps[0] if owner_caps else "",
        )
        owner_caps.append(capability)
        owner["capabilities"] = list(dict.fromkeys(owner_caps))
        nodes.append(capability)
        known.add(capability)
        sources.append({"capability": capability, "source": facet["source"]})
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
                        )
                    )
                ),
            }
        )
        if parent and parent != capability:
            edge = {"from": parent, "to": capability}
            if edge not in edges:
                edges.append(edge)
        bindings.append(
            {
                "capability": capability,
                "work_id": str(owner.get("work_id") or ""),
                "requirement_ref": str(owner.get("requirement_ref") or ""),
                "source": facet["source"],
            }
        )

    if not bindings:
        return plan
    graph["nodes"] = nodes
    graph["edges"] = edges
    graph["sources"] = sources
    graph["search_terms"] = search_terms
    mutable = dict(plan)
    mutable["planned_work"] = work
    mutable["capability_graph"] = graph
    mutable["design_retrieval_facets"] = bindings
    mutable["plan_sha256"] = ""
    mutable["plan_sha256"] = reuse._plan_hash(mutable)
    reuse.validate_pre_retrieval_plan(mutable, prompt=prompt, design=design)
    return mutable


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import game_design
    from . import reuse_planner as reuse
    from . import semantic_requirement_authority as semantic

    original_compile = _evidence._compile_tasks
    if not getattr(original_compile, "_mmm_cross_system_dependencies", False):
        _compile_tasks_with_cross_system_dependencies.__wrapped__ = original_compile
        _compile_tasks_with_cross_system_dependencies._mmm_cross_system_dependencies = True
        _evidence._compile_tasks = _compile_tasks_with_cross_system_dependencies

    original_semantic = semantic._call_semantic_model
    if not getattr(original_semantic, "_mmm_deep_semantic_leaf_planning", False):
        _semantic_model_with_leaf_decomposition.__wrapped__ = original_semantic
        _semantic_model_with_leaf_decomposition._mmm_deep_semantic_leaf_planning = True
        semantic._call_semantic_model = _semantic_model_with_leaf_decomposition

    original_prompt = game_design._system_prompt
    if not getattr(original_prompt, "_mmm_production_depth_game_design", False):
        _production_depth_game_design_prompt.__wrapped__ = original_prompt
        _production_depth_game_design_prompt._mmm_production_depth_game_design = True
        game_design._system_prompt = _production_depth_game_design_prompt

    original_plan = reuse.compile_pre_retrieval_plan
    if not getattr(original_plan, "_mmm_design_facet_retrieval", False):
        _compile_pre_retrieval_plan_with_design_facets.__wrapped__ = original_plan
        _compile_pre_retrieval_plan_with_design_facets._mmm_design_facet_retrieval = True
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
