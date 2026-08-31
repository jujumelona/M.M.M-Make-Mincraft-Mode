from __future__ import annotations

"""Bind frozen game-design leaves to evidence-first production execution.

Game-design generation, research ordering, prompt formatting, parsing, and readiness are
owned by ``game_design`` and ``agentic_research_game_design``. This contract starts only
after design freeze: it refines implementation tasks with design leaves and verified reuse
without creating new public requirements.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from . import evidence_first_planning as _evidence

_INSTALLED = False
_STEPS_MARKER = "__mmm_design_leaf_semantic_steps__"
_TASKS_MARKER = "__mmm_design_leaf_reuse_binding__"
_COMPILE_MARKER = "__mmm_design_leaf_evidence_plan__"
_VALIDATE_MARKER = "__mmm_design_leaf_plan_validation__"

_INDEXED_SOURCE = re.compile(
    r"^game_design\.(modules|core_loop|progression)\[(\d+)\]$"
)
_MAPPING_SOURCE = re.compile(
    r"^game_design\.(combat|mod_context)\.([^\[]+)\[(\d+)\]$"
)


@dataclass(frozen=True, slots=True)
class _ExecutionState:
    facets: tuple[dict[str, Any], ...]
    by_parent: dict[str, tuple[dict[str, Any], ...]]
    reuse_refs_by_outcome: dict[str, tuple[str, ...]]


_ACTIVE_DESIGN_EXECUTION: ContextVar[
    _ExecutionState | tuple[dict[str, Any], ...] | None
] = ContextVar(
    "mmm_active_design_execution_facets",
    default=None,
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


def _reuse_payload(
    game_design: Mapping[str, Any],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(explicit, Mapping):
        return dict(explicit)
    return _evidence._reuse_payload(game_design)


def _indexed_value(
    game_design: Mapping[str, Any],
    field: str,
    index: int,
) -> Any:
    values = game_design.get(field)
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray)
    ):
        return None
    return values[index] if 0 <= index < len(values) else None


def _design_detail(
    game_design: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> str:
    explicit = " ".join(str(binding.get("detail") or "").split())
    if explicit:
        return explicit

    source = str(binding.get("source") or "")
    indexed = _INDEXED_SOURCE.fullmatch(source)
    if indexed:
        field, raw_index = indexed.groups()
        value = _indexed_value(game_design, field, int(raw_index))
        if field == "modules" and isinstance(value, Mapping):
            return " ".join(
                str(value.get("reason") or value.get("plugin_id") or "").split()
            )
        return " ".join(str(value or "").split())

    mapped = _MAPPING_SOURCE.fullmatch(source)
    if mapped:
        field, key, raw_index = mapped.groups()
        container = game_design.get(field)
        if isinstance(container, Mapping):
            values = container.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray)
            ):
                index = int(raw_index)
                if 0 <= index < len(values):
                    return " ".join(str(values[index] or "").split())
            elif values is not None and int(raw_index) == 0:
                return " ".join(str(values).split())

    return " ".join(str(binding.get("capability") or "").split())


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

    raw_reuse = {
        str(item.get("capability") or ""): dict(item)
        for item in _reuse_payload(game_design, reuse_plan).get("capabilities", ())
        if isinstance(item, Mapping) and str(item.get("capability") or "")
    }
    module_parents = {
        str(binding.get("requirement_ref") or "")
        for binding in bindings
        if isinstance(binding, Mapping)
        and str(binding.get("source") or "").startswith("game_design.modules[")
    }

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        requirement_ref = str(binding.get("requirement_ref") or "")
        requirement = requirement_by_id.get(requirement_ref)
        capability = str(binding.get("capability") or "")
        source = str(binding.get("source") or "")
        if requirement is None or not capability:
            continue
        if requirement_ref in module_parents and not source.startswith(
            "game_design.modules["
        ):
            continue
        key = (requirement_ref, capability)
        if key in seen:
            continue
        seen.add(key)

        parent_capability = str(requirement.get("capability") or "").strip()
        if not parent_capability:
            continue
        raw = raw_reuse.get(capability, {})
        mode = str(raw.get("mode") or "fresh").strip().casefold()
        refs: list[str] = []
        if mode != "fresh":
            refs.extend(_strings(raw.get("component_refs")))
            source_id = str(raw.get("source_id") or "").strip()
            if source_id:
                refs.append(source_id)
        output.append(
            {
                "requirement_ref": requirement_ref,
                "parent_capability": parent_capability,
                "capability": capability,
                "detail": _design_detail(game_design, binding) or capability,
                "source": source,
                "reuse_refs": list(dict.fromkeys(refs)),
                "reuse_mode": mode,
                "proof_level": str(raw.get("proof_level") or "").strip(),
            }
        )
    return tuple(output)


def _leaf_outcome(item: Mapping[str, Any]) -> str:
    return (
        "Implement and independently verify design leaf "
        f"{item.get('capability')}: {item.get('detail')}"
    )


def _build_execution_state(
    context: Sequence[Mapping[str, Any]],
) -> _ExecutionState:
    facets = tuple(dict(raw) for raw in context)
    grouped: dict[str, list[dict[str, Any]]] = {}
    refs_by_outcome: dict[str, tuple[str, ...]] = {}
    for item in facets:
        parent = str(item.get("parent_capability") or "")
        if parent:
            grouped.setdefault(parent, []).append(item)
        refs = _strings(item.get("reuse_refs"))
        if refs:
            refs_by_outcome[_leaf_outcome(item)] = refs
    return _ExecutionState(
        facets=facets,
        by_parent={key: tuple(values) for key, values in grouped.items()},
        reuse_refs_by_outcome=refs_by_outcome,
    )


def _active_execution_state() -> _ExecutionState:
    current = _ACTIVE_DESIGN_EXECUTION.get()
    if isinstance(current, _ExecutionState):
        return current
    if isinstance(current, tuple):
        # Backward-compatible test/extension path. Runtime compile/validation always sets
        # the pre-indexed state and therefore never pays this conversion per capability.
        return _build_execution_state(current)
    return _build_execution_state(())


def _semantic_steps_with_design_leaves(
    capability: str,
    branches: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    original = _semantic_steps_with_design_leaves.__wrapped__
    facets = _active_execution_state().by_parent.get(str(capability), ())
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

    refs_by_outcome = _active_execution_state().reuse_refs_by_outcome
    if not refs_by_outcome:
        return tasks

    output: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        refs = refs_by_outcome.get(str(task.get("semantic_outcome") or ""), ())
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
    token = _ACTIVE_DESIGN_EXECUTION.set(_build_execution_state(context))
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

    state = _build_execution_state(context)
    token = _ACTIVE_DESIGN_EXECUTION.set(state)
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
    enriched["design_execution_facets"] = [dict(item) for item in state.facets]
    enriched["plan_sha256"] = ""
    enriched["plan_sha256"] = _evidence._hash_without(enriched, "plan_sha256")
    _evidence.validate_evidence_first_plan(enriched, prompt=prompt)
    return enriched


def install() -> None:
    """Install only post-design leaf-granular evidence execution exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

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
