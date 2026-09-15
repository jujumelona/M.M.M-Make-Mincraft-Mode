from __future__ import annotations

"""Bind frozen game-design detail to the host Minecraft template plan.

Design leaves are bounded values for already-selected templates, not an alternate task
planner.  This module therefore records their provenance on the compiled plan but never
replaces ``_semantic_steps`` or ``_compile_tasks``.  The Minecraft template compiler is
the single implementation-architecture owner.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import evidence_first_planning as _evidence

_INSTALLED = False
_COMPILE_MARKER = "__mmm_design_leaf_evidence_plan__"

_INDEXED_SOURCE = re.compile(r"^game_design\.(modules|core_loop|progression)\[(\d+)\]$")
_MAPPING_SOURCE = re.compile(r"^game_design\.(combat|mod_context)\.([^\[]+)\[(\d+)\]$")


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(
        dict.fromkeys(
            text
            for item in values
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
    game_design: Mapping[str, Any], field: str, index: int
) -> Any:
    values = game_design.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return None
    return values[index] if 0 <= index < len(values) else None


def _design_detail(
    game_design: Mapping[str, Any], binding: Mapping[str, Any]
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
    """Collect frozen design/research leaves as template-fill evidence only."""

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
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        requirement_ref = str(binding.get("requirement_ref") or "")
        requirement = requirement_by_id.get(requirement_ref)
        capability = str(binding.get("capability") or "")
        source = str(binding.get("source") or "")
        if requirement is None or not capability:
            continue
        key = (requirement_ref, capability, source)
        if key in seen:
            continue
        seen.add(key)

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
                "parent_capability": str(requirement.get("capability") or "").strip(),
                "design_leaf_capability": capability,
                "detail": _design_detail(game_design, binding) or capability,
                "source": source,
                "reuse_refs": list(dict.fromkeys(refs)),
                "reuse_mode": mode,
                "proof_level": str(raw.get("proof_level") or "").strip(),
                "authority": "template_fill_evidence_only",
            }
        )
    return tuple(output)


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
    plan = dict(
        original(
            prompt,
            game_design,
            component_catalog=component_catalog,
            reuse_plan=reuse_plan,
            target_decision=target_decision,
            semantic_router=semantic_router,
        )
    )
    context = _execution_context(game_design, reuse_plan)
    plan["design_execution_facets"] = [dict(item) for item in context]
    plan["design_execution_policy"] = {
        "architecture_owner": "minecraft_template_compiler",
        "design_leaf_role": "bounded_template_values_and_reuse_evidence",
        "may_create_tasks": False,
        "may_change_dependencies": False,
        "may_change_template_id": False,
    }
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = _evidence._hash_without(plan, "plan_sha256")
    _evidence.validate_evidence_first_plan(plan, prompt=prompt)
    return plan


def install() -> None:
    """Attach design provenance without monkeypatching task architecture."""

    global _INSTALLED
    if _INSTALLED:
        return
    current = _evidence.compile_evidence_first_plan
    if not getattr(current, _COMPILE_MARKER, False):
        _compile_plan_with_design_context.__wrapped__ = current  # type: ignore[attr-defined]
        setattr(_compile_plan_with_design_context, _COMPILE_MARKER, True)
        _evidence.compile_evidence_first_plan = _compile_plan_with_design_context
    _INSTALLED = True


__all__ = ["install"]
