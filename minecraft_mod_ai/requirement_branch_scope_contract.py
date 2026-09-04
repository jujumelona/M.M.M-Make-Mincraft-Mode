from __future__ import annotations

"""Keep conditional implementation branches local to the requirement that activated them.

The base evidence planner historically concatenated every requirement into one text blob and
activated branches globally. A GUI clause could therefore make an unrelated machine task
inherit client-screen work, or a persistence clause could make sibling tasks inherit codec
work. Branches are architecture decisions, not authored requirements, so their provenance
must name the exact requirement(s) that activated them and task compilation must consume only
that local view.
"""

import re
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from . import evidence_first_planning as _planning

_INSTALLED = False

_TERMS: dict[str, tuple[str, ...]] = {
    "needs_registry": (
        "item",
        "block",
        "block_entity",
        "machine",
        "entity",
        "recipe",
        "effect",
        "enchantment",
        "fluid",
        "biome",
        "dimension",
        "registry",
    ),
    "needs_datagen": ("recipe", "loot", "tag", "model", "worldgen", "datagen"),
    "needs_persistence": (
        "persistence",
        "saved",
        "storage",
        "serialize",
        "codec",
        "world_state",
    ),
    "needs_network": ("network", "payload", "packet", "sync"),
    "needs_client_render": (
        "gui",
        "screen",
        "render",
        "texture",
        "model",
        "client",
        "hud",
    ),
    "needs_worldgen": (
        "worldgen",
        "biome",
        "configured_feature",
        "placed_feature",
        "structure",
        "dimension",
    ),
    "needs_mixin": (
        "mixin",
        "optimization",
        "software.performance",
        "performance.optimization",
        "runtime.performance",
        "code.optimization",
        "performance optimization",
        "renderer_patch",
        "injection",
    ),
    "needs_loader_leaf": ("loader_leaf", "multiloader", "multi_loader"),
}


def _contains_term(text: str, term: str) -> bool:
    """Match semantic tokens without substring leakage.

    Compound capability IDs may use spaces, dots, hyphens or underscores interchangeably,
    while ordinary word prefixes/suffixes must not count. For example ``block_entity``
    matches ``block entity`` but ``item`` does not match ``itemization``.
    """

    parts = [part for part in re.split(r"[_\s.-]+", term.casefold()) if part]
    if not parts:
        return False
    compound = r"[_\s.-]+".join(re.escape(part) for part in parts)
    return re.search(rf"(?<![a-z0-9]){compound}(?![a-z0-9])", text.casefold()) is not None


def _requirement_text(requirement: Mapping[str, Any]) -> str:
    span = _planning._mapping(requirement.get("source_span"))
    values = [
        requirement.get("capability"),
        requirement.get("statement"),
        span.get("text"),
        *_planning._strings(requirement.get("gameplay_capabilities")),
        *_planning._strings(requirement.get("implementation_capabilities")),
    ]
    return " ".join(str(value or "").casefold() for value in values if str(value or "").strip())


def _component_supports_requirement(
    component: Mapping[str, Any], requirement: Mapping[str, Any]
) -> bool:
    required = {
        _planning._canonical_capability(value)
        for value in _planning._strings(requirement.get("provides"))
    }
    if not required:
        capability = _planning._canonical_capability(requirement.get("capability"))
        if capability:
            required.add(capability)
    provided = {
        _planning._canonical_capability(value)
        for value in _planning._strings(component.get("provides"))
        if str(value).casefold().startswith("capability:")
    }
    return bool(required & provided)


def _scoped_branch_predicates(
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    topology = _planning._mapping(target.get("project_topology"))
    loaders = _planning._strings(topology.get("loaders"))
    multi_loader = len(loaders) > 1
    result: dict[str, dict[str, Any]] = {}

    for branch in _planning._BRANCHES:
        per_requirement: dict[str, str] = {}
        evidence_by_requirement: dict[str, list[str]] = {}
        active_refs: list[str] = []
        for requirement in requirements:
            requirement_id = str(requirement.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            text = _requirement_text(requirement)
            active = any(_contains_term(text, term) for term in _TERMS[branch])
            if (
                branch == "needs_mixin"
                and requirement.get("semantic_type")
                and requirement.get("semantic_type") != "software_quality"
            ):
                active = False
            component_refs: list[str] = []
            if branch == "needs_datagen" and not active:
                component_refs = [
                    str(component.get("component_id") or "")
                    for component in components
                    if str(component.get("kind") or "").casefold() == "generated_resource"
                    and _component_supports_requirement(component, requirement)
                    and str(component.get("component_id") or "").strip()
                ]
                active = bool(component_refs)
            if branch == "needs_loader_leaf" and multi_loader:
                active = True

            per_requirement[requirement_id] = "ACTIVE" if active else "NOT_APPLICABLE"
            refs = [f"requirement:{requirement_id}"] if active else []
            refs.extend(f"component:{item}" for item in component_refs)
            if branch == "needs_loader_leaf" and multi_loader:
                refs.append("target-topology:multiple-loaders")
            evidence_by_requirement[requirement_id] = refs
            if active:
                active_refs.append(requirement_id)

        result[branch] = {
            "predicate": branch,
            "status": "ACTIVE" if active_refs else "NOT_APPLICABLE",
            "evidence_refs": active_refs or ["request-catalog:no-matching-capability"],
            "requirement_status": per_requirement,
            "requirement_evidence_refs": evidence_by_requirement,
            "reason": (
                "activated only for the listed requirement evidence"
                if active_refs
                else "no requirement or project evidence activates this branch"
            ),
        }
    return result


def _branches_for_requirement(
    branches: Mapping[str, Mapping[str, Any]], requirement_ref: str
) -> dict[str, dict[str, Any]]:
    scoped: dict[str, dict[str, Any]] = {}
    for name in _planning._BRANCHES:
        raw = branches.get(name)
        value = dict(raw) if isinstance(raw, Mapping) else {}
        statuses = value.get("requirement_status")
        status = (
            str(statuses.get(requirement_ref) or "NOT_APPLICABLE")
            if isinstance(statuses, Mapping)
            else "NOT_APPLICABLE"
        )
        evidence = value.get("requirement_evidence_refs")
        refs = (
            list(_planning._strings(evidence.get(requirement_ref)))
            if isinstance(evidence, Mapping)
            else []
        )
        value["status"] = status
        value["evidence_refs"] = refs or [
            f"requirement:{requirement_ref}:branch-not-applicable"
        ]
        value["scoped_requirement_ref"] = requirement_ref
        scoped[name] = value
    return scoped


def install_requirement_branch_scope_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    _planning._branch_predicates = _scoped_branch_predicates

    current_tasks = _planning._compile_tasks
    if not getattr(current_tasks, "_mmm_requirement_scoped_branches", False):
        @wraps(current_tasks)
        def compile_tasks(gaps, reuse, target, branches, ownership):
            compiled: list[dict[str, Any]] = []
            for gap in gaps:
                if not isinstance(gap, Mapping):
                    continue
                requirement_ref = str(gap.get("requirement_ref") or "").strip()
                if not requirement_ref:
                    raise _planning.EvidencePlanError(
                        "Implementation gap is missing requirement_ref."
                    )
                scoped = _branches_for_requirement(branches, requirement_ref)
                compiled.extend(
                    current_tasks((gap,), reuse, target, scoped, ownership)
                )
            # Rebind globally after per-requirement compilation. This preserves any
            # legitimate cross-gap provider edge while preventing branch activation
            # from leaking between requirements.
            return _planning._bind_consumes_dependencies(
                compiled, root_provides={"target:frozen"}
            )

        compile_tasks._mmm_requirement_scoped_branches = True
        compile_tasks.__wrapped__ = current_tasks
        _planning._compile_tasks = compile_tasks

    _INSTALLED = True


__all__ = [
    "_branches_for_requirement",
    "_contains_term",
    "_scoped_branch_predicates",
    "install_requirement_branch_scope_contract",
]
