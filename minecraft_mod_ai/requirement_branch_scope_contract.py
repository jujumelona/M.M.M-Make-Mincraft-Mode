from __future__ import annotations

"""Expose per-requirement branch provenance from the Minecraft template feature model.

Branch selection is no longer inferred from substrings and this module no longer wraps
task compilation.  The template compiler owns task architecture; this contract only adds
requirement-local provenance to the branch receipt for diagnostics and downstream views.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import evidence_first_planning as _planning
from .minecraft_template_catalog import requirement_branch_features

_INSTALLED = False


def _contains_term(text: str, term: str) -> bool:
    """Compatibility helper for callers/tests; not used for branch architecture."""

    parts = [part for part in re.split(r"[_\s.-]+", term.casefold()) if part]
    if not parts:
        return False
    compound = r"[_\s.-]+".join(re.escape(part) for part in parts)
    return re.search(
        rf"(?<![a-z0-9]){compound}(?![a-z0-9])", text.casefold()
    ) is not None


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
    module_ids = _planning._strings(topology.get("module_ids"))
    multi_loader = len(loaders) > 1 or len(module_ids) > 1
    result: dict[str, dict[str, Any]] = {}

    for branch in _planning._BRANCHES:
        per_requirement: dict[str, str] = {}
        evidence_by_requirement: dict[str, list[str]] = {}
        active_refs: list[str] = []
        for requirement in requirements:
            requirement_id = str(requirement.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            active = branch in requirement_branch_features(requirement)
            component_refs: list[str] = []
            if branch == "needs_datagen" and not active:
                component_refs = [
                    str(component.get("component_id") or "")
                    for component in components
                    if str(component.get("kind") or "").casefold()
                    == "generated_resource"
                    and _component_supports_requirement(component, requirement)
                    and str(component.get("component_id") or "").strip()
                ]
                active = bool(component_refs)
            if branch == "needs_loader_leaf" and multi_loader:
                active = True

            per_requirement[requirement_id] = (
                "ACTIVE" if active else "NOT_APPLICABLE"
            )
            refs = [f"requirement:{requirement_id}:template-feature"] if active else []
            refs.extend(f"component:{item}" for item in component_refs)
            if branch == "needs_loader_leaf" and multi_loader:
                refs.append("target-topology:multiple-loader-modules")
            evidence_by_requirement[requirement_id] = refs
            if active:
                active_refs.append(requirement_id)

        result[branch] = {
            "predicate": branch,
            "status": "ACTIVE" if active_refs else "NOT_APPLICABLE",
            "evidence_refs": active_refs
            or ["host-template-catalog:no-matching-feature"],
            "requirement_status": per_requirement,
            "requirement_evidence_refs": evidence_by_requirement,
            "reason": (
                "activated by exact host template features for the listed requirements"
                if active_refs
                else "no selected template or target topology activates this branch"
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
    """Install provenance-only branch scoping; task DAG compilation stays untouched."""

    global _INSTALLED
    if _INSTALLED:
        return
    _planning._branch_predicates = _scoped_branch_predicates
    _INSTALLED = True


__all__ = [
    "_branches_for_requirement",
    "_contains_term",
    "_scoped_branch_predicates",
    "install_requirement_branch_scope_contract",
]
