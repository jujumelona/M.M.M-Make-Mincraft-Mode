from __future__ import annotations

"""Minecraft knowledge routing with a lossless authored-requirement authority.

The static Minecraft/Fabric taxonomy remains useful for deterministic routing, but it
is not allowed to define the user's scope.  Every authored semantic clause is preserved
by EvidenceRequestCatalog before keyword-derived knowledge routes are attached.
"""

from collections.abc import Mapping
from typing import Any

from . import minecraft_knowledge_nodes as _nodes
from .evidence_first_planning import (
    _hash_without as _catalog_hash_without,
    build_request_catalog,
)

# Preserve the historical module surface while moving the static taxonomy into a
# data/route owner.  This also keeps internal helpers available to existing callers
# during the canonicalization migration without installing runtime monkey patches.
for _name in dir(_nodes):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_nodes, _name)

_base_compile_minecraft_knowledge_plan = _nodes.compile_minecraft_knowledge_plan
_base_validate_plan = _nodes.validate_plan
_base_compact_plan = _nodes.compact_plan


def _authored_requirement_lifecycle(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw = catalog.get("requirements", [])
    if not isinstance(raw, list):
        return records
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        records.append(
            {
                "requirement_id": str(item.get("requirement_id", "")),
                "statement": str(item.get("statement", "")),
                "mandatory": bool(item.get("mandatory", True)),
                "source_span": dict(item.get("source_span", {}))
                if isinstance(item.get("source_span"), Mapping)
                else {},
                "capability": str(item.get("capability", "")),
                "gameplay_capabilities": list(item.get("gameplay_capabilities", []))
                if isinstance(item.get("gameplay_capabilities"), list)
                else [],
                "implementation_capabilities": list(
                    item.get("implementation_capabilities", [])
                )
                if isinstance(item.get("implementation_capabilities"), list)
                else [],
                "semantic_status": str(item.get("semantic_status", "RESOLVED")),
                # Pre-design has not yet earned the right to call an authored
                # requirement implemented.  Unknown taxonomy terms are retained here
                # rather than disappearing from a keyword-derived feature set.
                "state": "PRESERVED_FOR_RESEARCH",
            }
        )
    return records


def compile_minecraft_knowledge_plan(
    prompt: str,
    game_design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile routing hints without allowing them to become scope authority."""

    design = dict(game_design or {})
    catalog = build_request_catalog(prompt, design, router=None)
    plan = dict(_base_compile_minecraft_knowledge_plan(prompt, design))
    plan["authored_request_catalog"] = catalog
    plan["authored_requirements"] = _authored_requirement_lifecycle(catalog)

    policy = dict(plan.get("policy", {}))
    policy.update(
        {
            "request_completeness_owner": "evidence_request_catalog",
            "feature_detection_role": "routing_hint_only",
            "authored_requirements_may_be_dropped": False,
            "unknown_authored_requirements": "preserve_for_research",
        }
    )
    plan["policy"] = policy
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = _nodes._sha({**plan, "plan_sha256": ""})
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Validate both technical routes and the immutable authored-scope ledger."""

    _base_validate_plan(plan)
    catalog = plan.get("authored_request_catalog")
    authored = plan.get("authored_requirements")
    policy = plan.get("policy")
    if not isinstance(catalog, Mapping):
        raise ValueError("Minecraft knowledge plan has no authored request catalog.")
    if catalog.get("catalog_sha256") != _catalog_hash_without(
        catalog, "catalog_sha256"
    ):
        raise ValueError("Minecraft authored request catalog hash mismatch.")
    catalog_requirements = catalog.get("requirements")
    if not isinstance(catalog_requirements, list) or not catalog_requirements:
        raise ValueError("Minecraft authored request catalog has no requirements.")
    if not isinstance(authored, list) or len(authored) != len(catalog_requirements):
        raise ValueError("Minecraft authored requirement lifecycle is incomplete.")
    catalog_ids = [
        str(item.get("requirement_id", ""))
        for item in catalog_requirements
        if isinstance(item, Mapping)
    ]
    authored_ids = [
        str(item.get("requirement_id", ""))
        for item in authored
        if isinstance(item, Mapping)
    ]
    if catalog_ids != authored_ids or len(authored_ids) != len(authored):
        raise ValueError("Minecraft authored requirement lifecycle lost source requirements.")
    if any(
        not isinstance(item, Mapping)
        or item.get("state") != "PRESERVED_FOR_RESEARCH"
        or not str(item.get("statement", "")).strip()
        for item in authored
    ):
        raise ValueError("Minecraft authored requirement lifecycle has invalid state.")
    if not isinstance(policy, Mapping):
        raise ValueError("Minecraft knowledge policy is missing.")
    if policy.get("request_completeness_owner") != "evidence_request_catalog":
        raise ValueError("Minecraft request completeness authority is invalid.")
    if policy.get("feature_detection_role") != "routing_hint_only":
        raise ValueError("Minecraft feature detection must remain routing-only.")
    if policy.get("authored_requirements_may_be_dropped") is not False:
        raise ValueError("Minecraft authored requirements may not be dropped.")


def compact_plan(
    plan: Mapping[str, Any],
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep authored scope visible even in the bounded design-facing plan."""

    value = dict(_base_compact_plan(plan, coverage))
    value["authored_requirements"] = [
        dict(item)
        for item in plan.get("authored_requirements", [])
        if isinstance(item, Mapping)
    ]
    catalog = plan.get("authored_request_catalog")
    if isinstance(catalog, Mapping):
        value["authored_request_catalog_sha256"] = str(
            catalog.get("catalog_sha256", "")
        )
    return value


__all__ = sorted(
    set(getattr(_nodes, "__all__", ()))
    | {
        "compile_minecraft_knowledge_plan",
        "compact_plan",
        "validate_plan",
    }
)
