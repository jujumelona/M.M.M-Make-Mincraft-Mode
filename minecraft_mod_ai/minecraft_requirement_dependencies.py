from __future__ import annotations

"""Compile cross-requirement prerequisites from the host Minecraft feature model.

The semantic model never emits dependency edges.  Once all authored capabilities are
known, the host binds only prerequisite capabilities that are actually present in the
same request.  This is the variability-resolution step of the Minecraft feature model:
absent features are never invented merely because another template can consume them.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .minecraft_template_catalog import selected_predecessor_capabilities


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


def _assert_acyclic(dependencies: Mapping[str, Sequence[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(requirement_id: str) -> None:
        if requirement_id in visited:
            return
        if requirement_id in visiting:
            raise ValueError(
                "host Minecraft feature dependencies produced a requirement cycle"
            )
        visiting.add(requirement_id)
        for dependency in dependencies.get(requirement_id, ()):
            visit(str(dependency))
        visiting.remove(requirement_id)
        visited.add(requirement_id)

    for requirement_id in dependencies:
        visit(requirement_id)


def bind_selected_feature_dependencies(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a catalog whose dependency/unlock fields are host-derived.

    Existing explicit host-approved dependency refs are preserved.  Feature-model
    predecessors are added only when their capability was independently selected from
    authored request text.  All matching requirement providers are bound so duplicate
    authored leaves cannot silently drop a prerequisite.
    """

    raw_requirements = catalog.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        return dict(catalog)
    requirements = [
        dict(item) if isinstance(item, Mapping) else item
        for item in raw_requirements
    ]
    if any(not isinstance(item, dict) for item in requirements):
        raise ValueError("host feature dependency binding requires requirement objects")

    selected_capabilities = tuple(
        str(item.get("capability") or "").strip().casefold()
        for item in requirements
        if str(item.get("capability") or "").strip()
    )
    requirement_ids_by_capability: dict[str, list[str]] = {}
    for item in requirements:
        requirement_id = str(item.get("requirement_id") or "").strip()
        capability = str(item.get("capability") or "").strip().casefold()
        if not requirement_id or not capability:
            raise ValueError(
                "host feature dependency binding requires requirement_id and capability"
            )
        requirement_ids_by_capability.setdefault(capability, []).append(requirement_id)

    dependency_map: dict[str, tuple[str, ...]] = {}
    dependency_capability_map: dict[str, tuple[str, ...]] = {}
    for item in requirements:
        requirement_id = str(item["requirement_id"])
        capability = str(item["capability"]).casefold()
        predecessor_capabilities = selected_predecessor_capabilities(
            capability,
            selected_capabilities,
        )
        derived_refs = tuple(
            dependency_id
            for predecessor in predecessor_capabilities
            for dependency_id in requirement_ids_by_capability.get(predecessor, ())
            if dependency_id != requirement_id
        )
        dependencies = tuple(
            dict.fromkeys((*_strings(item.get("depends_on")), *derived_refs))
        )
        dependency_map[requirement_id] = dependencies
        dependency_capability_map[requirement_id] = tuple(
            dict.fromkeys(predecessor_capabilities)
        )

    known_ids = set(dependency_map)
    for requirement_id, dependencies in dependency_map.items():
        unknown = [dependency for dependency in dependencies if dependency not in known_ids]
        if unknown:
            raise ValueError(
                f"requirement {requirement_id} has unknown dependency refs: {unknown}"
            )
        if requirement_id in dependencies:
            raise ValueError(f"requirement {requirement_id} may not depend on itself")
    _assert_acyclic(dependency_map)

    edges: list[dict[str, str]] = []
    for item in requirements:
        requirement_id = str(item["requirement_id"])
        dependencies = dependency_map[requirement_id]
        predecessor_capabilities = dependency_capability_map[requirement_id]
        item["depends_on"] = list(dependencies)
        unlock = dict(item.get("unlock_policy") or {})
        unlock["required_requirement_refs"] = list(dependencies)
        unlock["required_capabilities"] = list(predecessor_capabilities)
        unlock.setdefault("optional_requirement_refs", [])
        unlock.setdefault("optional_capabilities", [])
        unlock["policy"] = "host_feature_model_and_authored_state_only"
        item["unlock_policy"] = unlock
        item["dependency_provenance"] = {
            "owner": "host_minecraft_feature_model",
            "selected_predecessor_capabilities": list(predecessor_capabilities),
            "required_requirement_refs": list(dependencies),
        }
        edges.extend(
            {
                "from_requirement_ref": dependency,
                "to_requirement_ref": requirement_id,
                "kind": "host_feature_model_prerequisite",
            }
            for dependency in dependencies
        )

    result = dict(catalog)
    result["requirements"] = requirements
    graph = dict(result.get("requirement_graph") or {})
    graph["node_ids"] = [str(item["requirement_id"]) for item in requirements]
    graph["edges"] = edges
    result["requirement_graph"] = graph
    return result


__all__ = ["bind_selected_feature_dependencies"]
