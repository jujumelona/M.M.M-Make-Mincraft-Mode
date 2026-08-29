from __future__ import annotations

"""Explicit requires/provides SkillBank composition.

There is no runtime installer here. The canonical pre-design pipeline calls
``compose_research_skillbank`` after procedural skills have been compiled. Dependency
resolution uses only explicit requires/provides edges; unresolved or cyclic skills are
blocked instead of being inferred from lexical similarity.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_COMPOSED_SKILLS = 12


def _bounded_strings(value: Any, *, limit: int = 8, chars: int = 192) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for raw in value[:limit]:
        text = " ".join(str(raw).split())[:chars]
        if text and text not in result:
            result.append(text)
    return result


def _capability(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _skill_confidence(skill: Mapping[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(skill.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _compose_skills(
    query: str,
    available: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    *,
    limit: int = _MAX_COMPOSED_SKILLS,
) -> dict[str, Any]:
    """Resolve explicit skill dependencies and return only executable ordered skills."""

    by_id = {
        str(skill.get("skill_id", "")): dict(skill)
        for skill in available
        if str(skill.get("skill_id", ""))
    }
    selected: dict[str, dict[str, Any]] = {}
    for skill in seeds:
        identity = str(skill.get("skill_id", ""))
        if identity and identity in by_id:
            selected.setdefault(identity, by_id[identity])

    providers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for skill in by_id.values():
        for capability in _bounded_strings(skill.get("provides"), limit=8):
            normalized = _capability(capability)
            if normalized:
                providers[normalized].append(skill)
    for rows in providers.values():
        rows.sort(
            key=lambda skill: (
                -_skill_confidence(skill),
                str(skill.get("skill_id", "")),
            )
        )

    unresolved: list[dict[str, str]] = []
    changed = True
    while changed and len(selected) < max(1, int(limit)):
        changed = False
        for consumer_id, skill in list(selected.items()):
            for requirement in _bounded_strings(skill.get("requires"), limit=8):
                capability = _capability(requirement)
                if not capability:
                    continue
                already_provided = any(
                    capability
                    in {
                        _capability(item)
                        for item in _bounded_strings(candidate.get("provides"), limit=8)
                    }
                    for candidate in selected.values()
                )
                if already_provided:
                    continue
                candidates = providers.get(capability, ())
                if not candidates:
                    marker = {"skill_id": consumer_id, "requirement": requirement}
                    if marker not in unresolved:
                        unresolved.append(marker)
                    continue
                provider = candidates[0]
                provider_id = str(provider.get("skill_id", ""))
                if provider_id not in selected and len(selected) < max(1, int(limit)):
                    selected[provider_id] = dict(provider)
                    changed = True

    edges: list[dict[str, str]] = []
    missing_ids = {item["skill_id"] for item in unresolved}
    for consumer_id, skill in selected.items():
        for requirement in _bounded_strings(skill.get("requires"), limit=8):
            capability = _capability(requirement)
            candidates = [
                candidate
                for candidate in selected.values()
                if capability
                and capability
                in {
                    _capability(item)
                    for item in _bounded_strings(candidate.get("provides"), limit=8)
                }
            ]
            candidates.sort(
                key=lambda candidate: (
                    -_skill_confidence(candidate),
                    str(candidate.get("skill_id", "")),
                )
            )
            if not candidates:
                marker = {"skill_id": consumer_id, "requirement": requirement}
                if marker not in unresolved:
                    unresolved.append(marker)
                missing_ids.add(consumer_id)
                continue
            provider_id = str(candidates[0].get("skill_id", ""))
            edge = {
                "provider": provider_id,
                "consumer": consumer_id,
                "requirement": requirement,
            }
            if edge not in edges:
                edges.append(edge)

    adjacency: dict[str, list[str]] = {identity: [] for identity in selected}
    for edge in edges:
        adjacency.setdefault(edge["provider"], []).append(edge["consumer"])

    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    cycle_groups: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adjacency.get(node, ()):
            if target not in indices:
                strongconnect(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        component.sort()
        self_loop = len(component) == 1 and component[0] in adjacency.get(component[0], ())
        if len(component) > 1 or self_loop:
            cycle_groups.append(component)

    for identity in sorted(selected):
        if identity not in indices:
            strongconnect(identity)

    blocked = set(missing_ids)
    for group in cycle_groups:
        blocked.update(group)
    propagated = True
    while propagated:
        propagated = False
        for edge in edges:
            if edge["provider"] in blocked and edge["consumer"] not in blocked:
                blocked.add(edge["consumer"])
                propagated = True

    indegree = {identity: 0 for identity in selected if identity not in blocked}
    for edge in edges:
        if edge["provider"] in indegree and edge["consumer"] in indegree:
            indegree[edge["consumer"]] += 1
    ready = sorted(identity for identity, degree in indegree.items() if degree == 0)
    ordered_ids: list[str] = []
    while ready:
        identity = ready.pop(0)
        ordered_ids.append(identity)
        for target in sorted(adjacency.get(identity, ())):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()

    return {
        "schema_version": "mmm/ordered-skill-composition-v1",
        "composition_policy": "explicit_requires_provides_only",
        "query": query[:512],
        "ordered_skills": [selected[identity] for identity in ordered_ids],
        "dependency_edges": sorted(
            edges,
            key=lambda edge: (edge["consumer"], edge["provider"], edge["requirement"]),
        ),
        "unresolved_requirements": sorted(
            unresolved,
            key=lambda item: (item["skill_id"], item["requirement"]),
        ),
        "cycles": sorted(cycle_groups),
        "blocked_skill_ids": sorted(blocked),
    }


def compose_research_skillbank(
    router: Any,
    prompt: str,
    research: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the attached SkillBank explicitly; no research function is monkeypatched."""

    from . import external_procedural_skill_contract as skills

    value = dict(research)
    bank = value.get("procedural_skillbank")
    if not isinstance(bank, Mapping):
        return value

    bank_value = dict(bank)
    seeds = [
        dict(skill)
        for skill in bank_value.get("retrieved_skills", ())
        if isinstance(skill, Mapping)
    ]
    available = [
        dict(skill)
        for skill in bank_value.get("skills", ())
        if isinstance(skill, Mapping)
    ]
    path = skills._skillbank_path(router)
    if path is not None:
        available.extend(skills._load_persistent_skills(path))
    dedup = {
        str(skill.get("skill_id", "")): skill
        for skill in available
        if str(skill.get("skill_id", ""))
    }
    composition = _compose_skills(
        prompt,
        list(dedup.values()),
        seeds,
        limit=_MAX_COMPOSED_SKILLS,
    )
    bank_value["flat_retrieved_skills"] = seeds
    bank_value["retrieved_skills"] = composition["ordered_skills"]
    bank_value["skill_composition"] = composition
    value["procedural_skillbank"] = bank_value

    method = dict(value.get("method", {})) if isinstance(value.get("method"), Mapping) else {}
    method["skill_composition"] = (
        "explicit requires/provides DAG; unresolved/cyclic procedures are blocked"
    )
    value["method"] = method
    return value


__all__ = ["_compose_skills", "compose_research_skillbank"]
