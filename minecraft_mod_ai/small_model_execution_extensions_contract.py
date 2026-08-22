from __future__ import annotations

"""Host-side execution extensions that let small models do less, safer work.

The contract adds two complementary capabilities without changing model weights:

* one scalar source-edit ACI whose schema/materializer is owned by
  ``source_edit_scalar_protocol_contract`` while this module owns runtime dispatch;
* explicit requires/provides SkillBank composition with fail-closed dependency
  ordering. Dependencies are never inferred from lexical similarity.
"""

import hashlib
import json
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from .source_edit_scalar_protocol_contract import (
    SOURCE_EDIT_SCHEMA as _SOURCE_EDIT_SCHEMA,
    materialize_model_source_edit as _materialize_scalar_source_edit,
)

_INSTALLED = False
_SOURCE_EDIT_TOOL = "apply_source_edit"
_MAX_COMPOSED_SKILLS = 12


def _replacement_for_edit(runtime_module: Any, item: Mapping[str, Any], path: str) -> dict[str, Any]:
    operation = str(item.get("operation", "")).strip()
    count = item.get("count", 1)
    if type(count) is not int or count < 1:
        raise runtime_module.AgentToolRuntimeError(f"Invalid exact-match count for {path}")
    if operation == "replace_exact":
        allowed = {"operation", "path", "old", "new", "count"}
        if set(item) - allowed:
            raise runtime_module.AgentToolRuntimeError(f"Unknown replace_exact fields for {path}")
        old = item.get("old")
        new = item.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise runtime_module.AgentToolRuntimeError(
                f"replace_exact requires non-empty old and text new for {path}"
            )
        return {"old": old, "new": new, "count": count}
    if operation in {"insert_before", "insert_after"}:
        allowed = {"operation", "path", "anchor", "content", "count"}
        if set(item) - allowed:
            raise runtime_module.AgentToolRuntimeError(f"Unknown {operation} fields for {path}")
        anchor = item.get("anchor")
        content = item.get("content")
        if not isinstance(anchor, str) or not anchor or not isinstance(content, str):
            raise runtime_module.AgentToolRuntimeError(
                f"{operation} requires non-empty anchor and text content for {path}"
            )
        new = content + anchor if operation == "insert_before" else anchor + content
        return {"old": anchor, "new": new, "count": count}
    raise runtime_module.AgentToolRuntimeError(f"Unsupported source edit operation: {operation!r}")


def _materialize_model_source_edit(
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Delegate scalar source-write compilation to its single canonical owner."""

    return _materialize_scalar_source_edit(
        sys.modules[__name__],
        runtime_module,
        workspace_root,
        payload,
    )


def _install_partial_source_edit(runtime_module: Any) -> None:
    current_schemas = runtime_module.AgentToolRuntime.tool_schemas
    if not getattr(current_schemas, "_mmm_scalar_source_edit_v2", False):

        @wraps(current_schemas)
        def tool_schemas(self: Any, stage: str):
            schemas = current_schemas(self, stage)
            if str(stage).strip().lower() != "generation":
                return schemas
            if any(
                item.get("function", {}).get("name") == _SOURCE_EDIT_TOOL
                for item in schemas
                if isinstance(item, Mapping)
            ):
                return schemas
            edit_tool = {
                "type": "function",
                "function": {
                    "name": _SOURCE_EDIT_TOOL,
                    "description": (
                        "Make one bounded semantic source/resource write. Use exact replacement "
                        "or insert-before/after for local edits, create_file for a new file, "
                        "replace_file for a whole existing file, and delete_file only when removal "
                        "is intended. The host owns project selection, SHA-256 preconditions and "
                        "transactional execution."
                    ),
                    "parameters": _SOURCE_EDIT_SCHEMA,
                },
            }
            result = (*schemas, edit_tool)
            with self._lock:
                self._schema_cache["generation"] = result
                self._allowed_tool_cache["generation"] = frozenset(
                    str(item["function"]["name"])
                    for item in result
                )
            return result

        tool_schemas._mmm_scalar_source_edit_v2 = True  # type: ignore[attr-defined]
        tool_schemas.__wrapped__ = current_schemas  # type: ignore[attr-defined]
        runtime_module.AgentToolRuntime.tool_schemas = tool_schemas

    current_call_scoped = runtime_module.AgentToolRuntime.call_scoped
    if not getattr(current_call_scoped, "_mmm_scalar_source_edit_v2", False):

        @wraps(current_call_scoped)
        def call_scoped(
            self: Any,
            stage: str,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            *,
            external_server_ids: Sequence[str],
        ):
            if str(stage).strip().lower() == "generation" and str(name).strip() == _SOURCE_EDIT_TOOL:
                raw_patch = _materialize_model_source_edit(
                    runtime_module,
                    self.workspace_root,
                    dict(arguments or {}),
                )
                return self.call("generation", "apply_source_patch", raw_patch)
            return current_call_scoped(
                self,
                stage,
                name,
                arguments,
                external_server_ids=external_server_ids,
            )

        call_scoped._mmm_scalar_source_edit_v2 = True  # type: ignore[attr-defined]
        call_scoped.__wrapped__ = current_call_scoped  # type: ignore[attr-defined]
        runtime_module.AgentToolRuntime.call_scoped = call_scoped


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


def _rehash_skill(skill: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(skill)
    result.pop("skill_id", None)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["skill_id"] = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return result


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
        rows.sort(key=lambda skill: (-_skill_confidence(skill), str(skill.get("skill_id", ""))))

    unresolved: list[dict[str, str]] = []
    changed = True
    while changed and len(selected) < max(1, int(limit)):
        changed = False
        for consumer_id, skill in list(selected.items()):
            for requirement in _bounded_strings(skill.get("requires"), limit=8):
                capability = _capability(requirement)
                if not capability:
                    continue
                if any(
                    capability in {_capability(item) for item in _bounded_strings(candidate.get("provides"), limit=8)}
                    for candidate in selected.values()
                ):
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
                in {_capability(item) for item in _bounded_strings(candidate.get("provides"), limit=8)}
            ]
            candidates.sort(key=lambda candidate: (-_skill_confidence(candidate), str(candidate.get("skill_id", ""))))
            if not candidates:
                marker = {"skill_id": consumer_id, "requirement": requirement}
                if marker not in unresolved:
                    unresolved.append(marker)
                missing_ids.add(consumer_id)
                continue
            provider_id = str(candidates[0].get("skill_id", ""))
            edge = {"provider": provider_id, "consumer": consumer_id, "requirement": requirement}
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


def _install_ordered_skill_composition(skills_module: Any) -> None:
    current_sanitize = skills_module._sanitize_procedure
    if not getattr(current_sanitize, "_mmm_ordered_skill_composition_v1", False):

        @wraps(current_sanitize)
        def sanitize(value: Mapping[str, Any], domain_id: str):
            skill = current_sanitize(value, domain_id)
            if skill is None:
                return None
            result = dict(skill)
            result["requires"] = _bounded_strings(value.get("requires"), limit=8)
            result["provides"] = _bounded_strings(value.get("provides"), limit=8)
            return _rehash_skill(result)

        sanitize._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
        sanitize.__wrapped__ = current_sanitize  # type: ignore[attr-defined]
        skills_module._sanitize_procedure = sanitize

    current_consolidated = skills_module._consolidated_skill
    if not getattr(current_consolidated, "_mmm_ordered_skill_composition_v1", False):

        @wraps(current_consolidated)
        def consolidated(skills: Sequence[Mapping[str, Any]]):
            skill = current_consolidated(skills)
            if skill is None:
                return None
            require_sets = [set(_bounded_strings(value.get("requires"), limit=8)) for value in skills]
            provide_sets = [set(_bounded_strings(value.get("provides"), limit=8)) for value in skills]
            common_requires = set.intersection(*require_sets) if require_sets else set()
            common_provides = set.intersection(*provide_sets) if provide_sets else set()
            result = dict(skill)
            result["requires"] = sorted(common_requires)
            result["provides"] = sorted(common_provides)
            return _rehash_skill(result)

        consolidated._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
        consolidated.__wrapped__ = current_consolidated  # type: ignore[attr-defined]
        skills_module._consolidated_skill = consolidated

    current_schema = skills_module._procedure_schema
    if not getattr(current_schema, "_mmm_ordered_skill_composition_v1", False):

        @wraps(current_schema)
        def procedure_schema():
            schema = current_schema()
            properties = schema["items"]["properties"]
            properties["requires"] = {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 192},
            }
            properties["provides"] = {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 192},
            }
            return schema

        procedure_schema._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
        procedure_schema.__wrapped__ = current_schema  # type: ignore[attr-defined]
        skills_module._procedure_schema = procedure_schema

    current_install = skills_module._install_research_skill_compiler
    if not getattr(current_install, "_mmm_ordered_skill_composition_v1", False):

        @wraps(current_install)
        def install_compiler() -> None:
            current_install()
            from . import agentic_research_game_design as research

            current_messages = research._research_messages
            if not getattr(current_messages, "_mmm_ordered_skill_composition_v1", False):

                @wraps(current_messages)
                def research_messages(**kwargs: Any):
                    messages = [dict(message) for message in current_messages(**kwargs)]
                    if len(messages) >= 2 and isinstance(messages[1].get("content"), str):
                        try:
                            payload = json.loads(str(messages[1]["content"]))
                        except json.JSONDecodeError:
                            payload = None
                        if isinstance(payload, dict):
                            payload["skill_dependency_instruction"] = (
                                "For each procedure, emit requires and provides as exact capability "
                                "labels only when the cited evidence explicitly establishes the "
                                "dependency/output. Use [] when no explicit dependency is supported; "
                                "never infer requires/provides from lexical similarity."
                            )
                            messages[1]["content"] = json.dumps(
                                payload, ensure_ascii=False, sort_keys=True
                            )
                    return messages

                research_messages._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
                research_messages.__wrapped__ = current_messages  # type: ignore[attr-defined]
                research._research_messages = research_messages

            current_collect = research.collect_pre_design_research
            if not getattr(current_collect, "_mmm_ordered_skill_composition_v1", False):

                @wraps(current_collect)
                def collect(router: Any, prompt: str, *, trace_metadata=None):
                    result = current_collect(router, prompt, trace_metadata=trace_metadata)
                    if not isinstance(result, Mapping):
                        return result
                    value = dict(result)
                    bank = value.get("procedural_skillbank")
                    if not isinstance(bank, Mapping):
                        return value
                    bank_value = dict(bank)
                    flat = [
                        dict(skill)
                        for skill in bank_value.get("retrieved_skills", ())
                        if isinstance(skill, Mapping)
                    ]
                    available = [
                        dict(skill)
                        for skill in bank_value.get("skills", ())
                        if isinstance(skill, Mapping)
                    ]
                    path = skills_module._skillbank_path(router)
                    if path is not None:
                        available.extend(skills_module._load_persistent_skills(path))
                    dedup = {
                        str(skill.get("skill_id", "")): skill
                        for skill in available
                        if str(skill.get("skill_id", ""))
                    }
                    composition = _compose_skills(
                        prompt,
                        list(dedup.values()),
                        flat,
                        limit=_MAX_COMPOSED_SKILLS,
                    )
                    bank_value["flat_retrieved_skills"] = flat
                    bank_value["retrieved_skills"] = composition["ordered_skills"]
                    bank_value["skill_composition"] = composition
                    value["procedural_skillbank"] = bank_value
                    method = dict(value.get("method", {})) if isinstance(value.get("method"), Mapping) else {}
                    method["skill_composition"] = "explicit requires/provides DAG; unresolved/cyclic procedures are blocked"
                    value["method"] = method
                    value["research_sha256"] = research._json_sha256(value)
                    return value

                collect._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
                collect.__wrapped__ = current_collect  # type: ignore[attr-defined]
                research.collect_pre_design_research = collect

            current_compact = research._compact_research_for_design
            if not getattr(current_compact, "_mmm_ordered_skill_composition_v1", False):

                @wraps(current_compact)
                def compact(research: Mapping[str, Any]) -> dict[str, Any]:
                    result = dict(current_compact(research))
                    bank = research.get("procedural_skillbank")
                    if isinstance(bank, Mapping):
                        compact_bank = dict(result.get("procedural_skillbank", {}))
                        composition = bank.get("skill_composition")
                        if isinstance(composition, Mapping):
                            compact_bank["skill_composition"] = {
                                "schema_version": composition.get("schema_version"),
                                "composition_policy": composition.get("composition_policy"),
                                "ordered_skills": list(composition.get("ordered_skills", ()))[:_MAX_COMPOSED_SKILLS],
                                "dependency_edges": list(composition.get("dependency_edges", ()))[:24],
                                "unresolved_requirements": list(composition.get("unresolved_requirements", ()))[:12],
                                "cycles": list(composition.get("cycles", ()))[:8],
                                "blocked_skill_ids": list(composition.get("blocked_skill_ids", ()))[:16],
                            }
                        result["procedural_skillbank"] = compact_bank
                    return result

                compact._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
                compact.__wrapped__ = current_compact  # type: ignore[attr-defined]
                research._compact_research_for_design = compact

        install_compiler._mmm_ordered_skill_composition_v1 = True  # type: ignore[attr-defined]
        install_compiler.__wrapped__ = current_install  # type: ignore[attr-defined]
        skills_module._install_research_skill_compiler = install_compiler


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from . import agent_tool_runtime, external_procedural_skill_contract

    _install_partial_source_edit(agent_tool_runtime)
    _install_ordered_skill_composition(external_procedural_skill_contract)
    _INSTALLED = True


__all__ = [
    "install",
    "_compose_skills",
    "_materialize_model_source_edit",
]
