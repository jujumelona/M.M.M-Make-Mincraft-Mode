from __future__ import annotations

"""Deterministic Minecraft content offload for small-model generation turns.

The model supplies only bounded semantic module intent. Project discovery, mod/package
identity, version-bound generation and all file writes stay with the existing reviewed
host generator. This deliberately reuses ``extended_content_generator`` instead of
creating a second resource/datagen implementation.
"""

import json
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_TOOL_NAME = "apply_minecraft_content_spec"
_PARTIAL_EDIT_TOOL = "apply_source_edit"
_MAX_MODULES = 64
_MAX_RECEIPT_PATHS = 96
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_DYNAMIC_SKILLS = {
    _TOOL_NAME: "generate-datagen",
    _PARTIAL_EDIT_TOOL: "patch-existing-project",
}


def _tool_schema(extended_module: Any) -> dict[str, Any]:
    supported = sorted(str(kind) for kind in extended_module._SUPPORTED)
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Generate standard Minecraft/Fabric content from compact semantic module "
                "intent. Prefer this over emitting routine registry/resource boilerplate "
                "or whole JSON/Java files for supported item, block, tool, weapon, armor, "
                "food, crop, machine, effect, enchantment, command, recipe, advancement "
                "and loot modules. The host discovers the bound project, mod id and Java "
                "package, applies the pinned platform generator, and writes canonical "
                "artifacts. Do not provide paths, versions, package names or file contents."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["modules"],
                "properties": {
                    "modules": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_MODULES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "kind"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "pattern": _ID_RE.pattern,
                                    "minLength": 2,
                                    "maxLength": 64,
                                },
                                "kind": {"type": "string", "enum": supported},
                                "config": {"type": "object"},
                                "depends_on": {
                                    "type": "array",
                                    "maxItems": _MAX_MODULES,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "pattern": _ID_RE.pattern,
                                        "minLength": 2,
                                        "maxLength": 64,
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def _compile_modules(extended_module: Any, payload: Mapping[str, Any]) -> tuple[Any, ...]:
    extra = set(payload) - {"modules"}
    if extra:
        raise ValueError(
            "Minecraft content spec accepts only modules; host-owned project/version "
            f"fields are forbidden: {sorted(extra)}"
        )
    raw_modules = payload.get("modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise ValueError("modules must be a non-empty list")
    if len(raw_modules) > _MAX_MODULES:
        raise ValueError(f"modules exceeds the {_MAX_MODULES}-module batch limit")

    supported = frozenset(str(kind) for kind in extended_module._SUPPORTED)
    seen: set[str] = set()
    compiled: list[Any] = []
    for raw in raw_modules:
        if not isinstance(raw, Mapping):
            raise ValueError("Each Minecraft content module must be an object")
        extra_fields = set(raw) - {"id", "kind", "config", "depends_on"}
        if extra_fields:
            raise ValueError(
                "Minecraft content modules accept only id, kind, config and depends_on; "
                f"unknown fields: {sorted(extra_fields)}"
            )
        module_id = raw.get("id")
        kind = raw.get("kind")
        if not isinstance(module_id, str) or not _ID_RE.fullmatch(module_id):
            raise ValueError(f"Invalid strict Minecraft module id: {module_id!r}")
        if module_id in seen:
            raise ValueError(f"Duplicate Minecraft module id: {module_id}")
        seen.add(module_id)
        if not isinstance(kind, str) or kind not in supported:
            raise ValueError(f"Unsupported deterministic Minecraft module kind: {kind!r}")

        config = raw.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"Module config must be an object: {module_id}")
        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise ValueError(f"depends_on must be a list: {module_id}")
        if len(depends_on) > _MAX_MODULES:
            raise ValueError(f"depends_on exceeds the {_MAX_MODULES}-entry limit: {module_id}")
        if any(not isinstance(item, str) or not _ID_RE.fullmatch(item) for item in depends_on):
            raise ValueError(f"Invalid dependency id in module: {module_id}")
        if len(set(depends_on)) != len(depends_on):
            raise ValueError(f"Duplicate dependency in module: {module_id}")
        if module_id in depends_on:
            raise ValueError(f"Module cannot depend on itself: {module_id}")

        module = extended_module.ProductionModule(
            module_id=module_id,
            kind=kind,
            config=dict(config),
            depends_on=tuple(depends_on),
        )
        module.validate()
        compiled.append(module)
    return tuple(compiled)


def _relative_receipt_paths(
    project_root: Path,
    values: Any,
) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    root = project_root.resolve()
    result: list[str] = []
    for raw in values[:_MAX_RECEIPT_PATHS]:
        if not isinstance(raw, (str, Path)):
            continue
        path = Path(raw)
        if path.is_absolute():
            try:
                value = path.resolve(strict=False).relative_to(root).as_posix()
            except ValueError:
                continue
        else:
            value = path.as_posix()
        if value and value not in result:
            result.append(value)
    return result


def _compact_record(
    record: Mapping[str, Any],
    *,
    project_root: Path,
    requested_module_ids: Sequence[str],
) -> dict[str, Any]:
    generated = record.get("modules")
    module_ids = (
        [str(item) for item in generated[:_MAX_MODULES]]
        if isinstance(generated, list)
        else list(requested_module_ids)
    )
    gates = record.get("required_gates")
    required_gates = (
        [str(item)[:160] for item in gates[:16]]
        if isinstance(gates, list)
        else []
    )
    paths = _relative_receipt_paths(
        project_root,
        record.get("touched_paths", record.get("files", [])),
    )
    return {
        "schema_version": "mmm/deterministic-content-receipt-v1",
        "status": str(record.get("status", "UNKNOWN")),
        "generator": "extended_content_generator",
        "module_ids": module_ids,
        "module_count": len(module_ids),
        "catalog_module_count": int(record.get("catalog_module_count", len(module_ids)) or 0),
        "touched_paths": paths,
        "touched_path_count": len(paths),
        "required_gates": required_gates,
    }


def _execute(
    runtime_module: Any,
    extended_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    from .project_edit import inspect_fabric_project

    modules = _compile_modules(extended_module, payload)
    project_root, _project_argument = runtime_module._discover_model_project_root(
        workspace_root
    )
    info = inspect_fabric_project(project_root)
    record = extended_module.generate_extended_content(
        project_root=info.root,
        mod_id=info.mod_id,
        package_name=info.package_name,
        modules=modules,
    )
    if not isinstance(record, Mapping):
        raise runtime_module.AgentToolRuntimeError(
            "Deterministic Minecraft generator returned a non-object receipt"
        )
    return _compact_record(
        record,
        project_root=info.root,
        requested_module_ids=[module.module_id for module in modules],
    )


def _schema_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _install_runtime(runtime_module: Any, extended_module: Any) -> None:
    runtime_cls = runtime_module.AgentToolRuntime
    current_schemas = runtime_cls.tool_schemas
    if not getattr(current_schemas, "_mmm_deterministic_content_v1", False):

        @wraps(current_schemas)
        def tool_schemas(self: Any, stage: str):
            schemas = current_schemas(self, stage)
            selected = self._stage(stage)
            if selected != "generation" or any(
                _schema_name(item) == _TOOL_NAME
                for item in schemas
                if isinstance(item, Mapping)
            ):
                return schemas
            content_tool = _tool_schema(extended_module)
            result = (*schemas, content_tool)
            with self._lock:
                self._schema_cache[selected] = result
                self._allowed_tool_cache[selected] = frozenset(
                    _schema_name(item)
                    for item in result
                    if isinstance(item, Mapping) and _schema_name(item)
                )
            return result

        tool_schemas._mmm_deterministic_content_v1 = True  # type: ignore[attr-defined]
        tool_schemas.__wrapped__ = current_schemas  # type: ignore[attr-defined]
        runtime_cls.tool_schemas = tool_schemas

    current_call = runtime_cls._call
    if not getattr(current_call, "_mmm_deterministic_content_v1", False):

        @wraps(current_call)
        def call(
            self: Any,
            stage: str,
            name: str,
            arguments: Mapping[str, Any] | None,
            *,
            external_server_ids: frozenset[str] | None,
        ):
            selected = self._stage(stage)
            tool_name = str(name).strip()
            if selected != "generation" or tool_name != _TOOL_NAME:
                return current_call(
                    self,
                    stage,
                    name,
                    arguments,
                    external_server_ids=external_server_ids,
                )
            self.tool_schemas(selected)
            with self._lock:
                allowed = self._allowed_tool_cache.get(selected, frozenset())
            if tool_name not in allowed:
                raise runtime_module.AgentToolRuntimeError(
                    f"Tool {tool_name!r} is not exposed in stage {selected!r}."
                )
            try:
                payload = dict(arguments or {})
                result = _execute(
                    runtime_module,
                    extended_module,
                    self.workspace_root,
                    payload,
                )
            except runtime_module.AgentToolRuntimeError:
                raise
            except Exception as exc:
                raise runtime_module.AgentToolRuntimeError(
                    runtime_module._redact_text(str(exc))
                ) from exc
            return runtime_module._bounded_result(result)

        call._mmm_deterministic_content_v1 = True  # type: ignore[attr-defined]
        call.__wrapped__ = current_call  # type: ignore[attr-defined]
        runtime_cls._call = call


def _role_dynamic_tools(capability_module: Any, stage: str, model_role: str) -> dict[str, str]:
    if str(stage).strip().lower() != "generation":
        return {}
    policy = capability_module._role_policy_snapshot(stage, model_role)
    assigned = policy.skills
    if not assigned:
        return dict(_DYNAMIC_SKILLS)
    return {
        tool: skill
        for tool, skill in _DYNAMIC_SKILLS.items()
        if skill in assigned
    }


def _install_capability_policy(capability_module: Any) -> None:
    current_filter = capability_module.filter_tool_schemas_for_role
    if not getattr(current_filter, "_mmm_small_model_host_tools_v1", False):

        @wraps(current_filter)
        def filter_tool_schemas_for_role(
            stage: str,
            model_role: str,
            tool_schemas: Sequence[Mapping[str, Any]],
        ):
            raw = tuple(tool_schemas)
            filtered = tuple(current_filter(stage, model_role, raw))
            dynamic = _role_dynamic_tools(capability_module, stage, model_role)
            if not dynamic:
                return filtered
            raw_names = {_schema_name(schema) for schema in raw}
            allowed = {_schema_name(schema) for schema in filtered}
            allowed.update(name for name in dynamic if name in raw_names)
            return tuple(
                schema for schema in raw if _schema_name(schema) in allowed
            )

        filter_tool_schemas_for_role._mmm_small_model_host_tools_v1 = True  # type: ignore[attr-defined]
        filter_tool_schemas_for_role.__wrapped__ = current_filter  # type: ignore[attr-defined]
        capability_module.filter_tool_schemas_for_role = filter_tool_schemas_for_role

    current_skills_for_tool = capability_module.skills_for_tool
    if not getattr(current_skills_for_tool, "_mmm_small_model_host_tools_v1", False):

        @wraps(current_skills_for_tool)
        def skills_for_tool(
            stage: str,
            tool: str,
            *,
            model_role: str = "",
        ):
            selected = str(tool).strip()
            dynamic = _role_dynamic_tools(capability_module, stage, model_role)
            skill = dynamic.get(selected)
            if skill:
                return (skill,)
            return current_skills_for_tool(
                stage,
                tool,
                model_role=model_role,
            )

        skills_for_tool._mmm_small_model_host_tools_v1 = True  # type: ignore[attr-defined]
        skills_for_tool.__wrapped__ = current_skills_for_tool  # type: ignore[attr-defined]
        capability_module.skills_for_tool = skills_for_tool

    current_context = capability_module.build_agent_capability_context
    if not getattr(current_context, "_mmm_small_model_host_tools_v1", False):

        @wraps(current_context)
        def build_agent_capability_context(
            stage: str,
            tool_schemas: Sequence[Mapping[str, Any]],
            *,
            model_role: str = "",
        ) -> str:
            text = current_context(
                stage,
                tool_schemas,
                model_role=model_role,
            )
            dynamic = _role_dynamic_tools(capability_module, stage, model_role)
            exposed = {_schema_name(schema) for schema in tool_schemas}
            selected = {
                tool: skill
                for tool, skill in dynamic.items()
                if tool in exposed
            }
            prefix = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
            if not selected or not text.startswith(prefix):
                return text
            try:
                payload = json.loads(text[len(prefix) :])
            except json.JSONDecodeError:
                return text
            skills = payload.get("eligible_skills")
            if not isinstance(skills, list):
                return text
            for skill_entry in skills:
                if not isinstance(skill_entry, dict):
                    continue
                skill_name = str(skill_entry.get("name", ""))
                additions = [
                    tool for tool, skill in selected.items() if skill == skill_name
                ]
                if not additions:
                    continue
                model_tools = [
                    str(item)
                    for item in skill_entry.get("model_tools", [])
                    if str(item)
                ]
                for tool in additions:
                    if tool not in model_tools:
                        model_tools.append(tool)
                skill_entry["model_tools"] = model_tools
                host_tools = [
                    str(item)
                    for item in skill_entry.get("host_owned_tools", [])
                    if str(item) not in additions
                ]
                skill_entry["host_owned_tools"] = host_tools
            payload["small_model_host_tools"] = {
                tool: {
                    "skill": skill,
                    "policy": "semantic intent only; host owns deterministic mutation",
                }
                for tool, skill in sorted(selected.items())
            }
            return prefix + json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        build_agent_capability_context._mmm_small_model_host_tools_v1 = True  # type: ignore[attr-defined]
        build_agent_capability_context.__wrapped__ = current_context  # type: ignore[attr-defined]
        capability_module.build_agent_capability_context = build_agent_capability_context


def _install_transition(registry_module: Any) -> None:
    transition = registry_module.TransitionSpec(
        preconditions=frozenset({"project_observed"}),
        effects=frozenset({"project_changed", "source_generated", "generated"}),
        cost=1,
    )
    existing = registry_module.TRANSITIONS.get(_TOOL_NAME)
    if existing is not None and existing != transition:
        raise RuntimeError(
            f"Conflicting reviewed causal transition for {_TOOL_NAME!r}"
        )
    registry_module.TRANSITIONS.setdefault(_TOOL_NAME, transition)


def install(extended_module: Any | None = None) -> None:
    from . import (
        agent_capability_context,
        agent_tool_runtime,
        extended_content_generator,
        tool_transition_registry,
    )

    selected_extended = extended_module or extended_content_generator
    _install_transition(tool_transition_registry)
    _install_runtime(agent_tool_runtime, selected_extended)
    _install_capability_policy(agent_capability_context)


__all__ = [
    "_compact_record",
    "_compile_modules",
    "_execute",
    "_tool_schema",
    "install",
]
