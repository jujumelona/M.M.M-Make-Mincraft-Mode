from __future__ import annotations

"""Reconcile coder mutation authority with the canonical implementation contract.

The implementation contract distinguishes existing targets (modify) from host-reserved
targets (create-or-modify).  The task capsule must preserve that distinction: existing
owned files are writable, but creation is authorized only for host-reserved destinations.
Deletion is never an evidence-task coder operation.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

_MARKER = "_mmm_coder_mutation_authority_v1"
_EXISTING_STATUSES = frozenset({"existing", "reuse", "modify", "host_existing"})
_CREATE_OPERATIONS = frozenset(
    {
        "create",
        "create_file",
        "create_class",
        "create_type",
        "create_java_type",
        "create_java_class",
        "write",
        "write_file",
    }
)
_DELETE_OPERATIONS = frozenset({"delete", "delete_file", "remove", "remove_file"})
_MODEL_CREATE_OPERATIONS = frozenset({"create", "create_file", "create_java_type"})
_MODEL_DELETE_OPERATIONS = frozenset({"delete", "delete_file"})


def _primary_candidate(target_module: Any, task: Mapping[str, Any], task_id: str) -> tuple[str, str] | None:
    bindings = target_module._matching_bindings(task, task_id)
    candidates = target_module._binding_symbol_candidates(bindings)
    return candidates[0] if len(candidates) == 1 else None


def _parsed_anchors(target_module: Any, task: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = task.get("owned_anchors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    result: list[Any] = []
    for item in raw:
        anchor = target_module._task_anchor(item)
        if anchor is not None and anchor not in result:
            result.append(anchor)
    return tuple(result)


def _module_with_reserved_primary(target_module: Any, module: Any) -> tuple[Any, tuple[Any, ...]] | None:
    config = getattr(module, "config", None)
    if not isinstance(config, Mapping):
        return None
    task = config.get("evidence_task")
    if not isinstance(task, Mapping):
        return None
    task_id = str(task.get("task_id") or "").strip()
    candidate = _primary_candidate(target_module, task, task_id)
    if candidate is None:
        return None
    primary_path, _primary_symbol = candidate
    anchors = _parsed_anchors(target_module, task)
    primary = next((anchor for anchor in anchors if anchor.path == primary_path), None)
    if primary is None or primary.status.casefold() not in _EXISTING_STATUSES:
        return None

    patched_task = copy.deepcopy(dict(task))
    patched_anchors: list[Any] = []
    for raw_anchor in patched_task.get("owned_anchors", ()):
        if not isinstance(raw_anchor, Mapping):
            patched_anchors.append(raw_anchor)
            continue
        current = dict(raw_anchor)
        parsed = target_module._task_anchor(current)
        if parsed is not None and parsed.path == primary_path:
            current["status"] = "host_reserved"
        patched_anchors.append(current)
    patched_task["owned_anchors"] = patched_anchors

    patched_config = dict(config)
    patched_config["evidence_task"] = patched_task
    proxy = SimpleNamespace(
        module_id=getattr(module, "module_id", ""),
        kind=getattr(module, "kind", ""),
        config=patched_config,
        depends_on=getattr(module, "depends_on", ()),
        required_gates=getattr(module, "required_gates", ()),
    )
    return proxy, anchors


def _restore_capsule_authority(target_module: Any, capsule: Any, anchors: tuple[Any, ...]) -> Any:
    digest_input = {
        "task_id": capsule.task_id,
        "module_kind": capsule.module_kind,
        "primary_path": capsule.primary_path,
        "primary_symbol": capsule.primary_symbol,
        "anchors": [anchor.to_dict() for anchor in anchors],
        "reuse_action": capsule.reuse_action,
        "required_gates": capsule.required_gates,
        "task_sha256": capsule.task_sha256,
    }
    return replace(
        capsule,
        anchors=anchors,
        capsule_sha256=target_module._sha256(digest_input),
    )


def install(target_module: Any | None = None) -> None:
    if target_module is None:
        from . import small_model_task_capsule_contract as target_module

    if getattr(target_module, _MARKER, False):
        return

    original_compile = target_module.compile_task_capsule
    original_bind = target_module.bind_source_edit_arguments
    original_narrow = target_module.narrow_source_edit_schema

    def compile_task_capsule(module: Any):
        try:
            return original_compile(module)
        except target_module.TaskCapsuleContractError as exc:
            if "TASK_CAPSULE_PRIMARY_NOT_RESERVED" not in str(exc):
                raise
            reconciled = _module_with_reserved_primary(target_module, module)
            if reconciled is None:
                raise
            proxy, anchors = reconciled
            capsule = original_compile(proxy)
            if capsule is None:
                return None
            return _restore_capsule_authority(target_module, capsule, anchors)

    def bind_source_edit_arguments(arguments: Mapping[str, Any], capsule: Any) -> dict[str, Any]:
        bound = original_bind(arguments, capsule)
        operation = str(bound.get("operation") or "").strip().casefold()
        path = str(bound.get("path") or "").strip()
        if operation in _DELETE_OPERATIONS:
            raise target_module.TaskCapsuleContractError(
                "TASK_MUTATION_DELETE_FORBIDDEN: evidence-task coder may not delete files."
            )
        if operation in _CREATE_OPERATIONS and path not in capsule.creatable_paths:
            raise target_module.TaskCapsuleContractError(
                "TASK_MUTATION_CREATE_NOT_RESERVED: creation requires a host_reserved target; "
                f"path={path!r}, creatable={list(capsule.creatable_paths)!r}."
            )
        return bound

    def narrow_source_edit_schema(schema: Any, capsule: Any) -> Any:
        narrowed = original_narrow(schema, capsule)
        if not isinstance(narrowed, Mapping):
            return narrowed
        result = copy.deepcopy(dict(narrowed))
        function = result.get("function")
        parameters = function.get("parameters") if isinstance(function, dict) else None
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        operation = properties.get("operation") if isinstance(properties, dict) else None
        enum = operation.get("enum") if isinstance(operation, dict) else None
        if isinstance(enum, list):
            blocked = set(_MODEL_DELETE_OPERATIONS)
            if not capsule.creatable_paths:
                blocked.update(_MODEL_CREATE_OPERATIONS)
            operation["enum"] = [item for item in enum if str(item).casefold() not in blocked]
        return result

    compile_task_capsule.__name__ = original_compile.__name__
    bind_source_edit_arguments.__name__ = original_bind.__name__
    narrow_source_edit_schema.__name__ = original_narrow.__name__
    target_module.compile_task_capsule = compile_task_capsule
    target_module.bind_source_edit_arguments = bind_source_edit_arguments
    target_module.narrow_source_edit_schema = narrow_source_edit_schema
    setattr(target_module, _MARKER, True)


def assert_installed(target_module: Any | None = None) -> None:
    if target_module is None:
        from . import small_model_task_capsule_contract as target_module
    if not getattr(target_module, _MARKER, False):
        raise RuntimeError("coder mutation authority reconciliation is not installed")


__all__ = ["assert_installed", "install"]
