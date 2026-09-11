from __future__ import annotations

"""Reconcile coder target existence with the canonical implementation contract.

The implementation contract distinguishes existing targets (modify) from host-reserved
targets (create-or-modify). The task capsule and live tool loop must preserve that
distinction: existing owned files are writable and require source localization, while
creation is authorized only for host-reserved destinations.

Creation conflicts remain enforced by the canonical progress-aware mutation authority,
and file deletion remains rejected by the staged custom-module operation validator before
anything is committed to the live project. This module only repairs the stale semantic
assumptions that cannot be expressed by those existing owners.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

_MARKER = "_mmm_coder_mutation_authority_v1"
_LOOP_MARKER = "_mmm_coder_target_existence_v1"
_MUTATION_MARKER = "_mmm_coder_creation_conflict_v1"
_EXISTING_STATUSES = frozenset({"existing", "reuse", "modify", "host_existing"})


def _primary_candidate(
    target_module: Any,
    task: Mapping[str, Any],
    task_id: str,
) -> tuple[str, str] | None:
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


def _primary_status(target_module: Any, module: Any) -> str:
    config = getattr(module, "config", None)
    if not isinstance(config, Mapping):
        return ""
    task = config.get("evidence_task")
    if not isinstance(task, Mapping):
        return ""
    task_id = str(task.get("task_id") or "").strip()
    candidate = _primary_candidate(target_module, task, task_id)
    if candidate is None:
        return ""
    primary_path, _ = candidate
    anchors = _parsed_anchors(target_module, task)
    primary = next((anchor for anchor in anchors if anchor.path == primary_path), None)
    return str(getattr(primary, "status", "") or "").strip().casefold()


def _module_with_reserved_primary(
    target_module: Any,
    module: Any,
) -> tuple[Any, tuple[Any, ...]] | None:
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


def _restore_capsule_authority(
    target_module: Any,
    capsule: Any,
    anchors: tuple[Any, ...],
) -> Any:
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


def _task_owned_statuses(loop_module: Any, task: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    anchors = task.get("owned_anchors")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes, bytearray)):
        return result
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        locator = loop_module._normalized_target_path(anchor.get("locator"))
        path = locator.partition("#")[0].strip()
        status = str(anchor.get("status") or "").strip().casefold()
        if path and status:
            result[path] = status
    return result


def _status_aware_owned_symbol_context(loop_module: Any, payload: Any) -> Any | None:
    """Resolve target existence from anchor status, never from donor reuse_action."""

    if not isinstance(payload, Mapping):
        return None
    module = payload.get("module")
    task = loop_module._evidence_task_from_module(module)
    if task is None:
        return None
    bindings = task.get("production_bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
        return None
    production_bindings = tuple(item for item in bindings if isinstance(item, Mapping))
    if not production_bindings:
        return None
    actions = {
        str(item.get("reuse_action") or "").strip().casefold()
        for item in production_bindings
        if str(item.get("reuse_action") or "").strip()
    }
    if actions != {"fresh"}:
        return None

    task_statuses = _task_owned_statuses(loop_module, task)
    for binding in production_bindings:
        anchors = binding.get("owned_anchors")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes, bytearray)):
            continue
        for anchor in anchors:
            if not isinstance(anchor, Mapping) or str(anchor.get("kind") or "") != "symbol":
                continue
            locator = loop_module._normalized_target_path(anchor.get("locator"))
            target_path, separator, target_symbol = locator.partition("#")
            target_path = target_path.strip()
            if (
                not target_path
                or target_path.startswith("/")
                or ".." in target_path.split("/")
                or not loop_module._is_workspace_file_path(target_path)
            ):
                continue
            status = str(
                anchor.get("status") or task_statuses.get(target_path) or ""
            ).strip().casefold()
            if status == "host_reserved":
                is_new_file = True
                evidence_source = "evidence_host_reserved_owned_anchor"
            elif status in _EXISTING_STATUSES:
                is_new_file = False
                evidence_source = "evidence_existing_owned_anchor"
            else:
                continue
            return loop_module.TargetMutationContext(
                target_path=target_path,
                target_symbol=(
                    target_symbol.strip()
                    if separator and target_symbol.strip()
                    else None
                ),
                is_new_file=is_new_file,
                evidence_source=evidence_source,
            )
    return None


def _install_creation_conflict_classification(loop_module: Any) -> None:
    if getattr(loop_module, _MUTATION_MARKER, False):
        return
    original = loop_module._mutation_target_error

    def mutation_target_error(
        tool_name: str,
        arguments: Mapping[str, Any],
        context: Any,
    ) -> str | None:
        if tool_name == "apply_source_edit" and context is not None:
            pinned = loop_module._canonical_mutation_path(
                getattr(context, "target_path", None)
            )
            operation = str(arguments.get("operation") or "").strip().casefold()
            if (
                pinned
                and not bool(getattr(context, "is_new_file", False))
                and operation in loop_module._SOURCE_CREATE_OPERATIONS
            ):
                return (
                    "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "
                    f"{pinned!r} cannot be recreated by {operation!r}."
                )
        return original(tool_name, arguments, context)

    mutation_target_error.__name__ = original.__name__
    loop_module._mutation_target_error = mutation_target_error
    setattr(loop_module, _MUTATION_MARKER, True)


def install(target_module: Any | None = None, loop_module: Any | None = None) -> None:
    if target_module is None:
        from . import small_model_task_capsule_contract as target_module
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module

    if not getattr(target_module, _MARKER, False):
        original_compile = target_module.compile_task_capsule

        def compile_task_capsule(module: Any):
            try:
                return original_compile(module)
            except target_module.TaskCapsuleContractError as exc:
                status = _primary_status(target_module, module)
                if status in _EXISTING_STATUSES:
                    reconciled = _module_with_reserved_primary(target_module, module)
                    if reconciled is None:
                        raise
                    proxy, anchors = reconciled
                    capsule = original_compile(proxy)
                    if capsule is None:
                        return None
                    return _restore_capsule_authority(target_module, capsule, anchors)
                if status and "PRIMARY_NOT_RESERVED" not in str(exc):
                    raise target_module.TaskCapsuleContractError(
                        "TASK_CAPSULE_PRIMARY_NOT_RESERVED: approved custom-Java primary "
                        f"uses unsupported status {status!r}."
                    ) from exc
                raise

        compile_task_capsule.__name__ = original_compile.__name__
        target_module.compile_task_capsule = compile_task_capsule
        setattr(target_module, _MARKER, True)

    if not getattr(loop_module, _LOOP_MARKER, False):
        loop_module._fresh_owned_symbol_context = (
            lambda payload: _status_aware_owned_symbol_context(loop_module, payload)
        )
        setattr(loop_module, _LOOP_MARKER, True)

    _install_creation_conflict_classification(loop_module)


def assert_installed(target_module: Any | None = None, loop_module: Any | None = None) -> None:
    if target_module is None:
        from . import small_model_task_capsule_contract as target_module
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if not getattr(target_module, _MARKER, False):
        raise RuntimeError("coder mutation authority reconciliation is not installed")
    if not getattr(loop_module, _LOOP_MARKER, False):
        raise RuntimeError("coder target-existence localization reconciliation is not installed")
    if not getattr(loop_module, _MUTATION_MARKER, False):
        raise RuntimeError("coder creation-conflict classification is not installed")


__all__ = ["assert_installed", "install"]
