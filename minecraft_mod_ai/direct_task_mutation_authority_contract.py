from __future__ import annotations

"""Compile host-owned mutation authority before generation and enforce it end-to-end.

Ordinary fresh PlanIR tasks keep exact-path authority. Saved authored designs are a
separate host request shape: the host intentionally delegates file selection inside the
four generated source/resource roots. That distinction is made from the trusted
``ProductionModule`` object before model decode, never by reparsing model-facing text.
"""

import contextvars
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

from .mutation_authority import (
    AUTHORED_DESIGN_ROOTS,
    MutationAuthority,
    MutationAuthorityMode,
)

_MARKER = "_mmm_direct_task_mutation_authority_v1"
_SCHEMA = "mmm/direct-task-mutation-authority-v1"
_AUTHORED_SCHEMA = "mmm/authored-design-mutation-authority-v1"
_ALLOWED_PREFIXES = AUTHORED_DESIGN_ROOTS
_CURRENT_AUTHORITY: contextvars.ContextVar["DirectTaskMutationAuthority | None"] = (
    contextvars.ContextVar("mmm_direct_task_mutation_authority", default=None)
)


class DirectTaskMutationAuthorityError(RuntimeError):
    """Raised when approved host ownership cannot form an executable contract."""


@dataclass(frozen=True)
class DirectTaskMutationAuthority:
    task_id: str
    module_kind: str
    primary_path: str
    primary_symbol: str
    writable_anchors: tuple[dict[str, Any], ...]
    authority_sha256: str
    mutation_authority: MutationAuthority
    task_sha256: str = ""

    @property
    def writable_paths(self) -> tuple[str, ...]:
        if self.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS:
            return self.mutation_authority.roots
        return self.mutation_authority.paths

    @property
    def creatable_paths(self) -> tuple[str, ...]:
        if self.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS:
            return self.mutation_authority.roots
        return tuple(
            _anchor_path(anchor)
            for anchor in self.writable_anchors
            if str(anchor.get("status") or "").strip().casefold() == "host_reserved"
        )

    @property
    def is_bounded_authored_design(self) -> bool:
        return self.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS

    def to_host_payload(self) -> dict[str, Any]:
        if self.is_bounded_authored_design:
            return {
                "schema_version": _AUTHORED_SCHEMA,
                "task_id": self.task_id,
                "authority_sha256": self.authority_sha256,
                "mutation_authority": {
                    "mode": self.mutation_authority.mode.value,
                    "roots": list(self.mutation_authority.roots),
                    "delete_allowed": False,
                },
                "instruction": (
                    "Host-authored design authority is already fixed. You may create or edit "
                    "files only below the declared roots. Build configuration, host state and "
                    "deletes are forbidden. Retrieval/localization does not widen this authority."
                ),
            }

        primary_anchor = next(
            anchor
            for anchor in self.writable_anchors
            if _anchor_path(anchor) == self.primary_path
            and str(anchor.get("kind") or "").strip() == "symbol"
        )
        task: dict[str, Any] = {
            "task_id": self.task_id,
            "owned_anchors": [dict(anchor) for anchor in self.writable_anchors],
            "production_bindings": [
                {
                    "task_ref": self.task_id,
                    "reuse_action": "fresh",
                    "owned_anchors": [dict(primary_anchor)],
                }
            ],
        }
        if self.task_sha256:
            task["task_sha256"] = self.task_sha256
        return {
            "schema_version": _SCHEMA,
            "task_id": self.task_id,
            "authority_sha256": self.authority_sha256,
            "mutation_target": {
                "path": self.primary_path,
                "symbol": self.primary_symbol,
                "mode": "create_or_edit_exact_host_binding",
            },
            "module": {
                "module_id": self.task_id,
                "kind": self.module_kind,
                "config": {"evidence_task": task},
            },
        }


def _canonical_path(locator: Any) -> str:
    raw = str(locator or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = raw.split("#", 1)[0].strip()
    while path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or ":" in path:
        return ""
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    if not any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        return ""
    return PurePosixPath(path).as_posix()


def _anchor_path(anchor: Mapping[str, Any]) -> str:
    return _canonical_path(anchor.get("locator"))


def _canonical_anchor(anchor: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _anchor_path(anchor)
    if not path:
        return None
    locator = str(anchor.get("locator") or "").replace("\\", "/").strip()
    _raw_path, separator, symbol = locator.partition("#")
    normalized: dict[str, Any] = {
        "kind": str(anchor.get("kind") or "").strip(),
        "locator": path + (f"#{symbol.strip()}" if separator and symbol.strip() else ""),
        "status": str(anchor.get("status") or "").strip(),
    }
    for key in ("ownership", "module_id", "source_set"):
        value = str(anchor.get(key) or "").strip()
        if value:
            normalized[key] = value
    return normalized


def _module_config(module: Any) -> Mapping[str, Any]:
    config = getattr(module, "config", None)
    return config if isinstance(config, Mapping) else {}


def _is_authored_design(module: Any) -> bool:
    return isinstance(_module_config(module).get("authored_plan"), Mapping)


def _module_evidence_task(module: Any) -> Mapping[str, Any] | None:
    task = _module_config(module).get("evidence_task")
    return task if isinstance(task, Mapping) else None


def _matching_fresh_bindings(
    task: Mapping[str, Any], task_id: str
) -> tuple[Mapping[str, Any], ...] | None:
    raw = task.get("production_bindings")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    matching = tuple(
        binding
        for binding in raw
        if isinstance(binding, Mapping)
        and str(binding.get("task_ref") or "").strip() == task_id
    )
    if not matching:
        return None
    actions = {
        str(binding.get("reuse_action") or "").strip().casefold()
        for binding in matching
        if str(binding.get("reuse_action") or "").strip()
    }
    if actions != {"fresh"}:
        return ()
    return matching


def _binding_symbol_candidates(
    bindings: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[str, str]] = []
    for binding in bindings:
        anchors = binding.get("owned_anchors")
        if not isinstance(anchors, Sequence) or isinstance(
            anchors, (str, bytes, bytearray)
        ):
            continue
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                continue
            if str(anchor.get("kind") or "").strip() != "symbol":
                continue
            canonical = _canonical_anchor(anchor)
            if canonical is None:
                continue
            locator = str(canonical["locator"])
            path, separator, symbol = locator.partition("#")
            item = (path, symbol.strip() if separator else "")
            if item not in candidates:
                candidates.append(item)
    return tuple(candidates)


def _authority_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _compile_authored_authority(module: Any) -> DirectTaskMutationAuthority:
    module_id = str(getattr(module, "module_id", "") or "").strip()
    module_kind = str(getattr(module, "kind", "") or "").strip()
    if not module_id:
        raise DirectTaskMutationAuthorityError(
            "AUTHORED_AUTHORITY_TASK_MISSING: authored design requires a host module id."
        )
    mutation_authority = MutationAuthority.bounded_roots(
        AUTHORED_DESIGN_ROOTS,
        task_id=module_id,
    )
    payload = {
        "task_id": module_id,
        "module_kind": module_kind,
        "mode": mutation_authority.mode.value,
        "roots": mutation_authority.roots,
    }
    return DirectTaskMutationAuthority(
        task_id=module_id,
        module_kind=module_kind,
        primary_path="",
        primary_symbol="",
        writable_anchors=(),
        authority_sha256=_authority_digest(payload),
        mutation_authority=mutation_authority,
    )


def compile_direct_task_mutation_authority(
    module: Any,
) -> DirectTaskMutationAuthority | None:
    """Compile host authority from the trusted module object before model generation."""

    if module is None:
        return None
    if _is_authored_design(module):
        return _compile_authored_authority(module)

    module_kind = str(getattr(module, "kind", "") or "").strip()
    if module_kind != "custom_java":
        return None
    task = _module_evidence_task(module)
    if task is None:
        return None

    module_id = str(getattr(module, "module_id", "") or "").strip()
    task_id = str(task.get("task_id") or "").strip()
    if not module_id or not task_id or module_id != task_id:
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_TASK_MISMATCH: custom_java module_id and evidence_task.task_id "
            "must match exactly before generation."
        )

    raw_anchors = task.get("owned_anchors")
    if not isinstance(raw_anchors, Sequence) or isinstance(
        raw_anchors, (str, bytes, bytearray)
    ):
        return None
    anchors: list[dict[str, Any]] = []
    for raw in raw_anchors:
        if not isinstance(raw, Mapping):
            continue
        canonical = _canonical_anchor(raw)
        if canonical is not None and canonical not in anchors:
            anchors.append(canonical)

    host_reserved = tuple(
        anchor
        for anchor in anchors
        if str(anchor.get("status") or "").strip().casefold() == "host_reserved"
    )
    if not host_reserved:
        return None

    bindings = _matching_fresh_bindings(task, task_id)
    if bindings == ():
        return None
    if bindings is None:
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_BINDING_MISSING: fresh host-reserved custom_java task has no "
            "matching production_binding."
        )

    candidates = _binding_symbol_candidates(bindings)
    if len(candidates) != 1:
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_PRIMARY_AMBIGUOUS: fresh custom_java task requires exactly "
            f"one concrete production-binding symbol, found {len(candidates)}."
        )
    primary_path, primary_symbol = candidates[0]
    if not primary_path.endswith(".java"):
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_PRIMARY_NOT_JAVA: custom_java primary mutation target must be .java."
        )

    writable_paths = {_anchor_path(anchor) for anchor in anchors}
    creatable_paths = {_anchor_path(anchor) for anchor in host_reserved}
    if primary_path not in writable_paths or primary_path not in creatable_paths:
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_PRIMARY_NOT_OWNED: production-binding primary must be a "
            "host_reserved task owned_anchor."
        )

    mutation_authority = MutationAuthority.exact(writable_paths, task_id=task_id)
    payload = {
        "task_id": task_id,
        "module_kind": module_kind,
        "primary_path": primary_path,
        "primary_symbol": primary_symbol,
        "writable_anchors": anchors,
        "task_sha256": str(task.get("task_sha256") or "").strip(),
    }
    return DirectTaskMutationAuthority(
        task_id=task_id,
        module_kind=module_kind,
        primary_path=primary_path,
        primary_symbol=primary_symbol,
        writable_anchors=tuple(anchors),
        authority_sha256=_authority_digest(payload),
        mutation_authority=mutation_authority,
        task_sha256=str(task.get("task_sha256") or "").strip(),
    )


def _authority_message(authority: DirectTaskMutationAuthority) -> dict[str, str]:
    return {
        "role": "developer",
        "content": json.dumps(
            authority.to_host_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _insert_authority_message(
    messages: Sequence[Mapping[str, Any]], authority: DirectTaskMutationAuthority
) -> tuple[Mapping[str, Any], ...]:
    items = list(messages)
    insert_at = (
        1
        if items
        and str(items[0].get("role") or "").strip().casefold() == "system"
        else 0
    )
    items.insert(insert_at, _authority_message(authority))
    return tuple(items)


def _mutation_path(arguments: Mapping[str, Any], loop_module: Any) -> str:
    for key in tuple(getattr(loop_module, "_SOURCE_EDIT_PATH_KEYS", ("path",))):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _validate_operation_with_authority(
    operation: Mapping[str, Any], authority: DirectTaskMutationAuthority
) -> str | None:
    path = (
        operation.get("path")
        or operation.get("file")
        or operation.get("target_path")
        or operation.get("target_file")
    )
    return authority.mutation_authority.mutation_error(
        path,
        operation=operation.get("operation"),
    )


def install(
    *,
    custom_module_generator_module: Any | None = None,
    loop_module: Any | None = None,
) -> None:
    """Install one host authority across loop dispatch and final staged-patch validation."""

    if custom_module_generator_module is None:
        from . import custom_module_generator as custom_module_generator_module
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module

    if getattr(custom_module_generator_module, _MARKER, False):
        return

    Generator = custom_module_generator_module.CustomModuleGenerator
    original_generate = Generator.generate
    original_validate_operations = Generator._validate_operations
    original_loop_generate = loop_module.generate_with_tools
    original_target_error = loop_module._mutation_target_error
    original_turn = loop_module._generate_turn_with_context_recovery

    @wraps(original_generate)
    def generate(self, *args: Any, **kwargs: Any):
        module = kwargs.get("module")
        authority = compile_direct_task_mutation_authority(module)
        token = _CURRENT_AUTHORITY.set(authority)
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            _CURRENT_AUTHORITY.reset(token)

    @wraps(original_validate_operations)
    def validate_operations(self, operations: list[dict[str, Any]]) -> None:
        original_validate_operations(self, operations)
        authority = _CURRENT_AUTHORITY.get()
        if authority is None:
            return
        for operation in operations:
            error = _validate_operation_with_authority(operation, authority)
            if error is not None:
                raise custom_module_generator_module.CustomModuleGenerationError(error)

    @wraps(original_target_error)
    def mutation_target_error(
        tool_name: str,
        arguments: Mapping[str, Any],
        context: Any,
    ) -> str | None:
        authority = _CURRENT_AUTHORITY.get()
        if authority is None or tool_name != "apply_source_edit":
            return original_target_error(tool_name, arguments, context)
        supplied = _mutation_path(arguments, loop_module)
        error = authority.mutation_authority.mutation_error(
            supplied,
            operation=arguments.get("operation"),
        )
        if error is not None:
            return error
        if authority.is_bounded_authored_design:
            # Localization can identify fabric.mod.json or another evidence anchor. It is
            # deliberately separate from the authored design's bounded write authority.
            return None
        return original_target_error(tool_name, arguments, context)

    @wraps(original_turn)
    def generate_turn_with_host_authority(*args: Any, **kwargs: Any):
        authority = _CURRENT_AUTHORITY.get()
        tool_choice = kwargs.get("tool_choice")
        function = tool_choice.get("function") if isinstance(tool_choice, Mapping) else None
        forced = (
            str(function.get("name") or "").strip()
            if isinstance(function, Mapping)
            else ""
        )
        if (
            authority is not None
            and authority.is_bounded_authored_design
            and forced == "apply_source_edit"
        ):
            kwargs["parallel_tool_calls"] = True
            request = kwargs.get("request")
            if request is not None and hasattr(request, "parallel_tool_calls"):
                kwargs["request"] = replace(request, parallel_tool_calls=True)
        return original_turn(*args, **kwargs)

    @wraps(original_loop_generate)
    def generate_with_tools(
        router,
        *,
        config,
        adapter,
        request,
        runtime,
        stage,
        role,
    ):
        authority = _CURRENT_AUTHORITY.get()
        if authority is not None and str(stage) == "generation" and role in {
            "coder",
            "coder_safe",
        }:
            payload = authority.to_host_payload()
            if not authority.is_bounded_authored_design:
                authority_parser = getattr(loop_module, "_planir_owned_anchor_sets", None)
                if callable(authority_parser):
                    writable, creatable = authority_parser(payload)
                    if (
                        authority.primary_path not in set(writable)
                        or authority.primary_path not in set(creatable)
                    ):
                        raise DirectTaskMutationAuthorityError(
                            "PLANIR_AUTHORITY_RUNTIME_DRIFT: installed PlanIR parser no longer "
                            "recognizes the host-bound primary path before coder decode."
                        )
            request = replace(
                request,
                messages=_insert_authority_message(request.messages, authority),
            )
        return original_loop_generate(
            router,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    Generator.generate = generate
    Generator._validate_operations = validate_operations
    loop_module._mutation_target_error = mutation_target_error
    loop_module._generate_turn_with_context_recovery = generate_turn_with_host_authority
    loop_module.generate_with_tools = generate_with_tools
    setattr(custom_module_generator_module, _MARKER, True)
    setattr(loop_module, _MARKER, True)


__all__ = [
    "DirectTaskMutationAuthority",
    "DirectTaskMutationAuthorityError",
    "_CURRENT_AUTHORITY",
    "compile_direct_task_mutation_authority",
    "install",
]
