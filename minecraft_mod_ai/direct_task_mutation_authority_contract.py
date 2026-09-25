from __future__ import annotations

"""Compile exact host-owned mutation authority before generation.

Repository localization and file selection must finish before this contract is compiled.
An editor receives only exact owned paths; a raw saved-authored request is rejected rather
than being upgraded to directory/root authority.
"""

import contextvars
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .mutation_authority import MutationAuthority, MutationAuthorityMode
from .owned_target_contract import target_is_writable

_SCHEMA = "mmm/direct-task-mutation-authority-v1"
_ALLOWED_PREFIXES = (
    "src/main/java/",
    "src/client/java/",
    "src/main/resources/",
    "src/client/resources/",
    "src/test/java/",
    "src/gametest/",
)
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
        return _authority_creatable_paths(self)

    def to_host_payload(self) -> dict[str, Any]:
        return _authority_host_payload(self)


def _authority_creatable_paths(authority: DirectTaskMutationAuthority) -> tuple[str, ...]:
    return tuple(
        _anchor_path(anchor)
        for anchor in authority.writable_anchors
        if str(anchor.get("status") or "").strip().casefold() == "host_reserved"
    )


def _authority_host_payload(authority: DirectTaskMutationAuthority) -> dict[str, Any]:
    if authority.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS:
        return {
            "schema_version": "mmm/authored-bounded-mutation-authority-v2",
            "task_id": authority.task_id,
            "authority_sha256": authority.authority_sha256,
            "mutation_authority": {
                "mode": authority.mutation_authority.mode.value,
                "roots": list(authority.mutation_authority.roots),
                "delete_allowed": False,
            },
            "instruction": (
                "Implement the approved authored design coherently inside these exact "
                "host-owned roots. Choose the Java/resource files required by the design; "
                "document headings are contract facets, not file/class names. Preserve the "
                "existing canonical Fabric entrypoint and never write build configuration "
                "or another mod namespace."
            ),
        }

    primary_anchor = next(
        anchor
        for anchor in authority.writable_anchors
        if _anchor_path(anchor) == authority.primary_path
        and str(anchor.get("kind") or "").strip() == "symbol"
    )
    task: dict[str, Any] = {
        "task_id": authority.task_id,
        "owned_anchors": [dict(anchor) for anchor in authority.writable_anchors],
        "production_bindings": [{
            "task_ref": authority.task_id,
            "reuse_action": "fresh",
            "owned_anchors": [dict(primary_anchor)],
        }],
    }
    if authority.task_sha256:
        task["task_sha256"] = authority.task_sha256
    return {
        "schema_version": _SCHEMA,
        "task_id": authority.task_id,
        "authority_sha256": authority.authority_sha256,
        "mutation_target": {
            "path": authority.primary_path,
            "symbol": authority.primary_symbol,
            "mode": "create_or_edit_exact_host_binding",
        },
        "module": {
            "module_id": authority.task_id,
            "kind": authority.module_kind,
            "config": {"evidence_task": task},
        },
    }


def _authored_bounded_roots(module: Any) -> tuple[str, ...]:
    config = _module_config(module)
    if config.get("authored_bounded_scope") is not True:
        return ()
    package_name = str(config.get("authored_java_package") or "").strip()
    mod_id = str(config.get("authored_mod_id") or "").strip()
    if not package_name or re.fullmatch(
        r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", package_name
    ) is None:
        raise DirectTaskMutationAuthorityError(
            "AUTHORED_AUTHORITY_PACKAGE_INVALID: bounded authored generation requires "
            "one safe host-generated Java package."
        )
    if not mod_id or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", mod_id) is None:
        raise DirectTaskMutationAuthorityError(
            "AUTHORED_AUTHORITY_MOD_ID_INVALID: bounded authored generation requires "
            "one safe host-owned mod id."
        )
    package_path = package_name.replace(".", "/")
    return (
        f"src/main/java/{package_path}/",
        f"src/client/java/{package_path}/",
        f"src/main/resources/assets/{mod_id}/",
        f"src/main/resources/data/{mod_id}/",
        f"src/client/resources/assets/{mod_id}/",
        f"src/test/java/{package_path}/",
        f"src/gametest/{package_path}/",
    )


def _compile_bounded_authored_authority(module: Any) -> DirectTaskMutationAuthority:
    module_id = str(getattr(module, "module_id", "") or "").strip()
    if not module_id:
        raise DirectTaskMutationAuthorityError(
            "AUTHORED_AUTHORITY_TASK_MISSING: bounded authored design has no module id."
        )
    roots = _authored_bounded_roots(module)
    if not roots:
        raise DirectTaskMutationAuthorityError(
            "AUTHORED_AUTHORITY_SCOPE_MISSING: bounded authored design has no roots."
        )
    mutation_authority = MutationAuthority.bounded_roots(roots, task_id=module_id)
    payload = {
        "task_id": module_id,
        "module_kind": str(getattr(module, "kind", "") or "").strip(),
        "mode": mutation_authority.mode.value,
        "roots": mutation_authority.roots,
    }
    return DirectTaskMutationAuthority(
        task_id=module_id,
        module_kind=str(getattr(module, "kind", "") or "").strip(),
        primary_path="",
        primary_symbol="",
        writable_anchors=(),
        authority_sha256=_authority_digest(payload),
        mutation_authority=mutation_authority,
    )


def _special_authority(module: Any) -> tuple[bool, DirectTaskMutationAuthority | None]:
    if module is None:
        return True, None
    if _is_authored_design(module) and _module_evidence_task(module) is None:
        if _module_config(module).get("authored_bounded_scope") is True:
            return True, _compile_bounded_authored_authority(module)
        raise DirectTaskMutationAuthorityError(
            "AUTHORED_LOCALIZATION_REQUIRED: saved authored work must be localized "
            "and frozen to an exact evidence_task before mutation authority is compiled."
        )
    return False, None


def _custom_java_task(module: Any) -> tuple[str, Mapping[str, Any]] | None:
    module_kind = str(getattr(module, "kind", "") or "").strip()
    if module_kind != "custom_java":
        return None
    task = _module_evidence_task(module)
    if task is None:
        return None
    return module_kind, task


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


def compile_direct_task_mutation_authority(
    module: Any,
) -> DirectTaskMutationAuthority | None:
    """Compile host authority from the trusted module object before model generation."""

    handled, special = _special_authority(module)
    if handled:
        return special
    custom_task = _custom_java_task(module)
    if custom_task is None:
        return None
    module_kind, task = custom_task

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

    writable_anchors = tuple(
        anchor
        for anchor in anchors
        if target_is_writable(anchor.get("status"))
    )
    if not writable_anchors:
        return None
    unsupported = tuple(
        anchor
        for anchor in anchors
        if not target_is_writable(anchor.get("status"))
    )
    if unsupported:
        rendered = [
            f"{anchor.get('locator')}:{anchor.get('status') or '<empty>'}"
            for anchor in unsupported
        ]
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_TARGET_STATUS_INVALID: exact custom_java authority contains "
            f"unsupported writable-target status: {rendered!r}."
        )

    bindings = _matching_fresh_bindings(task, task_id)
    if bindings == ():
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_REUSE_UNSUPPORTED: exact custom_java authority requires the "
            "canonical fresh production binding."
        )
    if bindings is None:
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_BINDING_MISSING: exact custom_java task has no matching "
            "production_binding."
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

    writable_paths = tuple(
        dict.fromkeys(_anchor_path(anchor) for anchor in writable_anchors)
    )
    if primary_path not in writable_paths:
        raise DirectTaskMutationAuthorityError(
            "PLANIR_AUTHORITY_PRIMARY_NOT_OWNED: production-binding primary must be an "
            "exact writable task owned_anchor."
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
    keys = tuple(getattr(loop_module, "_SOURCE_EDIT_PATH_KEYS", ("path",)))
    candidates = (arguments.get(key) for key in keys)
    return next(
        (value for value in candidates if isinstance(value, str) and value.strip()),
        "",
    )


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


__all__ = [
    "DirectTaskMutationAuthority",
    "DirectTaskMutationAuthorityError",
    "_CURRENT_AUTHORITY",
    "compile_direct_task_mutation_authority",
]
