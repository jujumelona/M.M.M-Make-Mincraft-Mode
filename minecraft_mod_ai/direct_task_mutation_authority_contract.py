from __future__ import annotations

"""Carry fresh PlanIR mutation authority out-of-band from the host module call.

The canonical custom-module path receives its approved ``ProductionModule`` as a Python
object. Mutation authority therefore must not depend on reparsing the model-facing
``role=user`` JSON envelope. This contract captures the exact host-reserved task paths
at ``CustomModuleGenerator.generate`` entry and injects a compact host-role authority
receipt immediately before the tool loop consumes the request.

The model may observe arbitrary project files (for example ``fabric.mod.json``) through
RAG, but those observations never become write authority. Missing or inconsistent
fresh-task ownership fails before the first coder decode instead of falling back to a
retrieved file and retrying an impossible mutation.
"""

import contextvars
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

_MARKER = "_mmm_direct_task_mutation_authority_v1"
_SCHEMA = "mmm/direct-task-mutation-authority-v1"
_ALLOWED_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
)
_CURRENT_AUTHORITY: contextvars.ContextVar["DirectTaskMutationAuthority | None"] = (
    contextvars.ContextVar("mmm_direct_task_mutation_authority", default=None)
)


class DirectTaskMutationAuthorityError(RuntimeError):
    """Raised when approved fresh-task ownership cannot form an executable contract."""


@dataclass(frozen=True)
class DirectTaskMutationAuthority:
    task_id: str
    module_kind: str
    primary_path: str
    primary_symbol: str
    writable_anchors: tuple[dict[str, Any], ...]
    authority_sha256: str
    task_sha256: str = ""

    @property
    def writable_paths(self) -> tuple[str, ...]:
        return tuple(_anchor_path(anchor) for anchor in self.writable_anchors)

    @property
    def creatable_paths(self) -> tuple[str, ...]:
        return tuple(
            _anchor_path(anchor)
            for anchor in self.writable_anchors
            if str(anchor.get("status") or "").strip().casefold() == "host_reserved"
        )

    def to_host_payload(self) -> dict[str, Any]:
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


def _module_evidence_task(module: Any) -> Mapping[str, Any] | None:
    config = getattr(module, "config", None)
    if not isinstance(config, Mapping):
        return None
    task = config.get("evidence_task")
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
    """Compile one fresh custom-Java task into immutable exact-path authority.

    Returns ``None`` for legacy/non-fresh module shapes that remain owned by the existing
    localization contracts. A module that *does* declare fresh host-reserved PlanIR
    anchors is never allowed to degrade silently: malformed ownership raises before any
    coder model call.
    """

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
        # Reuse/adapt tasks keep their existing localization path; this contract only
        # replaces the fragile fresh-task authority reconstruction path.
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


def install(
    *,
    custom_module_generator_module: Any | None = None,
    loop_module: Any | None = None,
) -> None:
    """Install the out-of-band fresh-task authority bridge before runtime finalization."""

    if custom_module_generator_module is None:
        from . import custom_module_generator as custom_module_generator_module
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module

    if getattr(custom_module_generator_module, _MARKER, False):
        return

    Generator = custom_module_generator_module.CustomModuleGenerator
    original_generate = Generator.generate
    original_loop_generate = loop_module.generate_with_tools

    @wraps(original_generate)
    def generate(self, *args: Any, **kwargs: Any):
        module = kwargs.get("module")
        authority = compile_direct_task_mutation_authority(module)
        token = _CURRENT_AUTHORITY.set(authority)
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            _CURRENT_AUTHORITY.reset(token)

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
    loop_module.generate_with_tools = generate_with_tools
    setattr(custom_module_generator_module, _MARKER, True)
    setattr(loop_module, _MARKER, True)


__all__ = [
    "DirectTaskMutationAuthority",
    "DirectTaskMutationAuthorityError",
    "compile_direct_task_mutation_authority",
    "install",
]
