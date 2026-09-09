from __future__ import annotations

"""Final guard for host-owned mutation authority.

The runtime composes several mutation/localization wrappers.  This guard is installed
last and makes their shared invariant explicit: a host-issued task target is immutable
for the lifetime of one coder run.  Retrieval may refine source/symbol evidence for that
target, but it cannot replace the target or expand the writable exact-set.

This deliberately derives authority structurally from host-role task payloads instead
of coupling correctness to an ``evidence_source`` string produced by another wrapper.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_MARKER = "_mmm_mutation_authority_final_guard_v1"
_HOST_ROLES = frozenset({"system", "developer", "tool"})
_AUTHORITY_PRIORITY = {
    "mmm/direct-task-mutation-authority-v1": 300,
    "mmm/small-model-task-capsule": 200,
}
_DEFAULT_AUTHORITY_PRIORITY = 100


def _structured_payload(content: Any) -> Any | None:
    if isinstance(content, (Mapping, list, tuple)):
        return content
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if not raw.startswith(("{", "[")):
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _explicit_mutation_target(payload: Any, loop_module: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    target = payload.get("mutation_target")
    if not isinstance(target, Mapping):
        return ""
    return loop_module._canonical_mutation_path(target.get("path"))


def _authority_priority(payload: Mapping[str, Any]) -> int:
    schema = str(payload.get("schema_version") or "").strip()
    return _AUTHORITY_PRIORITY.get(schema, _DEFAULT_AUTHORITY_PRIORITY)


def _union_paths(loop_module: Any, *groups: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for group in groups:
        for raw in group:
            path = loop_module._canonical_mutation_path(raw)
            if path and path not in result:
                result.append(path)
    return tuple(result)


def _authority_signature(context: Any, loop_module: Any) -> tuple[Any, ...]:
    return (
        loop_module._canonical_mutation_path(getattr(context, "target_path", "")),
        tuple(getattr(context, "writable_paths", ()) or ()),
        tuple(getattr(context, "creatable_paths", ()) or ()),
    )


def _host_pin_from_messages(
    messages: Sequence[Mapping[str, Any]], loop_module: Any
) -> Any | None:
    """Resolve the strongest explicit host target, independent of wrapper message order."""

    candidates: list[tuple[int, int, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().casefold()
        if role not in _HOST_ROLES:
            continue
        payload = _structured_payload(message.get("content"))
        if not isinstance(payload, Mapping):
            continue
        explicit = _explicit_mutation_target(payload, loop_module)
        if not explicit:
            continue
        parser = getattr(loop_module, "_planir_owned_anchor_sets", None)
        if not callable(parser):
            continue
        writable, creatable = parser(payload)
        writable = _union_paths(loop_module, writable)
        creatable = _union_paths(loop_module, creatable)
        if explicit not in set(writable):
            continue

        context = loop_module._extract_mutation_context_from_payload(payload)
        if context is None:
            continue
        target = loop_module._canonical_mutation_path(context.target_path)
        if target != explicit:
            continue
        if not hasattr(context, "target_pinned"):
            continue
        pinned = replace(
            context,
            writable_paths=writable,
            creatable_paths=creatable,
            target_pinned=True,
        )
        candidates.append((_authority_priority(payload), index, pinned))

    if not candidates:
        return None

    strongest = max(priority for priority, _index, _context in candidates)
    finalists = [
        (index, context)
        for priority, index, context in candidates
        if priority == strongest
    ]
    signatures = {
        _authority_signature(context, loop_module)
        for _index, context in finalists
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "MUTATION_AUTHORITY_CONFLICT: equally authoritative host task receipts "
            "disagree on the exact mutation target or writable set."
        )

    # Equal signatures are semantically identical. Choosing the last receipt makes the
    # tie deterministic without allowing wrapper insertion order to alter authority.
    return max(finalists, key=lambda item: item[0])[1]


def install(loop_module: Any | None = None) -> None:
    """Install after all runtime wrappers so authority cannot be weakened later."""

    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module

    if getattr(loop_module, _MARKER, False):
        return

    Context = loop_module.TargetMutationContext
    original_merge = Context.merge
    original_is_ready = loop_module.is_mutation_ready

    def merge(self: Any, other: Any):
        """Pinned authority is immutable; retrieval is evidence only."""

        if not bool(getattr(self, "target_pinned", False)):
            return original_merge(self, other)
        if other is None:
            return self

        left = loop_module._canonical_mutation_path(getattr(self, "target_path", ""))
        right = loop_module._canonical_mutation_path(getattr(other, "target_path", ""))
        if left and right and left != right:
            return self

        merged = original_merge(self, other)
        if not hasattr(merged, "target_pinned"):
            return merged
        return replace(
            merged,
            writable_paths=tuple(getattr(self, "writable_paths", ()) or ()),
            creatable_paths=tuple(getattr(self, "creatable_paths", ()) or ()),
            target_pinned=True,
        )

    def is_mutation_ready(messages: Sequence[Mapping[str, Any]], state: Any) -> bool:
        # Install the host pin *before* generic message scanning.  This prevents a
        # retrieved/fixture file from becoming target identity merely because it is a
        # READY source snapshot and appears earlier in the execution flow.
        host_pin = _host_pin_from_messages(messages, loop_module)
        if host_pin is not None:
            with state._lock:
                current = state.mutation_context
                current_pinned = bool(getattr(current, "target_pinned", False)) if current else False
                current_path = loop_module._canonical_mutation_path(
                    getattr(current, "target_path", "") if current else ""
                )
                host_path = loop_module._canonical_mutation_path(host_pin.target_path)
                if not current_pinned or current_path != host_path:
                    state.mutation_context = host_pin
        return original_is_ready(messages, state)

    Context.merge = merge
    loop_module.is_mutation_ready = is_mutation_ready
    setattr(loop_module, _MARKER, True)


def assert_installed(loop_module: Any | None = None) -> None:
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if not getattr(loop_module, _MARKER, False):
        raise RuntimeError("final mutation-authority guard is not installed")


__all__ = ["assert_installed", "install"]
