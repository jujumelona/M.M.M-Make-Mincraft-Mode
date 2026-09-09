from __future__ import annotations

"""Final guard for host-owned mutation authority and semantic failure boundaries.

The runtime composes several mutation/localization wrappers. This guard is installed
last and makes their shared invariants explicit:

* a host-issued task target is immutable for the lifetime of one coder run;
* retrieval may refine source/symbol evidence, but cannot replace the target or expand
  the writable exact-set;
* once a mutation tool has parsed its arguments and the host has returned a structured
  semantic rejection, the coder must not re-enter argument generation for the same
  action. That failure belongs to the outer adjudication/replan boundary.

Authority is derived structurally from host-role task payloads instead of coupling
correctness to an ``evidence_source`` string produced by another wrapper.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

_MARKER = "_mmm_mutation_authority_final_guard_v1"
_SEMANTIC_BOUNDARY_MARKER = "_mmm_post_argument_semantic_boundary_v1"
_HOST_ROLES = frozenset({"system", "developer", "tool"})
_AUTHORITY_PRIORITY = {
    "mmm/direct-task-mutation-authority-v1": 300,
    "mmm/small-model-task-capsule": 200,
}
_DEFAULT_AUTHORITY_PRIORITY = 100
_POST_ARGUMENT_SEMANTIC_FAILURE_CODES = frozenset(
    {
        "MUTATION_TARGET_DRIFT",
        "MUTATION_TARGET_UNBOUND",
        "MUTATION_TARGET_CREATION_CONFLICT",
        "PATH_OUTSIDE_WRITABLE_SET",
        "PHASE_PROTOCOL_VIOLATION",
    }
)
_POST_ARGUMENT_SEMANTIC_FAILURE_PREFIXES = (
    "MUTATION_AUTHORITY_",
    "MUTATION_TARGET_",
    "WRITE_SCOPE_",
)


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


def _is_post_argument_semantic_failure_code(code: Any) -> bool:
    normalized = str(code or "").strip().upper()
    if not normalized:
        return False
    return normalized in _POST_ARGUMENT_SEMANTIC_FAILURE_CODES or normalized.startswith(
        _POST_ARGUMENT_SEMANTIC_FAILURE_PREFIXES
    )


def _latest_post_argument_semantic_failure(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Return a trailing host/tool semantic rejection, never an argument decode error.

    A real post-argument failure exists only after a tool result was emitted. Native
    argument generation/JSON/schema failures happen before such a result exists and are
    intentionally left to ``native_atomic_argument_recovery``.
    """

    for message in reversed(messages):
        if not isinstance(message, Mapping):
            break
        role = str(message.get("role") or "").strip().casefold()
        if role != "tool":
            break
        payload = _structured_payload(message.get("content"))
        if not isinstance(payload, Mapping) or payload.get("ok") is not False:
            continue
        code = str(payload.get("failure_code") or "").strip().upper()
        if not _is_post_argument_semantic_failure_code(code):
            continue
        reason = str(payload.get("error") or payload.get("reason") or code).strip()
        return code, reason
    return None


def _forced_tool_name(tool_choice: Any) -> str:
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            return str(function.get("name") or "").strip()
        name = tool_choice.get("name")
        if isinstance(name, str):
            return name.strip()
    return ""


def _install_semantic_generation_boundary(loop_module: Any) -> None:
    if getattr(loop_module, _SEMANTIC_BOUNDARY_MARKER, False):
        return

    original_generate = getattr(loop_module, "_generate_turn_with_context_recovery", None)
    if not callable(original_generate):
        raise RuntimeError("progress tool loop has no generation boundary to guard")

    @wraps(original_generate)
    def generate_with_semantic_boundary(*args: Any, **kwargs: Any):
        messages = kwargs.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            failure = _latest_post_argument_semantic_failure(messages)
            if failure is not None:
                code, reason = failure
                tool_choice = kwargs.get("tool_choice")
                if tool_choice is None and len(args) > 6:
                    tool_choice = args[6]
                if tool_choice is None and "request" in kwargs:
                    tool_choice = getattr(kwargs["request"], "tool_choice", None)
                elif tool_choice is None and len(args) > 3:
                    tool_choice = getattr(args[3], "tool_choice", None)

                forced = _forced_tool_name(tool_choice)
                is_host_directed = bool(forced or tool_choice == "required")

                if code == "PHASE_PROTOCOL_VIOLATION" and is_host_directed:
                    # The host explicitly scheduled this action for the updated phase;
                    # this is an authorized phase transition, not a repeated unguided retry.
                    pass
                else:
                    emit = getattr(loop_module, "emit_root_cause", None)
                    if callable(emit):
                        emit(
                            "post_argument_semantic_failure",
                            stage="generation",
                            operation="semantic_failure_boundary",
                            gate="tool_result_phase",
                            result="FAIL",
                            reason=code,
                            details={"failure_code": code, "error": reason},
                        )
                    raise loop_module.ModelConfigurationError(
                        "POST_ARGUMENT_SEMANTIC_FAILURE: "
                        f"{code}: {reason}; outer adjudication/replan required. "
                        "Refusing to regenerate tool arguments for an already-executed semantic rejection."
                    )
        return original_generate(*args, **kwargs)

    loop_module._generate_turn_with_context_recovery = generate_with_semantic_boundary
    setattr(loop_module, _SEMANTIC_BOUNDARY_MARKER, True)


def install(loop_module: Any | None = None) -> None:
    """Install after all runtime wrappers so authority cannot be weakened later."""

    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module

    if not getattr(loop_module, _MARKER, False):
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
            # Install the host pin *before* generic message scanning. This prevents a
            # retrieved/fixture file from becoming target identity merely because it is a
            # READY source snapshot and appears earlier in the execution flow.
            host_pin = _host_pin_from_messages(messages, loop_module)
            if host_pin is not None:
                with state._lock:
                    current = state.mutation_context
                    current_pinned = (
                        bool(getattr(current, "target_pinned", False)) if current else False
                    )
                    current_path = loop_module._canonical_mutation_path(
                        getattr(current, "target_path", "") if current else ""
                    )
                    host_path = loop_module._canonical_mutation_path(host_pin.target_path)
                    if not current_pinned or current_path != host_path:
                        state.mutation_context = host_pin
            ready = original_is_ready(messages, state)
            if host_pin is not None:
                with state._lock:
                    current = state.mutation_context
                    if current is not None:
                        current_path = loop_module._canonical_mutation_path(
                            getattr(current, "target_path", "")
                        )
                        host_path = loop_module._canonical_mutation_path(host_pin.target_path)
                        if current_path == host_path:
                            state.mutation_context = replace(
                                current,
                                writable_paths=tuple(
                                    getattr(host_pin, "writable_paths", ()) or ()
                                ),
                                creatable_paths=tuple(
                                    getattr(host_pin, "creatable_paths", ()) or ()
                                ),
                                target_pinned=True,
                            )
                        else:
                            state.mutation_context = host_pin
                    else:
                        state.mutation_context = host_pin
            return ready


        Context.merge = merge
        loop_module.is_mutation_ready = is_mutation_ready
        setattr(loop_module, _MARKER, True)

    _install_semantic_generation_boundary(loop_module)


def assert_installed(loop_module: Any | None = None) -> None:
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if not getattr(loop_module, _MARKER, False):
        raise RuntimeError("final mutation-authority guard is not installed")
    if not getattr(loop_module, _SEMANTIC_BOUNDARY_MARKER, False):
        raise RuntimeError("post-argument semantic failure boundary is not installed")


__all__ = ["assert_installed", "install"]
