from __future__ import annotations

"""Host-owned coder state machine; evidence policy lives in generation_evidence_controller."""

import hashlib
import json
import re
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import getattr_static
from pathlib import Path
from typing import Any

from .agent_intent import implementation_requested
from . import generation_compile_recovery as _compile_recovery
from .external_mcp_recovery_contract import (
    constrain_recovery_tools,
    record_discovery,
    recovery_state_snapshot,
)
from .generation_evidence_controller import (
    authoritative_java_evidence as _authoritative_java_evidence,
    authoritative_java_evidence_diagnostic as _authoritative_java_evidence_diagnostic,
    evidence_obligation_satisfied,
    initial_evidence_frontier,
    initial_evidence_required,
    normalize_forced_evidence_rejection_calls,
    normalize_recovery_evidence_calls,
    recovery_evidence_frontier,
    repair_evidence_route_for_errors,
    repair_route_requires_retrieval,
    semantic_fresh_java as _semantic_fresh_java,
)
from .generation_loop_outcomes import (
    MUTATION_ACT_TOOLS as _MUTATION_ACT_TOOLS,
    VERIFY_TOOLS as _VERIFY_TOOLS,
    fixed_point_tool_calls as _fixed_point_tool_calls,
    fixed_point_tool_results as _fixed_point_tool_results,
    runtime_failure_code as _runtime_failure_code,
    verification_outcome as _verification_outcome,
)
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    completion_boundary_error,
    completion_boundary_kind,
    mark_context_recovery_exhausted,
)
from .model_adapters import GenerationRequest, ModelConfigurationError
from .model_context_budget import (
    bounded_tool_message,
    emergency_fit_messages,
    fit_messages_to_context,
    request_message_budget,
)
from .mutation_authority import CURRENT_MUTATION_AUTHORITY, MutationAuthorityMode
from .mutation_context_binding import (
    context_is_host_pinned,
    context_is_localized,
    materialized_create_context,
    observed_context_may_bind,
    recover_stale_existing_context as _recover_stale,
)
from .root_cause_trace import emit_root_cause, trace_scope
from .retrieval_progress import (
    RetrievalDecision,
    RetrievalNoProgressError,
    RetrievalObservation,
    RetrievalProgress,
    _stable_value,
    evidence_fingerprint,
    normalize_retrieval_query,
    retrieval_query_signature,
    retrieval_source_key,
)
from .small_model_task_capsule_contract import task_capsule_tool_loop
from .source_mutation_contract import mutation_history_applied, mutation_payload_applied
from .source_repair_semantics import (
    atomic_repair_scope_error,
    existing_java_structurally_subsumes_candidate,
    existing_source_repair_semantic_error,
    java_path_package_error as _java_path_package_error,
    java_semantic_footprint_error as _java_semantic_footprint_error,
    java_source_identity_error as _java_source_identity_error,
)
from .target_mutation_context import (
    LocalizationStage,
    TargetMutationContext,
    _canonical_mutation_path,
    _extract_mutation_context_from_payload,
    _mutation_context_dict,
    _task_authority_context,
    _without_target_path,
)
from .verifier_repair_admission_recovery import (
    model_tool_rejection_feedback as _model_tool_rejection_feedback,
    recover_schema_rejected_host_bound_existing_calls,
    recover_schema_rejected_verifier_repair_calls,
)
from .verifier_repair_frontier import reject_noop_repair, target_scoped_verifier_files
from .verifier_repair_window import (
    exact_rollback_arguments,
    normalize_model_repair_replacement,
    repair_replacement_max_chars,
    select_verifier_repair_window,
    selected_repair_diagnostic,
)
from .value_shapes import structured_payload as _structured_payload

class LoopPhase(str, Enum):
    OBSERVE = "OBSERVE"
    ACT = "ACT"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"


_LOCALIZATION_EVIDENCE_TOOLS = frozenset({
    "search_code_rag",
    "search_project_rag",
    "java_workspace_symbols",
    "read_reuse_source",
})
_READ_OBSERVE_TOOLS = frozenset({
    "search_code_rag",
    "search_project_rag",
    "discover_ecosystem_resources",
    "inspect_modrinth_project",
    "inspect_github_repository",
    "inspect_huggingface_model",
    "assess_technology_compatibility",
    "java_workspace_symbols",
    "read_complete_plan_section",
    "read_quality_contract",
    "quality_status",
    "work_status",
    "work_tasks",
    "external_mcp_capabilities",
    "external_mcp_schema",
    "external_mcp_call",
    "read_reuse_source",
})
_RECOVERY_EVIDENCE_TOOLS = frozenset({
    "search_code_rag",
    "search_project_rag",
    "inspect_modrinth_project",
    "java_workspace_symbols",
    "external_mcp_call",
    "read_reuse_source",
})
_SOURCE_EDIT_PATH_KEYS = ("path", "file", "target_path", "target_file")
_SOURCE_CREATE_OPERATIONS = frozenset({
    "create", "create_file", "create_java_type", "create_class", "create_type",
    "write", "write_file",
})
_REPAIR_CONTEXT_PREFIX = "MMM_CORE_VERIFIER_REPAIR_"
_HOST_AUTHORITY_ROLES = frozenset({"system", "developer", "tool"})
_GENERATION_VERIFICATION_RECEIPT: ContextVar[dict[str, Any] | None] = ContextVar(
    "mmm_generation_verification_receipt",
    default=None,
)
def clear_generation_verification_receipt() -> None:
    _GENERATION_VERIFICATION_RECEIPT.set(None)
def current_generation_verification_receipt() -> dict[str, Any] | None:
    value = _GENERATION_VERIFICATION_RECEIPT.get()
    return deepcopy(value) if isinstance(value, dict) else None
def _record_terminal_generation_verification(
    state: HostRunState,
    *,
    terminal_status: str,
    compile_backed_java: bool,
) -> None:
    context = state.mutation_context
    verifier_tool = str(state.latest_verifier_tool or "").strip()
    _GENERATION_VERIFICATION_RECEIPT.set(
        {
            "schema_version": "mmm/generation-verification-v1",
            "status": terminal_status,
            "authority": "generation_tool_loop",
            "validation_status": str(state.validation_status or "").strip().upper(),
            "termination_reason": str(state.termination_reason or "").strip(),
            "verifier_tool": verifier_tool or None,
            "target_path": (
                str(context.target_path)
                if context is not None and str(context.target_path or "").strip()
                else None
            ),
            "compile_backed_java": bool(compile_backed_java),
            "downstream_required_gate": (
                "target_compile"
                if terminal_status == "DEFERRED_TO_TARGET_COMPILE"
                else None
            ),
        }
    )


def _tool_name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""

def _planir_owned_anchor_sets(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the exact host-owned writable/creatable PlanIR paths."""

    context = _task_authority_context(payload)
    if context is None:
        return (), ()
    return context.writable_paths, context.creatable_paths


def _constrain_existing_repair_schema(
    schema: Mapping[str, Any],
    *,
    target_path: str,
    repair_window: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Project verifier repair onto one host-bound bounded source-window replacement.

    The host selects the exact old source window from verifier location evidence.
    The small model authors only replacement text for that window; operation, path,
    old text, and optimistic concurrency remain host-owned.
    """

    cloned = deepcopy(dict(schema))
    function = cloned.get("function")
    if not isinstance(function, dict):
        raise ModelConfigurationError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no function schema"
        )
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        raise ModelConfigurationError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no parameter schema"
        )
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ModelConfigurationError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no properties schema"
        )
    new_schema = deepcopy(properties.get("new") or {"type": "string"})
    if not isinstance(new_schema, dict):
        raise ModelConfigurationError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no new-source schema"
        )
    old_text = repair_window.get("old") if repair_window else None
    max_chars = repair_replacement_max_chars(old_text)
    new_schema["type"] = "string"
    new_schema["maxLength"] = max_chars
    new_schema["description"] = (
        "Replacement text only for the verifier-selected bounded source window. "
        f"Emit at most {max_chars} characters. Never emit the complete source file. "
        f"The host binds operation=replace_exact and path={target_path!r}, supplies the "
        "exact old window, and executes against the live file."
    )
    parameters["properties"] = {"new": new_schema}
    parameters["required"] = ["new"]
    parameters["additionalProperties"] = False
    function["description"] = (
        "Repair exactly one verifier-selected source window. Emit only replacement "
        "text in new; operation, path, old text, count, and SHA are host-owned."
    )
    return cloned


def _bind_existing_verifier_repair_call(
    call: Any,
    state: Any,
) -> Any:
    """Bind model replacement text to the live host-selected verifier repair window."""

    if str(getattr(call, "name", "") or "").strip() != "apply_source_edit":
        return call
    if str(getattr(state, "validation_status", "") or "") != "FAIL":
        return call
    context = getattr(state, "mutation_context", None)
    if context is None or context.is_new_file or not context.is_mutation_ready:
        return call
    target = _canonical_mutation_path(context.target_path)
    repair_window = _repair_source_window(state)
    if not target or repair_window is None:
        return call
    raw_arguments = getattr(call, "arguments", None)
    if not isinstance(raw_arguments, Mapping):
        return call
    new_source = raw_arguments.get("new")
    old_source = repair_window.get("old")
    if not isinstance(new_source, str) or not isinstance(old_source, str) or not old_source:
        return call
    new_source = normalize_model_repair_replacement(
        getattr(context, "source_body", None),
        old_source,
        new_source,
    )
    bound = {
        "operation": "replace_exact",
        "path": target,
        "old": old_source,
        "new": new_source,
        "count": 1,
    }
    return replace(
        call,
        arguments=bound,
        raw_arguments=json.dumps(bound, ensure_ascii=False, separators=(",", ":")),
    )


def _constrain_verifier_repair_tools(
    tools: Sequence[Any],
    state: Any,
) -> tuple[Any, ...]:
    """Project a verifier repair directly from live host state.

    Repair authority already lives in HostRunState; re-parsing a system prompt to
    reconstruct that authority creates a second, stale control path.
    """

    if str(getattr(state, "validation_status", "") or "") != "FAIL":
        return tuple(tools)
    if _authored_implementation_recovery(state):
        return tuple(
            _source_edit_schema_for_context(
                schema, state.mutation_context, implementation_recovery=True
            )
            for schema in tools
        )
    context = getattr(state, "mutation_context", None)
    if context is None or context.is_new_file or not context.is_mutation_ready:
        return tuple(tools)
    target_path = _canonical_mutation_path(context.target_path)
    if not target_path:
        return tuple(tools)
    repair_window = _repair_source_window(state)
    if repair_window is None:
        raise ModelConfigurationError(
            "VERIFIER_REPAIR_LOCALIZATION_UNAVAILABLE: verifier repair requires one "
            "bounded exact host-selected source span; whole-file reconstruction is forbidden"
        )

    projected: list[Any] = []
    for schema in tools:
        if _tool_name(schema) != "apply_source_edit":
            projected.append(schema)
            continue
        if not isinstance(schema, Mapping):
            raise ModelConfigurationError(
                "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit schema is not a mapping"
            )
        projected.append(
            _constrain_existing_repair_schema(
                schema,
                target_path=target_path,
                repair_window=repair_window,
            )
        )
    return tuple(projected)


def _trusted_internal_user_payload(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("phase") or "").strip() != "implement_module":
        return False
    if str(payload.get("workspace_project_root") or "").strip() != ".":
        return False
    grounding = payload.get("host_grounding")
    if not isinstance(grounding, Mapping):
        return False
    policy = grounding.get("policy")
    return bool(
        isinstance(policy, Mapping)
        and policy.get("resolved_before_first_coder_decode") is True
        and policy.get("baseline_grounding_owned_by_host") is True
        and policy.get("writes_still_require_approved_pipeline") is True
    )


def _authority_allowed(role: str, payload: Mapping[str, Any]) -> bool:
    if role in _HOST_AUTHORITY_ROLES:
        return True
    if role != "user":
        return False
    if _trusted_internal_user_payload(payload):
        return True
    from .task_authority_transport import is_preserved_host_continuation

    return is_preserved_host_continuation(payload)


def _bind_authority_message(message: Any, state: HostRunState) -> None:
    if not isinstance(message, Mapping):
        return
    payload = _structured_payload(message.get("content"))
    if not isinstance(payload, Mapping):
        return
    role = str(message.get("role") or "").strip().casefold()
    if not _authority_allowed(role, payload):
        return
    context = _task_authority_context(payload)
    if context is None:
        return
    with state._lock:
        current = state.mutation_context
        if current is None or not current.target_pinned:
            state.mutation_context = context
            return
        if _canonical_mutation_path(current.target_path) == _canonical_mutation_path(context.target_path):
            state.mutation_context = current.merge(context)


def _strip_untrusted_owned_anchors(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_untrusted_owned_anchors(item)
            for key, item in value.items()
            if str(key) != "owned_anchors"
        }
    if isinstance(value, list):
        return [_strip_untrusted_owned_anchors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_untrusted_owned_anchors(item) for item in value)
    return value


def _bind_observed_message(message: Any, state: HostRunState) -> None:
    if not isinstance(message, Mapping):
        return
    payload = _structured_payload(message.get("content"))
    if payload is None:
        return
    role = str(message.get("role") or "").strip().casefold()
    if role not in _HOST_AUTHORITY_ROLES and not _trusted_internal_user_payload(payload):
        payload = _strip_untrusted_owned_anchors(payload)
    context = _extract_mutation_context_from_payload(payload)
    if not observed_context_may_bind(context, binding_enabled=state.retrieval_target_binding_enabled):
        return
    with state._lock:
        if state.mutation_context is None:
            state.mutation_context = context
            return
        state.mutation_context = state.mutation_context.merge(context)


def is_mutation_ready(messages: Sequence[Mapping[str, Any]], state: HostRunState) -> bool:
    for message in messages:
        _bind_authority_message(message, state)
    for message in messages:
        _bind_observed_message(message, state)
    with state._lock:
        context = state.mutation_context
        return bool(context and context.is_mutation_ready)


def _reconcile_materialized_target_from_workspace(
    state: HostRunState,
    runtime: Any,
) -> TargetMutationContext | None:
    """Refresh the exact pinned target from the staged workspace when it exists."""

    root_value = getattr(runtime, "workspace_root", None)
    if root_value in (None, ""):
        return None
    with state._lock:
        context = state.mutation_context
        if context is None or not context.target_pinned:
            return None
        target = _canonical_mutation_path(context.target_path)
    if not target or not target.casefold().endswith((".java", ".kt")):
        return None

    try:
        root = Path(str(root_value)).expanduser().resolve()
        candidate = (root / target).resolve()
        candidate.relative_to(root)
        raw_source = candidate.read_bytes()
        source = raw_source.decode("utf-8")
        source_sha256 = hashlib.sha256(raw_source).hexdigest()
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return None

    with state._lock:
        current = state.mutation_context
        if (
            current is None
            or _canonical_mutation_path(current.target_path) != target
        ):
            return None
        _compile_recovery.capture_baseline(state, target, source)
        reconciled = replace(
            current,
            source_body=source,
            is_new_file=False,
            evidence_source="workspace_existing_target",
            base_revision_sha=source_sha256,
            creatable_paths=_without_target_path(current.creatable_paths, target),
        )
        state.mutation_context = reconciled
        state.created_paths.add(target)
        return reconciled


_HOST_BOUND_EXISTING_REWRITE_MAX_BYTES = 12 * 1024


def _host_bound_existing_rewrite_ready(
    context: TargetMutationContext | None, *, implementation_recovery: bool = False
) -> bool:
    """Whether the host can hide exact existing-file preconditions from the model."""

    if (
        context is None
        or context.is_new_file
        or not context.is_mutation_ready
        or context.evidence_source not in (
            {"workspace_existing_target", "mutation_receipt", "verifier_workspace_source"}
            if implementation_recovery else {"workspace_existing_target"}
        )
        or not isinstance(context.source_body, str)
        or not context.source_body
    ):
        return False
    authority = CURRENT_MUTATION_AUTHORITY.get()
    if authority is None:
        return False
    target = _canonical_mutation_path(context.target_path)
    if authority.mode is MutationAuthorityMode.EXACT:
        if target not in authority.paths:
            return False
    elif authority.mode is not MutationAuthorityMode.BOUNDED_ROOTS:
        return False
    return (
        len(context.source_body.encode("utf-8"))
        <= _HOST_BOUND_EXISTING_REWRITE_MAX_BYTES
    )


def _authored_implementation_recovery(state: Any) -> bool:
    """An unimplemented fresh host slot needs its implementation, not a line repair.

    Only trusted target-compile findings for the exact scaffold can select this
    mode. An unfinished implementation can also have host integration defects;
    those do not turn its remaining implementation into a bounded line repair.
    Compiler/API failures and side-only failures without an unfinished-body
    finding retain ordinary bounded repair.
    """
    context = getattr(state, "mutation_context", None)
    baseline = _compile_recovery.trusted_baseline(state, context)
    if (
        getattr(state, "validation_status", None) != "FAIL"
        or getattr(state, "latest_verifier_tool", None) != "target_compile"
        or not getattr(state, "semantic_fresh_java", False)
        or not baseline
        or "MMM_AUTHORED_FEATURE_BODY_" not in baseline
        or not _host_bound_existing_rewrite_ready(context, implementation_recovery=True)
    ):
        return False
    diagnostics = getattr(state, "latest_verifier_errors", ())
    unfinished_codes = {"host:authored-placeholder", "host:authored-empty"}
    integration_codes = unfinished_codes | {
        "host:authored-side-only", "host:authored-surface",
    }
    return bool(diagnostics) and all(
        isinstance(item, Mapping)
        and item.get("source") == "host-authored-contract"
        and item.get("code") in integration_codes
        and _canonical_mutation_path(item.get("path", ""))
        == _canonical_mutation_path(context.target_path)
        for item in diagnostics
    ) and any(item.get("code") in unfinished_codes for item in diagnostics)


def _bind_host_owned_existing_source_call(
    call: Any,
    state: Any,
) -> Any:
    """Bind exact live source preconditions for any host-owned existing target.

    The small coder supplies only the desired updated source. Exact path selection,
    operation type, and the current source precondition are already host-owned facts and
    must not be copied back through model text.
    """

    if str(getattr(call, "name", "") or "").strip() != "apply_source_edit":
        return call
    implementation_recovery = _authored_implementation_recovery(state)
    if str(getattr(state, "validation_status", "") or "") == "FAIL" and not implementation_recovery:
        # Verifier repair has a stricter bounded-window binder of its own.
        return call
    context = getattr(state, "mutation_context", None)
    if not _host_bound_existing_rewrite_ready(context, implementation_recovery=implementation_recovery):
        return call
    raw_arguments = getattr(call, "arguments", None)
    if not isinstance(raw_arguments, Mapping):
        return call
    new_source = raw_arguments.get("new")
    if not isinstance(new_source, str) or not new_source:
        return call
    target = _canonical_mutation_path(context.target_path)
    old_source = context.source_body
    if not target or not isinstance(old_source, str) or not old_source:
        return call
    model_old = raw_arguments.get("old")
    if isinstance(model_old, str) and model_old and model_old != old_source:
        if old_source.count(model_old) != 1:
            return call
        # Non-validating/legacy adapters can still echo an exact old/new span even
        # after the visible schema has been reduced to {"new"}. Merge that span
        # into the host-owned live source before constructing the transactional
        # whole-source replacement. This keeps package/type identity intact without
        # trusting the model for path or precondition authority.
        new_source = old_source.replace(model_old, new_source, 1)
    bound = {
        "operation": "replace_exact",
        "path": target,
        "old": old_source,
        "new": new_source,
        "count": 1,
    }
    return replace(
        call,
        arguments=bound,
        raw_arguments=json.dumps(bound, ensure_ascii=False, separators=(",", ":")),
    )


def _existing_target_refresh_message(context: TargetMutationContext) -> dict[str, str]:
    source = context.source_body or ""
    return {
        "role": "system",
        "content": (
            "MMM_EXISTING_TARGET_REFRESH_V1\n"
            "The host found that this exact task-owned target was already materialized in the "
            "current staged workspace by an earlier atomic step or resumed checkpoint. Treat it "
            "as an existing file, not a fresh creation target. Work from the exact current source "
            "below and make only the current obligation's required delta.\n"
            f"TARGET_PATH={_canonical_mutation_path(context.target_path)}\n"
            "CURRENT_SOURCE_BEGIN\n"
            + source
            + "\nCURRENT_SOURCE_END"
        ),
    }


def _recover_creation_conflict_target(
    state: HostRunState,
    runtime: Any,
    arguments: Mapping[str, Any],
) -> TargetMutationContext | None:
    """Turn a failed create into an exact existing-file edit target.

    Bounded authored-design generation deliberately lets the coder propose a path inside
    host-owned roots. If that path was materialized by an earlier atomic fragment, asking
    a small model to rediscover the file through RAG is both redundant and fragile. The
    failed mutation already supplies the exact authorized path, so the host reads that
    file directly and pins the next ACT turn to its current source.
    """

    path = _source_edit_path(arguments)
    root = getattr(runtime, "workspace_root", None)
    if not path or root in (None, ""):
        return None

    with state._lock:
        current = state.mutation_context
        if current is not None and current.target_pinned:
            current_path = _canonical_mutation_path(current.target_path)
            if current_path and current_path != path:
                return None

    refreshed = _recover_stale(
        state,
        root,
        path,
        None,
        TargetMutationContext,
    )
    if refreshed is None:
        return None

    # The action space changed from "create a new file" to "edit this exact live file".
    # A failed-create fingerprint must not poison convergence for the corrective edit.
    with state._lock:
        state.unchanged_mutation_fingerprints.clear()
        state.unapplied_mutation_fixed_point = False
        state.semantic_fixed_point = False
    return refreshed


def _source_edit_path(arguments: Mapping[str, Any]) -> str:
    for key in _SOURCE_EDIT_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _canonical_mutation_path(value)
    return ""


def _creation_authorized(
    supplied: str,
    pinned: str,
    context: TargetMutationContext,
) -> bool:
    return bool(
        context.is_new_file
        and supplied == pinned
        and supplied in {
            _canonical_mutation_path(path)
            for path in context.creatable_paths
            if _canonical_mutation_path(path)
        }
    )


def _creation_authorized_for_context(supplied: str, pinned: str, context: TargetMutationContext, authority: Any) -> bool:
    return _creation_authorized(supplied, pinned, context) or bool(authority is not None and authority.mode is MutationAuthorityMode.EXACT and not context_is_host_pinned(context) and authority.authorizes(supplied, operation="create_file"))


def _contextless_authority_error(authority: Any, arguments: Mapping[str, Any], operation: str) -> str | None:
    if authority is None:
        return "MUTATION_TARGET_UNBOUND: no host-pinned mutation target is READY"
    error = authority.mutation_error(_source_edit_path(arguments), operation=arguments.get("operation"))
    if error is not None:
        return error
    if authority.mode is MutationAuthorityMode.BOUNDED_ROOTS:
        supplied = _source_edit_path(arguments)
        if operation in _SOURCE_CREATE_OPERATIONS and supplied.casefold().endswith(".java"):
            create_source = arguments.get("content")
            if not isinstance(create_source, str):
                create_source = arguments.get("text")
            package_error = _java_path_package_error(supplied, create_source)
            if package_error is not None:
                return package_error
    if operation == "replace_exact" and "old" not in arguments:
        return "MUTATION_ATOMIC_SPAN_REQUIRED: existing source replacement requires one exact old span; whole-file model replacement is forbidden"
    return None


def _repair_phase_for_route(route: str | None) -> LoopPhase:
    return LoopPhase.RECOVER if repair_route_requires_retrieval(route) else LoopPhase.ACT

def _resume_local_repair(state: Any) -> LoopPhase:
    return _repair_phase_for_route(state.repair_evidence_route) if state.phase is LoopPhase.RECOVER and state.validation_status == "FAIL" else state.phase

def _mutation_target_error(tool_name: str, arguments: Mapping[str, Any], context: TargetMutationContext | None, *, state: Any = None) -> str | None:
    if tool_name != "apply_source_edit":
        return None
    operation = str(arguments.get("operation") or "").strip().casefold()
    authority = CURRENT_MUTATION_AUTHORITY.get()
    if context is None:
        return _contextless_authority_error(authority, arguments, operation)
    supplied = _source_edit_path(arguments)
    pinned = _canonical_mutation_path(context.target_path)
    creation_authorized = _creation_authorized_for_context(supplied, pinned, context, authority)
    if authority is not None:
        error = authority.mutation_error(_source_edit_path(arguments), operation=arguments.get("operation"))
        if error is not None:
            return error
        if authority.mode is MutationAuthorityMode.BOUNDED_ROOTS and context.evidence_source != "verifier_workspace_source":
            if operation in _SOURCE_CREATE_OPERATIONS and supplied.casefold().endswith(".java"):
                create_source = arguments.get("content")
                if not isinstance(create_source, str):
                    create_source = arguments.get("text")
                package_error = _java_path_package_error(supplied, create_source)
                if package_error is not None:
                    return package_error
            # Once an existing target is pinned, exact source semantics remain host-owned.
            if context.is_new_file or not context_is_host_pinned(context):
                if operation == "replace_exact" and "old" not in arguments:
                    return (
                        "MUTATION_ATOMIC_SPAN_REQUIRED: existing source replacement requires "
                        "one exact old span; whole-file model replacement is forbidden"
                    )
                return None
    allowed = set(context.writable_paths) or ({pinned} if pinned else set())
    if not supplied:
        return "MUTATION_TARGET_UNBOUND: apply_source_edit requires an explicit host-bound path"
    if not pinned:
        return "MUTATION_TARGET_UNBOUND: apply_source_edit requires an explicit host-bound path"
    if supplied not in allowed:
        return (
            f"MUTATION_TARGET_DRIFT: writable exact-set {sorted(allowed)!r} "
            f"does not authorize {supplied!r}"
        )
    if operation in _SOURCE_CREATE_OPERATIONS and not creation_authorized:
        return (
            "MUTATION_TARGET_CREATION_CONFLICT: create operation is not authorized "
            f"for existing target {supplied!r}"
        )
    if not context.is_mutation_ready:
        return "MUTATION_TARGET_UNBOUND: no host-pinned mutation target is READY"
    if operation == "replace_exact" and "old" not in arguments:
        return (
            "MUTATION_ATOMIC_SPAN_REQUIRED: existing source replacement requires "
            "one exact old span; whole-file model replacement is forbidden"
        )
    semantic_error = existing_source_repair_semantic_error(
        operation=operation,
        supplied=supplied,
        pinned=pinned,
        is_new_file=creation_authorized,
        current_source=context.source_body,
        old_text=arguments.get("old"),
        new_text=arguments.get("new"),
        identity_check=_java_source_identity_error,
        footprint_check=_java_semantic_footprint_error,
        semantic_baseline_source=_compile_recovery.trusted_baseline(state, context),
    )
    if semantic_error is not None:
        return semantic_error
    if (
        operation == "replace_exact"
        and context.evidence_source == "verifier_workspace_source"
        and not _authored_implementation_recovery(state)
    ):
        old_text = arguments.get("old")
        atomic_error = atomic_repair_scope_error(
            old_text=old_text,
            new_text=arguments.get("new"),
            max_chars=repair_replacement_max_chars(old_text),
        )
        if atomic_error is not None:
            return atomic_error
    if operation not in _SOURCE_CREATE_OPERATIONS:
        return None
    if creation_authorized:
        return None
    return (
        "MUTATION_TARGET_CREATION_CONFLICT: create operation is not authorized "
        f"for existing target {supplied!r}"
    )


def _atomic_output_recovery_instruction(request: GenerationRequest) -> str:
    names = frozenset(_tool_name(schema) for schema in request.tools if _tool_name(schema))
    if "apply_source_edit" in names:
        for schema in request.tools:
            if _tool_name(schema) != "apply_source_edit":
                continue
            function = schema.get("function") if isinstance(schema, Mapping) else None
            parameters = function.get("parameters") if isinstance(function, Mapping) else None
            properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
            if isinstance(properties, Mapping) and set(properties) == {"new"}:
                return (
                    "The preceding host-bound source-edit output exceeded the bounded allowance and "
                    "is discarded. Call apply_source_edit exactly once with no prose and emit only "
                    "replacement text or the requested complete updated source in the visible new "
                    "argument, according to the active schema. For a bounded verifier repair window, "
                    "replacement text is the window only; never emit the complete source file for "
                    "that repair. Do not emit operation, path, old text, count, anchors, or any "
                    "additional tool call; the host binds all exact mutation preconditions."
                )
            break
        return (
            "The preceding assistant action exceeded the bounded output allowance and is discarded. "
            "Do not continue, reproduce, or complete that oversized payload. The host will preserve the "
            "same mutation target and workspace state. Use exactly one visible source-mutation tool: "
            "call apply_source_edit exactly once with no prose and make one small semantic edit. "
            "For a fresh host-pinned Java target, use operation=create_file and provide one complete Java "
            "file that is minimal and compilable at the already-authorized path. For an existing target, "
            "use exactly one bounded replace/insert operation; never reconstruct the complete file. "
            "Do not invent Java mutation tools that are not visible in this turn."
        )
    if names & _MUTATION_ACT_TOOLS:
        return (
            "The preceding assistant action exceeded the bounded output allowance and is discarded. "
            "Do not continue or reconstruct that oversized payload. Call exactly one visible mutation "
            "tool now, perform only the current host-pinned semantic action, and emit no prose."
        )
    return (
        "The preceding assistant action exceeded the bounded output allowance and is discarded. "
        "Do not continue that oversized payload. Produce exactly one concise visible tool call or one "
        "concise final answer using the already-grounded state; do not emit a long reconstruction."
    )


_ACT_REDUNDANT_SYSTEM_PREFIXES = (
    "Host research context follows.",
    "MMM reviewed Skill/tool/Minecraft-MCP routing context:",
)


def _forced_act_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    state: Any,
    require_rag: bool,
    phase_names: Collection[str],
    evidence_ready: bool | None = None,
) -> list[dict[str, Any]]:
    """Project a forced mutation turn onto only execution-relevant context."""

    names = frozenset(str(name).strip() for name in phase_names if str(name).strip())
    if evidence_ready is None:
        evidence_ready = not require_rag
    forced_mutation = (
        getattr(state, "phase", None) == LoopPhase.ACT
        and bool(evidence_ready)
        and len(names) == 1
        and bool(names & _MUTATION_ACT_TOOLS)
    )
    if not forced_mutation:
        return [dict(message) for message in messages]

    projected: list[dict[str, Any]] = []
    for raw in messages:
        message = dict(raw)
        content = message.get("content")
        if (
            str(message.get("role") or "").strip().casefold() == "system"
            and isinstance(content, str)
            and content.startswith(_ACT_REDUNDANT_SYSTEM_PREFIXES)
        ):
            continue
        projected.append(message)

    context = getattr(state, "mutation_context", None)
    target = _canonical_mutation_path(getattr(context, "target_path", ""))
    fresh_java = bool(
        context is not None
        and getattr(context, "is_new_file", False)
        and target.casefold().endswith(".java")
    )
    repair_window = _repair_source_window(state)
    active_authority = CURRENT_MUTATION_AUTHORITY.get()
    bounded_roots = (
        tuple(active_authority.roots)
        if (
            active_authority is not None
            and active_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
        )
        else ()
    )
    if _authored_implementation_recovery(state):
        directive = (
            f"HOST FORCED ACT: {target!r} is still an unimplemented authored feature. "
            "Implement the approved task requirements in this exact file. Emit the complete "
            "updated source in new, including required imports, state and helpers. Merely "
            "removing or renaming the placeholder is not implementation. Preserve package/type "
            "identity and approved behavior. The host binds the exact path and live old source; "
            "the next mutation must pass target_compile before completion."
        )
    elif (
        context is not None
        and getattr(state, "validation_status", "") == "FAIL"
        and repair_window is not None
    ):
        directive = (
            f"HOST FORCED ACT: repair exactly one verifier-selected bounded source window in "
            f"{target!r}. The host already owns the exact old text, path, count, and live SHA. "
            "Emit only replacement text in the single visible new argument. Never regenerate the "
            "complete file. Preserve approved behavior and do not restart generation or retrieve."
        )
    elif context is not None and getattr(state, "validation_status", "") == "FAIL":
        directive = (
            f"HOST FORCED ACT: repair the verified defect in {target!r} with exactly one bounded "
            "existing-file edit from the visible schema. Never regenerate the complete file, "
            "never change paths, and do not restart generation or retrieve."
        )
    elif _host_bound_existing_rewrite_ready(context):
        directive = (
            f"HOST FORCED ACT: {target!r} is an exact existing authored target whose live source "
            "is already host-owned. Call apply_source_edit exactly once with no prose and emit "
            "only the complete updated source in the visible new argument. Preserve unrelated "
            "valid behavior and the existing package/type identity. Do not emit operation, path, "
            "old text, count, or retrieval calls; the host binds those exact values."
        )
    elif bounded_roots:
        directive = (
            "HOST FORCED ACT: this saved authored design has host-owned bounded-root write "
            "authority. Choose exactly one project-relative file below one of these roots: "
            f"{list(bounded_roots)!r}. Call the single visible mutation tool exactly once with "
            "no prose. Create or edit only the file needed for the current authored-design "
            "fragment; deletes, build files, host state, retrieval, replanning, narration, and "
            "multi-file payloads are forbidden in this turn."
        )
    elif fresh_java:
        directive = (
            "HOST FORCED ACT: target localization, write authority, and evidence policy are already "
            f"resolved for {target!r}. Call the single visible mutation tool exactly once with no prose. "
            "Use create_file and emit one complete minimal compilable Java source for this task. "
            "Do not perform retrieval, planning, narration, or multi-step file construction."
        )
    else:
        directive = (
            "HOST FORCED ACT: target localization, write authority, and evidence policy are already "
            "resolved. Call the single visible mutation tool exactly once with no prose and perform "
            "only the current host-pinned edit."
        )
    projected.append({"role": "system", "content": directive})
    return projected


def _model_rejection_progress_key(
    state: Any,
    rejection_payloads: Sequence[Mapping[str, Any]],
    *,
    forced_evidence_tool: str | None,
) -> Mapping[str, Any]:
    """Return a stable no-progress key for model tool-admission failures.

    When the host has forced exactly one evidence route, varying rejected query text
    must not reset convergence. The route and rejection class are the semantic state;
    raw model arguments are incidental noise.
    """

    base: dict[str, Any] = {
        "phase": state.phase.value,
        "validation": state.validation_status,
        "verifier": state.latest_verifier_fingerprint,
    }
    forced = str(forced_evidence_tool or "").strip()
    if not forced:
        base["model_tool_rejections"] = list(rejection_payloads)
        return base
    base["forced_evidence_tool"] = forced
    base["rejection_codes"] = sorted({
        str(payload.get("failure_code") or "MODEL_TOOL_CALL_REJECTED").strip()
        for payload in rejection_payloads
    })
    return base


@dataclass(frozen=True)
class ExecutionStepTrace:
    step_index: int
    phase_before: str
    localization_stage_before: str
    mutation_context_before: dict[str, Any] | None
    exposed_tools: list[str]
    tool_choice: Any
    input_messages_count: int
    model_response_content: str | None
    model_tool_calls: list[dict[str, Any]]
    query_signatures: list[str]
    tool_results: list[dict[str, Any]]
    mutation_context_after: dict[str, Any] | None
    localization_stage_after: str
    phase_after: str
    turn_made_progress: bool
    no_progress_streak_after: int
    action_decision: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def format_trajectory_summary(trajectory: Sequence[ExecutionStepTrace]) -> str:
    lines: list[str] = []
    for item in trajectory[-8:]:
        calls = ", ".join(call.get("name", "") for call in item.model_tool_calls) or "<none>"
        results = ", ".join(
            f"{result.get('name')}:{'OK' if result.get('ok') else result.get('failure_code') or 'FAIL'}"
            for result in item.tool_results
        ) or "<none>"
        lines.append(
            f"Step {item.step_index} [{item.phase_before}:{item.localization_stage_before} -> "
            f"{item.phase_after}:{item.localization_stage_after}] "
            f"calls={calls} results={results} progress={item.turn_made_progress} "
            f"streak={item.no_progress_streak_after}"
        )
    return "\n".join(lines)


_ATOMIC_OUTPUT_RECOVERY_MARKER = "MMM_ATOMIC_OUTPUT_RECOVERY_V1"
_ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS = 4096
_VERIFIER_REPAIR_OUTPUT_TOKENS = 2048


def _state_requires_authoritative_java_evidence(state: Any) -> bool:
    if getattr(state, "repair_evidence_route", None) == "official_api":
        return True
    explicit = getattr(state, "semantic_fresh_java", None)
    if explicit is not None:
        return bool(explicit)
    context = getattr(state, "mutation_context", None)
    return bool(
        context is not None
        and getattr(context, "is_new_file", False)
        and _canonical_mutation_path(getattr(context, "target_path", ""))
        .casefold()
        .endswith(".java")
    )


def _host_target_execution_authority(state: Any) -> bool:
    """Return whether the host has exact authority to mutate the pinned target.

    Fresh host-reserved targets are executable when the exact pinned path is also in
    the host-owned creatable set. Once that path is materialized by an applied create
    mutation, the same authority remains valid for verification/repair of that file.
    """
    context = state.mutation_context
    if context is None or not context.target_pinned or not context.is_mutation_ready:
        return False
    target = _canonical_mutation_path(context.target_path)
    if not target:
        return False
    if context.is_new_file:
        creatable = {
            _canonical_mutation_path(path)
            for path in context.creatable_paths
            if _canonical_mutation_path(path)
        }
        return target in creatable

    writable = {
        _canonical_mutation_path(path)
        for path in context.writable_paths
        if _canonical_mutation_path(path)
    }
    if (
        target in writable
        and isinstance(context.source_body, str)
        and context.source_body.strip()
    ):
        # target_pinned + writable_paths can only originate from host-owned authority
        # binding. The source's provenance label may change as exact source, workspace
        # refresh, and verifier repair contexts merge; do not re-run retrieval solely
        # because that descriptive label changed.
        return True
    return target in state.created_paths


def _authored_workspace_refresh_requested(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether a later authored fragment must inspect the live staged workspace."""

    for message in reversed(messages):
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("phase") or "").strip() != "implement_authored_design":
            continue
        source_context = payload.get("initial_exact_source_context")
        execution = payload.get("authored_execution")
        if not isinstance(source_context, Mapping) or not isinstance(execution, Mapping):
            return False
        try:
            fragment_index = int(execution.get("fragment_index") or 0)
        except (TypeError, ValueError):
            return False
        return bool(
            fragment_index > 1
            and str(source_context.get("mode") or "").strip()
            == "retrieve_current_authored_fragment_with_tools"
        )
    return False


def _target_evidence_ready(
    state: HostRunState,
    *,
    require_rag: bool,
    fresh_java_target: bool,
    compile_backed_java: bool = False,
) -> bool:
    """Decide whether the current evidence obligation is satisfied.

    Both initial fresh-task policy and concrete repair obligations require the
    controller-approved evidence class. A target compiler is not API evidence.
    """

    del compile_backed_java
    return evidence_obligation_satisfied(
        require_evidence=bool(
            require_rag or repair_route_requires_retrieval(state.repair_evidence_route)
        ),
        semantic_fresh_java_target=bool(
            fresh_java_target
            or getattr(state, "semantic_fresh_java", False) is True
            or state.repair_evidence_route == "official_api"
        ),
        has_fresh_evidence=state.has_fresh_evidence,
        has_authoritative_java_evidence=state.has_authoritative_java_evidence,
    )

def _record_evidence_locked(state: Any, value: Any, fingerprint: str) -> bool:
    if fingerprint in state.evidence_fingerprints:
        return False
    state.evidence_fingerprints.add(fingerprint)
    context = state.mutation_context
    if (
        _state_requires_authoritative_java_evidence(state)
        and _authoritative_java_evidence(
            value,
            target_path=context.target_path if context is not None else None,
        )
    ):
        state.authoritative_java_evidence_fingerprints.add(fingerprint)
    context = _extract_mutation_context_from_payload(value)
    if observed_context_may_bind(context, binding_enabled=state.retrieval_target_binding_enabled):
        if state.mutation_context is None:
            state.mutation_context = context
        else:
            state.mutation_context = state.mutation_context.merge(context)
    return True


def _record_unapplied_mutation(state: Any, signature: str) -> bool:
    if not signature:
        return False
    repeated = signature in state.unchanged_mutation_fingerprints
    state.unchanged_mutation_fingerprints.add(signature)
    if repeated:
        state.unapplied_mutation_fixed_point = True
        state.semantic_fixed_point = True
    return False


def _mutation_operation(arguments: Mapping[str, Any]) -> str:
    return str(arguments.get("operation") or "").strip().casefold()


def _mutated_source_body(
    body: str | None,
    operation: str,
    arguments: Mapping[str, Any],
) -> str | None:
    content = arguments.get("content")
    if operation in _SOURCE_CREATE_OPERATIONS and isinstance(content, str):
        return content
    if operation != "replace_exact":
        return body
    new = arguments.get("new")
    if not isinstance(new, str):
        return body
    old = arguments.get("old")
    if old is None:
        # Compatibility for host/internal full replacements outside model-facing
        # verifier repair. Verifier repair itself always supplies an exact old span.
        return new
    if not isinstance(body, str) or not isinstance(old, str):
        return body
    if not old or body.count(old) != 1:
        return body
    return body.replace(old, new, 1)


def _update_mutation_context_after_edit(
    state: Any,
    path: str,
    operation: str,
    arguments: Mapping[str, Any],
) -> None:
    context = state.mutation_context
    if context is None:
        state.mutation_context = materialized_create_context(path, operation, arguments, _SOURCE_CREATE_OPERATIONS, TargetMutationContext)
        return
    if path != _canonical_mutation_path(context.target_path):
        return
    body = _mutated_source_body(context.source_body, operation, arguments)
    remaining = _without_target_path(context.creatable_paths, path)
    state.mutation_context = replace(
        context,
        source_body=body,
        is_new_file=False,
        evidence_source="mutation_receipt",
        creatable_paths=remaining,
    )


def _record_applied_mutation(
    state: Any,
    tool_name: str,
    arguments: Mapping[str, Any],
    signature: str,
) -> bool:
    context = state.mutation_context
    operation = _mutation_operation(arguments)
    path = _source_edit_path(arguments)
    if (
        state.validation_status == "FAIL"
        and context is not None
        and path
        and path == _canonical_mutation_path(context.target_path)
        and operation == "replace_exact"
    ):
        state.repair_baseline_error_count = len(state.latest_verifier_errors)
        state.repair_baseline_errors = tuple(state.latest_verifier_errors)
        state.repair_baseline_target_diagnostics = tuple(
            state.repair_target_diagnostics
        )
        state.repair_baseline_fingerprint = state.latest_verifier_fingerprint
        state.repair_previous_source = context.source_body
        state.repair_previous_path = path
        state.last_verifier_quality = None
    else:
        state.repair_baseline_error_count = None
        state.repair_baseline_errors = ()
        state.repair_baseline_target_diagnostics = ()
        state.repair_baseline_fingerprint = None
        state.repair_previous_source = None
        state.repair_previous_path = None
        state.last_verifier_quality = None
    if signature:
        state.mutation_fingerprints.add(signature)
    state.unchanged_mutation_fingerprints.clear()
    state.unapplied_mutation_fixed_point = False
    state.semantic_fixed_point = False
    state.applied_mutations.append(tool_name)
    state.workspace_changed = True
    state.validation_status = "PENDING"
    state.repair_evidence_route = None
    state.latest_verifier_tool = None
    state.latest_verifier_errors = ()
    state.repair_target_diagnostics = ()
    state.latest_verifier_fingerprint = None
    state.repair_guidance_fingerprint = None
    if path and operation in _SOURCE_CREATE_OPERATIONS:
        state.created_paths.add(path)
    _update_mutation_context_after_edit(state, path, operation, arguments)
    return True


def _implementation_obligation_has_progress(state: Any) -> bool:
    return bool(
        getattr(state, "workspace_changed", False)
        or getattr(state, "preserved_existing_source", False)
    )


def _repair_source_window(state: Any) -> dict[str, Any] | None:
    """Select one bounded, exact, host-owned source window for verifier repair."""

    if _authored_implementation_recovery(state):
        return None
    context = getattr(state, "mutation_context", None)
    source = getattr(context, "source_body", None)
    if not isinstance(source, str) or not source:
        return None
    diagnostics = tuple(
        getattr(state, "repair_target_diagnostics", ())
        or getattr(state, "latest_verifier_errors", ())
        or ()
    )
    return select_verifier_repair_window(
        source,
        diagnostics,
        start_line=getattr(context, "start_line", None),
        end_line=getattr(context, "end_line", None),
    )


def _repair_guidance_payload(state: Any) -> dict[str, Any] | None:
    if state.validation_status != "FAIL":
        return None
    if not state.latest_verifier_fingerprint:
        return None
    if state.latest_verifier_fingerprint == state.repair_guidance_fingerprint:
        return None
    state.repair_guidance_fingerprint = state.latest_verifier_fingerprint
    context = state.mutation_context
    source = context.source_body if context and isinstance(context.source_body, str) else None
    repair_window = _repair_source_window(state)
    target_diagnostics = tuple(state.repair_target_diagnostics)
    selected_diagnostic = selected_repair_diagnostic(
        target_diagnostics,
        repair_window,
    )
    selected_errors = (
        target_diagnostics
        if _authored_implementation_recovery(state)
        else (selected_diagnostic,) if selected_diagnostic is not None else ()
    )
    diagnostic_snapshot = (
        json.loads(_bounded_verifier_recovery_observation(
            state,
            errors=selected_errors,
            budget_bytes=_REPAIR_GUIDANCE_VERIFIER_DIAGNOSTIC_BYTES,
        ))
        if context and selected_errors
        else None
    )
    return {
        "verifier": state.latest_verifier_tool,
        "diagnostics": (
            diagnostic_snapshot["diagnostics"]
            if diagnostic_snapshot else list(state.latest_verifier_errors)
        ),
        **({
            "omitted_diagnostic_count": diagnostic_snapshot["omitted_diagnostic_count"],
            "omitted_target_diagnostic_count": max(
                0,
                len(target_diagnostics) - len(selected_errors),
            ),
            "other_file_diagnostic_count": (
                len(state.latest_verifier_errors) - len(target_diagnostics)
            ),
        } if diagnostic_snapshot else {}),
        "target_path": context.target_path if context else None,
        "target_is_new_file": context.is_new_file if context else None,
        "writable_paths": list(context.writable_paths) if context else [],
        "repair_window": repair_window,
        "repair_window_policy": (
            "host_owned_authored_implementation_required"
            if _authored_implementation_recovery(state)
            else "host_selected_bounded_exact_span_required"
        ),
        "current_source_chars": len(source) if isinstance(source, str) else 0,
        "current_source_sha256": (
            hashlib.sha256(source.encode("utf-8")).hexdigest()
            if isinstance(source, str)
            else None
        ),
    }


@dataclass
class HostRunState:
    phase: LoopPhase = LoopPhase.OBSERVE
    step_index: int = 0
    no_progress_streak: int = 0
    seen_no_progress_digests: set[str] = field(default_factory=set)
    no_progress_digest_first_step: dict[str, int] = field(default_factory=dict)
    semantic_fixed_point: bool = False
    fixed_point_digest: str | None = None
    fixed_point_first_seen_step: int | None = None
    fixed_point_repeat_step: int | None = None
    fixed_point_snapshot: dict[str, Any] | None = None
    attempted_queries: set[str] = field(default_factory=set)
    attempted_sources: set[str] = field(default_factory=set)
    evidence_fingerprints: set[str] = field(default_factory=set)
    authoritative_java_evidence_fingerprints: set[str] = field(default_factory=set)
    semantic_fresh_java: bool | None = None
    require_evidence: bool = False
    host_grounded: bool = False
    compile_backed_java: bool = False
    evidence_adjudications: list[dict[str, Any]] = field(default_factory=list)
    recovery_evidence_epoch_fingerprint: str | None = None
    retrieval_target_binding_enabled: bool = True
    mutation_context: TargetMutationContext | None = None
    applied_mutations: list[str] = field(default_factory=list)
    mutation_fingerprints: set[str] = field(default_factory=set)
    unchanged_mutation_fingerprints: set[str] = field(default_factory=set)
    unapplied_mutation_fixed_point: bool = False
    created_paths: set[str] = field(default_factory=set)
    workspace_changed: bool = False
    preserved_existing_source: bool = False
    validation_status: str = "PENDING"
    latest_verifier_tool: str | None = None
    latest_verifier_errors: tuple[dict[str, Any], ...] = ()
    repair_target_diagnostics: tuple[dict[str, Any], ...] = ()
    latest_verifier_fingerprint: str | None = None
    repair_guidance_fingerprint: str | None = None
    repair_baseline_error_count: int | None = None
    repair_baseline_errors: tuple[dict[str, Any], ...] = ()
    repair_baseline_target_diagnostics: tuple[dict[str, Any], ...] = ()
    repair_baseline_fingerprint: str | None = None
    repair_previous_source: str | None = None
    repair_previous_path: str | None = None
    last_verifier_quality: str | None = None
    repair_evidence_route: str | None = None
    trusted_materialized_baseline: tuple[str, str, bool] | None = None
    last_failure_reason: str | None = None
    termination_reason: str | None = None
    trajectory: list[ExecutionStepTrace] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def has_fresh_evidence(self) -> bool:
        with self._lock:
            return bool(self.evidence_fingerprints)

    @property
    def has_authoritative_java_evidence(self) -> bool:
        with self._lock:
            return bool(self.authoritative_java_evidence_fingerprints)

    def record_query(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        sig = retrieval_query_signature(tool_name, arguments)
        with self._lock:
            if sig in self.attempted_queries:
                return False
            self.attempted_queries.add(sig)
            self.attempted_sources.add(str(tool_name or "").strip())
            self.attempted_sources.add(retrieval_source_key(tool_name, arguments))
            return True

    def record_source_attempt(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> None:
        """Consume one route even when its result is not classified as RAG evidence."""
        with self._lock:
            self.attempted_sources.add(str(tool_name or "").strip())
            self.attempted_sources.add(retrieval_source_key(tool_name, arguments))

    def is_query_attempted(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        sig = retrieval_query_signature(tool_name, arguments)
        with self._lock:
            return sig in self.attempted_queries

    def record_evidence(self, value: Any, *, usable: bool) -> bool:
        if not usable:
            return False
        fp = evidence_fingerprint(value)
        if fp is None:
            return False
        with self._lock:
            return _record_evidence_locked(self, value, fp)

    def record_evidence_adjudication(self, value: Mapping[str, Any]) -> None:
        with self._lock:
            self.evidence_adjudications.append(dict(value))
            if len(self.evidence_adjudications) > 16:
                del self.evidence_adjudications[:-16]


    def record_mutation(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        signature = evidence_fingerprint({"tool": tool_name, "arguments": dict(arguments)}) or ""
        applied = mutation_payload_applied(tool_name, payload)
        with self._lock:
            if not applied:
                return _record_unapplied_mutation(self, signature)
            return _record_applied_mutation(self, tool_name, arguments, signature)


    def record_verification(
        self, tool_name: str, payload: Mapping[str, Any], status: str
    ) -> bool:
        from .validation_diagnostic_contract import diagnostic_errors
        result = payload.get("result")
        receipt = result if isinstance(result, Mapping) else payload
        errors: list[dict[str, Any]] = []
        if status == "FAIL":
            for item in diagnostic_errors(receipt):
                compact = {
                    key: item.get(key)
                    for key in (
                        "uri", "path", "file", "severity", "code", "source",
                        "message", "range", "line",
                    )
                    if item.get(key) not in (None, "", [], {})
                }
                if compact:
                    errors.append(compact)
        fp = evidence_fingerprint({"tool": tool_name, "status": status, "errors": errors})
        with self._lock:
            prior_status = self.validation_status
            prior_fp = self.latest_verifier_fingerprint
            baseline_count = self.repair_baseline_error_count
            new_error_count = len(errors)
            if status == "PASS":
                quality = "IMPROVED"
                progress = True
            elif status == "FAIL" and baseline_count is not None:
                if new_error_count < baseline_count:
                    quality = "IMPROVED"
                    progress = True
                elif new_error_count > baseline_count:
                    quality = "NON_IMPROVING"
                    progress = False
                else:
                    quality = (
                        "UNCHANGED"
                        if fp == self.repair_baseline_fingerprint
                        else "NON_IMPROVING"
                    )
                    progress = False
            else:
                quality = "OBSERVED"
                progress = status != prior_status or fp != prior_fp
            self.validation_status = status
            self.latest_verifier_tool = tool_name
            self.latest_verifier_errors = tuple(errors)
            self.latest_verifier_fingerprint = fp
            self.last_verifier_quality = quality
            if status != "FAIL":
                self.repair_guidance_fingerprint = None
                self.repair_baseline_error_count = None
                self.repair_baseline_errors = ()
                self.repair_baseline_target_diagnostics = ()
                self.repair_baseline_fingerprint = None
                self.repair_previous_source = None
                self.repair_previous_path = None
                self.repair_evidence_route = None
            return progress

    def take_verifier_repair_guidance(self) -> str | None:
        with self._lock:
            payload = _repair_guidance_payload(self)
        if payload is None:
            return None
        if payload["repair_window_policy"] == "host_owned_authored_implementation_required":
            return (
                "MMM_CORE_VERIFIER_REPAIR_V5\n"
                "The exact fresh authored feature is still unimplemented. Complete its approved "
                "task requirements in the same file, with needed imports, state, and helpers. "
                "Emit the complete updated source in new. The host binds the exact path, live "
                "old source and count. Preserve source identity and approved behavior. Do not "
                "merely rename/remove the placeholder or emit another scaffold. Validation is "
                "still FAIL; the next mutation goes to VERIFY. Equal or worse results are rolled "
                "back and count as no progress.\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            )
        return (
            "MMM_CORE_VERIFIER_REPAIR_V5\n"
            "The verifier failure is the active repair obligation. Do not restart generation, "
            "do not search unrelated ecosystem candidates, and never write a different path. "
            "The payload includes a host-selected bounded repair_window when verifier location "
            "evidence can localize the defect. Any earlier host_reserved/fresh metadata is "
            "pre-materialization history only. Never regenerate the complete source file. "
            "repair_window and its single diagnostic are host-selected. Emit only replacement "
            "text for that exact old window in new; the host binds operation=replace_exact, "
            "path, old text, count, and optimistic-concurrency SHA. These binding fields are "
            "host-owned. Whole-file reconstruction is forbidden. "
            "Preserve package/type identity and approved behavior. Make one materially different "
            "edit that reduces severity-1 diagnostics. "
            "An equal or worse verifier "
            "result is rolled back and counts as no progress. The next successful mutation goes "
            "directly back to VERIFY.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )


    def record_failure(self, tool_name: str, error: Any) -> None:
        with self._lock:
            self.last_failure_reason = f"{tool_name}: {str(error).strip()}"

    def clear_failure(self) -> None:
        with self._lock:
            self.last_failure_reason = None

    def record_no_progress_result(self, value: Any) -> bool:
        stable = _stable_value(value)
        canonical = json.dumps(
            stable,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            first_seen_step = self.no_progress_digest_first_step.get(digest)
            repeated = first_seen_step is not None
            if first_seen_step is None:
                self.no_progress_digest_first_step[digest] = self.step_index
            self.seen_no_progress_digests.add(digest)
            self.semantic_fixed_point = repeated
            self.no_progress_streak += 1
            if repeated:
                self.fixed_point_digest = digest
                self.fixed_point_first_seen_step = first_seen_step
                self.fixed_point_repeat_step = self.step_index
                self.fixed_point_snapshot = (
                    dict(stable)
                    if isinstance(stable, Mapping)
                    else {"value": stable}
                )
            return repeated

    def clear_no_progress_result(self) -> None:
        with self._lock:
            self.seen_no_progress_digests.clear()
            self.no_progress_digest_first_step.clear()
            self.fixed_point_digest = None
            self.fixed_point_first_seen_step = None
            self.fixed_point_repeat_step = None
            self.fixed_point_snapshot = None
            self.semantic_fixed_point = self.unapplied_mutation_fixed_point
            self.no_progress_streak = 0

    def begin_recovery_evidence_epoch(self, verifier_fingerprint: str | None) -> bool:
        """Re-open retrieval once for a new concrete verifier diagnostic state."""

        fingerprint = str(verifier_fingerprint or "").strip()
        if not fingerprint:
            return False
        with self._lock:
            if self.recovery_evidence_epoch_fingerprint == fingerprint:
                return False
            self.recovery_evidence_epoch_fingerprint = fingerprint
            self.attempted_queries.clear()
            self.attempted_sources.clear()
            self.seen_no_progress_digests.clear()
            self.no_progress_digest_first_step.clear()
            self.fixed_point_digest = None
            self.fixed_point_first_seen_step = None
            self.fixed_point_repeat_step = None
            self.fixed_point_snapshot = None
            self.semantic_fixed_point = self.unapplied_mutation_fixed_point
            self.no_progress_streak = 0
            self._external_mcp_completed_capabilities = set()
            self._external_mcp_schema_capability = ""
            return True

    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
        localization_stage: LocalizationStage | None = None,
    ) -> str | None:
        del localization_stage
        exposed = set(exposed_tools)
        with self._lock:
            attempted = set(self.attempted_sources)
        for name in preferred:
            if name in exposed and name not in attempted:
                return name
        return None



def _source_edit_schema_for_context(
    schema: Mapping[str, Any],
    context: TargetMutationContext | None,
    *,
    implementation_recovery: bool = False,
) -> Mapping[str, Any]:
    """Project source-edit choices onto the exact live target mutation state.

    A fresh Java target exposes only create_file/path/content; aliases and repair-only
    operations remain host-side compatibility rather than model-facing choices.
    """
    if _tool_name(schema) != "apply_source_edit":
        return schema
    active_authority = CURRENT_MUTATION_AUTHORITY.get()
    bounded_unpinned = bool(
        not context_is_host_pinned(context)
        and active_authority is not None
        and active_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
    )
    if not context_is_host_pinned(context) and not bounded_unpinned:
        return schema
    cloned = deepcopy(schema)
    if not isinstance(cloned, dict):
        return schema
    function = cloned.get("function")
    if not isinstance(function, dict):
        return cloned
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if bounded_unpinned and isinstance(parameters, dict) and isinstance(properties, dict):
        roots = tuple(active_authority.roots)
        path_schema = deepcopy(properties.get("path") or {"type": "string"})
        if isinstance(path_schema, dict):
            path_schema["type"] = "string"
            path_schema["pattern"] = (
                "^(?:"
                + "|".join(re.escape(root) for root in roots)
                + ").+"
            )
            path_schema["description"] = (
                "Choose exactly one project-relative destination below one of these "
                f"host-owned roots: {list(roots)!r}. Paths outside them are invalid."
            )
            properties["path"] = path_schema
        for alias in ("file", "target_path", "target_file"):
            properties.pop(alias, None)
        parameters["required"] = list(
            dict.fromkeys([*(parameters.get("required") or ()), "operation", "path"])
        )
        parameters["additionalProperties"] = False
        function["description"] = (
            str(function.get("description") or "").strip()
            + " Authored bounded-root destination: path must match the host-supplied root pattern."
        ).strip()
        return cloned

    operation = properties.get("operation") if isinstance(properties, dict) else None
    fresh_java = bool(
        context.is_new_file and context.target_path.casefold().endswith(".java")
    )
    if isinstance(operation, dict):
        enum = operation.get("enum")
        if isinstance(enum, list):
            if fresh_java:
                operation["enum"] = [
                    value for value in enum
                    if str(value).strip().casefold() == "create_file"
                ]
            elif not context.is_new_file:
                operation["enum"] = [
                    value for value in enum
                    if str(value).strip().casefold() not in _SOURCE_CREATE_OPERATIONS
                ]

    if (
        not context.is_new_file
        and isinstance(parameters, dict)
        and isinstance(properties, dict)
    ):
        target_path = _canonical_mutation_path(context.target_path)
        if target_path:
            path_schema = deepcopy(properties.get("path") or {"type": "string"})
            if isinstance(path_schema, dict):
                path_schema["type"] = "string"
                path_schema["enum"] = [target_path]
                path_schema["description"] = (
                    "Exact existing host-pinned target. Do not choose or invent another path."
                )
                properties["path"] = path_schema
            for alias in ("file", "target_path", "target_file"):
                properties.pop(alias, None)

        if _host_bound_existing_rewrite_ready(context, implementation_recovery=implementation_recovery):
            new_schema = deepcopy(properties.get("new") or {"type": "string"})
            if isinstance(new_schema, dict):
                new_schema["type"] = "string"
                new_schema["maxLength"] = _HOST_BOUND_EXISTING_REWRITE_MAX_BYTES * 2
                new_schema["description"] = (
                    "Updated complete source for the exact existing host-pinned file. "
                    "Preserve unrelated valid behavior and source identity. Emit source text "
                    "only; the host binds operation=replace_exact, exact path, exact current "
                    "old source, count, and transactional precondition."
                )
            parameters["properties"] = {"new": new_schema}
            parameters["required"] = ["new"]
            parameters["additionalProperties"] = False
            properties = parameters["properties"]

    if fresh_java and isinstance(parameters, dict) and isinstance(properties, dict):
        # Operation and destination are already host-owned by TargetMutationContext.
        # Asking the small model to regenerate them creates avoidable tool-markup tokens
        # and another opportunity for protocol drift. The TaskCapsule binds both after
        # admission, so the model authors only the source body.
        content_schema = deepcopy(properties.get("content") or {"type": "string"})
        if isinstance(content_schema, dict):
            content_schema["type"] = "string"
            content_schema["description"] = (
                "Complete minimal compilable Java source for the already host-pinned "
                "fresh target. Emit source text only; the host binds operation and path."
            )
        parameters["properties"] = {"content": content_schema}
        parameters["required"] = ["content"]
        parameters["additionalProperties"] = False
        properties = parameters["properties"]
    description = str(function.get("description") or "").strip()
    if context.is_new_file and context.target_path.casefold().endswith(".java"):
        suffix = (
            "Fresh host-pinned Java target: create exactly one complete source file with "
            "create_file; the host compiles it immediately before any repair edit."
        )
    elif _host_bound_existing_rewrite_ready(context, implementation_recovery=implementation_recovery):
        suffix = (
            "Host-owned existing target: emit only the updated source in new. "
            "The host owns the exact path, current old source, replace_exact operation, "
            "count, and transactional precondition."
        )
    elif not context.is_new_file:
        suffix = (
            "Existing host-pinned target: creation operations are unavailable. Use "
            "one bounded existing-file edit only; verifier repair never reconstructs "
            "the complete source file."
        )
    else:
        suffix = ""
    function["description"] = f"{description} {suffix}".strip()
    return cloned


def _fresh_observe_names(
    by_name: Mapping[str, Mapping[str, Any]],
    attempted: set[str],
    mutation_context: TargetMutationContext,
    *,
    semantic_retrieval_choice: bool,
) -> list[str]:
    del semantic_retrieval_choice
    return list(
        initial_evidence_frontier(
            available=tuple(by_name),
            attempted=attempted,
            localization_stage=mutation_context.localization_stage.value,
            semantic_fresh_java_target=True,
        )
    )


def _localized_observe_names(
    by_name: Mapping[str, Mapping[str, Any]],
    attempted: set[str],
    mutation_context: TargetMutationContext | None,
    *,
    semantic_retrieval_choice: bool,
    semantic_fresh_java_target: bool = False,
) -> list[str]:
    del semantic_retrieval_choice
    stage = (
        mutation_context.localization_stage.value
        if mutation_context is not None
        else LocalizationStage.NEED_FILE.value
    )
    return list(
        initial_evidence_frontier(
            available=tuple(by_name),
            attempted=attempted,
            localization_stage=stage,
            semantic_fresh_java_target=bool(
                semantic_fresh_java_target
                or (
                    mutation_context is not None
                    and mutation_context.is_new_file
                    and mutation_context.is_mutation_ready
                )
            ),
        )
    )


def _filter_tools_for_phase(
    exposed_tools: Sequence[Mapping[str, Any]],
    phase: LoopPhase,
    role: str,
    *,
    mutation_context: TargetMutationContext | None = None,
    attempted_sources: Sequence[str] | set[str] | frozenset[str] = frozenset(),
    localization_active: bool | None = None,
    semantic_retrieval_choice: bool = False,
    semantic_fresh_java_target: bool = False,
    repair_evidence_route: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    del role
    by_name = {_tool_name(schema): schema for schema in exposed_tools if _tool_name(schema)}
    attempted = set(attempted_sources)
    if phase == LoopPhase.ACT:
        names = ["apply_source_edit"] if "apply_source_edit" in by_name else [
            name for name in by_name if name in _MUTATION_ACT_TOOLS
        ]
    elif phase == LoopPhase.VERIFY:
        names = [name for name in by_name if name in _VERIFY_TOOLS]
    elif phase == LoopPhase.RECOVER:
        names = list(
            recovery_evidence_frontier(
                available=tuple(by_name),
                attempted=attempted,
                route=repair_evidence_route,
            )
        )
    else:
        active = mutation_context is not None if localization_active is None else localization_active
        if not active:
            names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
            if mutation_context is None and "search_code_rag" in names and "java_workspace_symbols" in names:
                names.remove("java_workspace_symbols")
        else:
            names = _localized_observe_names(
                by_name,
                attempted,
                mutation_context,
                semantic_retrieval_choice=semantic_retrieval_choice,
                semantic_fresh_java_target=semantic_fresh_java_target,
            )
    return tuple(
        _source_edit_schema_for_context(by_name[name], mutation_context)
        for name in names
        if name in by_name
    )


def _replace_live_messages(
    messages: list[dict[str, Any]],
    fitted: Sequence[Mapping[str, Any]],
) -> bool:
    replacement = [dict(message) for message in fitted]
    if replacement == messages:
        return False
    messages[:] = replacement
    return True


def _retry_atomic_after_output_exhaustion(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    media_paths: tuple[Any, ...],
) -> Any:
    """Retry one bounded native tool action without resetting HostRunState."""

    already_recovered = any(
        isinstance(message.get("content"), str)
        and _ATOMIC_OUTPUT_RECOVERY_MARKER in str(message.get("content"))
        for message in messages
        if isinstance(message, Mapping)
    )
    if already_recovered:
        raise ModelConfigurationError(
            "ATOMIC_ACTION_OUTPUT_STALLED: bounded output recovery was already used "
            "for this live tool transcript."
        )
    messages.append({
        "role": "system",
        "content": (
            _ATOMIC_OUTPUT_RECOVERY_MARKER
            + "\n"
            + _atomic_output_recovery_instruction(request)
        ),
    })
    retry_metadata = (
        dict(request.metadata) if isinstance(request.metadata, Mapping) else {}
    )
    retry_metadata["mmm_atomic_output_recovery"] = True
    retry_metadata["mmm_disable_lora"] = True
    try:
        existing_ceiling = int(
            retry_metadata.get("mmm_output_token_ceiling")
            or _ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS
        )
    except (TypeError, ValueError):
        existing_ceiling = _ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS
    retry_metadata["mmm_output_token_ceiling"] = min(
        max(1, existing_ceiling),
        _ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS,
    )
    retry_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
        metadata=retry_metadata,
        parallel_tool_calls=False if request.tools else request.parallel_tool_calls,
    )
    emit_root_cause(
        "atomic_output_boundary_recovery",
        operation="generate_with_tools",
        gate="completion_boundary",
        result="RETRY",
        details={"tool_choice": request.tool_choice},
    )
    try:
        return _generate_turn_in_scope(
            router, config=config, adapter=adapter, turn_request=retry_request
        )
    except BaseException as retry_exc:
        if completion_boundary_kind(retry_exc) == OUTPUT_EXHAUSTED:
            boundary = completion_boundary_error(retry_exc)
            partial = (
                boundary.partial_message
                if boundary is not None and isinstance(boundary.partial_message, Mapping)
                else {}
            )
            raw_calls = partial.get("tool_calls") if isinstance(partial, Mapping) else None
            calls = raw_calls if isinstance(raw_calls, Sequence) and not isinstance(
                raw_calls, (str, bytes, bytearray)
            ) else ()
            argument_chars = 0
            tool_names: list[str] = []
            for raw_call in calls:
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, Mapping):
                    continue
                name = str(function.get("name") or "").strip()
                if name:
                    tool_names.append(name)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    argument_chars += len(arguments)
                elif arguments is not None:
                    try:
                        argument_chars += len(json.dumps(arguments, ensure_ascii=False))
                    except (TypeError, ValueError):
                        argument_chars += len(str(arguments))
            emit_root_cause(
                "atomic_output_recovery_exhausted",
                stage="generation",
                operation="generate_with_tools",
                gate="completion_boundary",
                result="FAIL",
                reason="second bounded atomic decode exhausted before one tool action completed",
                details={
                    "partial_bytes": int(getattr(boundary, "partial_bytes", 0) or 0),
                    "partial_sha256": str(getattr(boundary, "partial_sha256", "") or ""),
                    "content_chars": len(str(partial.get("content") or "")),
                    "reasoning_chars": len(
                        str(partial.get("reasoning_content") or partial.get("reasoning") or "")
                    ),
                    "tool_call_count": len(calls),
                    "tool_names": tool_names,
                    "tool_argument_chars": argument_chars,
                },
            )
            raise ModelConfigurationError(
                "ATOMIC_ACTION_OUTPUT_STALLED: the model exceeded the output allowance twice "
                "without completing one bounded semantic action; refusing to reset agent state."
            ) from retry_exc
        raise


def _exact_context_recovery_candidate(
    messages: Sequence[Mapping[str, Any]],
    *,
    turn_request: GenerationRequest,
    exact_accounting: Any,
    config: Any,
    tools: Sequence[Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, int]] | None:
    """Select the largest deterministic retry with useful live output space."""

    base_budget = max(1, int(request_message_budget(config, tools)))
    budgets = tuple(
        dict.fromkeys(
            max(1, base_budget * numerator // 8)
            for numerator in (8, 7, 6, 5, 4, 3, 2, 1)
        )
    )
    original = tuple(messages)
    fallback: tuple[tuple[Mapping[str, Any], ...], dict[str, int]] | None = None

    for budget in budgets:
        candidate = tuple(emergency_fit_messages(original, budget_bytes=budget))
        if candidate == original:
            continue
        accounting = exact_accounting(replace(turn_request, messages=candidate))
        input_tokens = int(accounting.input_tokens)
        context_tokens = int(accounting.context_tokens)
        remaining_tokens = context_tokens - input_tokens
        if remaining_tokens <= 0:
            continue
        receipt = {
            "budget_bytes": budget,
            "input_tokens": input_tokens,
            "context_tokens": context_tokens,
            "remaining_tokens": remaining_tokens,
        }
        if fallback is None:
            fallback = (candidate, receipt)
        configured_output = max(1, int(getattr(config, "max_new_tokens", 0) or 1))
        desired_reserve = min(configured_output, max(1, context_tokens // 4))
        if remaining_tokens >= desired_reserve:
            return candidate, receipt
    return fallback


def _generate_turn_in_scope(
    router: Any, *, config: Any, adapter: Any, turn_request: GenerationRequest
) -> Any:
    scope_factory = getattr(router, "_generation_scope", None)
    scope = scope_factory(config) if callable(scope_factory) else nullcontext()
    with scope:
        return adapter.generate_turn(turn_request)

def _forced_tool_choice_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name") or "").strip()


def _generate_turn_with_context_recovery(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    media_paths: tuple[Any, ...],
    tool_choice: Any,
    parallel_tool_calls: bool,
    verifier_relative_files: tuple[str, ...] = (),
) -> Any:
    """Fit one live turn and recover typed completion boundaries in-place."""

    if _forced_tool_choice_name(tool_choice) == "java_diagnostics":
        from .generation_verifier_resilience import synthesized_verifier_turn

        emit_root_cause(
            "generation_verifier_model_turn_elided",
            stage="generation",
            operation="java_diagnostics",
            gate="host_verifier_authority",
            result="PASS",
            reason=(
                "forced verifier selection is mechanical and does not require "
                "coder inference"
            ),
        )
        return synthesized_verifier_turn(
            messages,
            relative_files=verifier_relative_files or None,
        )

    request_metadata = (
        dict(request.metadata) if isinstance(request.metadata, Mapping) else {}
    )
    if _forced_tool_choice_name(tool_choice) == "apply_source_edit":
        try:
            existing_ceiling = int(
                request_metadata.get("mmm_output_token_ceiling")
                or _ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS
            )
        except (TypeError, ValueError):
            existing_ceiling = _ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS
        request_metadata["mmm_output_token_ceiling"] = min(
            max(1, existing_ceiling),
            _ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS,
        )

    turn_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
        metadata=request_metadata,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    try:
        declared_accounting = getattr_static(adapter, "input_context_accounting")
    except AttributeError:
        declared_accounting = None
    exact_accounting = (
        getattr(adapter, "input_context_accounting", None)
        if callable(declared_accounting)
        else None
    )
    # Canonical fitting is not only an overflow fallback: it removes duplicated
    # implementation receipts and bounds exact-source seed data on every small-model
    # turn. Exact accounting must never bypass that semantic compaction merely because
    # the unprojected prompt happens to fit inside the runtime slot.
    fitted = fit_messages_to_context(messages, config=config, tools=request.tools)
    _replace_live_messages(messages, fitted)
    turn_request = replace(turn_request, messages=tuple(messages))

    try:
        return _generate_turn_in_scope(
            router, config=config, adapter=adapter, turn_request=turn_request
        )
    except BaseException as exc:
        boundary_kind = completion_boundary_kind(exc)
        if boundary_kind == OUTPUT_EXHAUSTED:
            return _retry_atomic_after_output_exhaustion(
                router,
                config=config,
                adapter=adapter,
                request=turn_request,
                messages=messages,
                media_paths=media_paths,
            )
        if boundary_kind != CONTEXT_PRESSURE:
            raise

        recovery_receipt: dict[str, int] = {}
        if callable(exact_accounting):
            exact_recovery = _exact_context_recovery_candidate(
                messages,
                turn_request=turn_request,
                exact_accounting=exact_accounting,
                config=config,
                tools=request.tools,
            )
            if exact_recovery is None:
                mark_context_recovery_exhausted(exc)
                raise
            emergency, recovery_receipt = exact_recovery
        else:
            active_budget = max(1, request_message_budget(config, request.tools))
            emergency_budget = max(1, active_budget * 3 // 4)
            emergency = emergency_fit_messages(messages, budget_bytes=emergency_budget)
            recovery_receipt = {"budget_bytes": emergency_budget}

        if not _replace_live_messages(messages, emergency):
            mark_context_recovery_exhausted(exc)
            raise
        retry_request = replace(
            turn_request,
            messages=tuple(messages),
            media_paths=media_paths,
        )
        emit_root_cause(
            "context_boundary_recovery",
            operation="generate_with_tools",
            gate="completion_boundary",
            result="RETRY",
            reason=f"{type(exc).__name__}: {exc}",
            details=recovery_receipt,
        )
        try:
            return _generate_turn_in_scope(
                router, config=config, adapter=adapter, turn_request=retry_request
            )
        except BaseException as retry_exc:
            retry_kind = completion_boundary_kind(retry_exc)
            if retry_kind == OUTPUT_EXHAUSTED:
                return _retry_atomic_after_output_exhaustion(
                    router,
                    config=config,
                    adapter=adapter,
                    request=retry_request,
                    messages=messages,
                    media_paths=media_paths,
                )
            if retry_kind == CONTEXT_PRESSURE:
                mark_context_recovery_exhausted(exc)
                raise exc from retry_exc
            raise


_PHASE_HANDOFF_VERIFIER_DIAGNOSTIC_BYTES = 4 * 1024
_REPAIR_GUIDANCE_VERIFIER_DIAGNOSTIC_BYTES = 12 * 1024
_PHASE_HANDOFF_DIAGNOSTIC_TEXT_LIMIT = 640
_PHASE_HANDOFF_TOTAL_BYTES = 6 * 1024
_PHASE_HANDOFF_TOOL_RECORD_LIMIT = 6
_PHASE_HANDOFF_TOOL_TEXT_LIMIT = 960
_PHASE_HANDOFF_NESTED_KEYS = (
    "structured_content",
    "result",
    "data",
    "body",
    "raw_result",
    "structured",
    "observation",
    "hits",
    "results",
    "records",
    "documents",
    "chunks",
    "resources",
    "sources",
    "items",
    "symbols",
    "operations",
)
_PHASE_HANDOFF_RECORD_KEYS = (
    "schema_version",
    "status",
    "result_count",
    "path",
    "source_path",
    "file",
    "uri",
    "line",
    "start_line",
    "end_line",
    "symbol",
    "name",
    "code",
    "severity",
    "message",
    "text",
    "snippet",
    "content",
    "source",
    "changed_paths",
    "operation",
    "after_sha256",
)


def _phase_handoff_scalar(value: Any, *, limit: int = _PHASE_HANDOFF_TOOL_TEXT_LIMIT) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "…"
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        scalars = []
        for item in value[:8]:
            if isinstance(item, (str, bool, int, float)) or item is None:
                scalars.append(_phase_handoff_scalar(item, limit=320))
        return scalars
    return None


def _phase_handoff_records(value: Any, records: list[dict[str, Any]]) -> None:
    if len(records) >= _PHASE_HANDOFF_TOOL_RECORD_LIMIT:
        return
    if isinstance(value, Mapping):
        record: dict[str, Any] = {}
        for key in _PHASE_HANDOFF_RECORD_KEYS:
            if key not in value:
                continue
            projected = _phase_handoff_scalar(value.get(key))
            if projected not in (None, "", [], {}):
                record[key] = projected
        if record:
            records.append(record)
            if len(records) >= _PHASE_HANDOFF_TOOL_RECORD_LIMIT:
                return
        for key in _PHASE_HANDOFF_NESTED_KEYS:
            child = value.get(key)
            if child is not None:
                _phase_handoff_records(child, records)
                if len(records) >= _PHASE_HANDOFF_TOOL_RECORD_LIMIT:
                    return
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _phase_handoff_records(child, records)
            if len(records) >= _PHASE_HANDOFF_TOOL_RECORD_LIMIT:
                return


def _bounded_phase_tool_observation(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    parsed: Any = content
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = content

    records: list[dict[str, Any]] = []
    _phase_handoff_records(parsed, records)
    fingerprint = evidence_fingerprint(parsed)
    plain_excerpt = (
        _phase_handoff_scalar(parsed)
        if isinstance(parsed, str) and parsed.strip()
        else None
    )
    payload = {
        "schema_version": "mmm/phase-tool-observation-v1",
        "tool_result_fingerprint": (
            "sha256:" + fingerprint if fingerprint else None
        ),
        "records": records,
        "excerpt": plain_excerpt,
        "policy": (
            "Bounded host projection of the completed tool result. Raw tool payload "
            "is intentionally not replayed across phase boundaries."
        ),
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    if len(rendered.encode("utf-8")) <= _PHASE_HANDOFF_TOTAL_BYTES:
        return rendered

    while records:
        records.pop()
        payload["records"] = records
        payload["omitted_record_count"] = 1
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if len(rendered.encode("utf-8")) <= _PHASE_HANDOFF_TOTAL_BYTES:
            return rendered
    return json.dumps(
        {
            "schema_version": "mmm/phase-tool-observation-v1",
            "tool_result_fingerprint": (
                "sha256:" + fingerprint if fingerprint else None
            ),
            "records": [],
            "excerpt": plain_excerpt,
            "omitted_record_count": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _bounded_phase_handoff_content(
    *,
    previous_phase: LoopPhase,
    next_phase: LoopPhase,
    observations: Sequence[str],
) -> str:
    base = (
        f"MMM_PHASE_HANDOFF {previous_phase.value}->{next_phase.value}\n"
        "Prior-phase tool calls are closed and cannot be replayed.\n"
    )
    kept: list[str] = []
    omitted = 0
    for value in observations:
        candidate = base + "\n".join(
            f"Observation {index}:\n{item}"
            for index, item in enumerate((*kept, value), 1)
        )
        if len(candidate.encode("utf-8")) <= _PHASE_HANDOFF_TOTAL_BYTES:
            kept.append(value)
        else:
            omitted += 1
    suffix = f"\nOmitted observations: {omitted}" if omitted else ""
    return (
        base
        + "\n".join(
            f"Observation {index}:\n{item}"
            for index, item in enumerate(kept, 1)
        )
        + suffix
    )


def _bounded_verifier_recovery_observation(
    state: HostRunState,
    *,
    errors: Sequence[Mapping[str, Any]] | None = None,
    budget_bytes: int | None = None,
) -> str:
    """Serialize only prompt-useful verifier evidence across VERIFY->RECOVER.

    The raw JDT receipt can contain the complete workspace diagnostic graph and is
    intentionally much larger than one repair turn. Elevating that raw tool payload
    into a system phase-handoff message makes it mandatory context and defeats the
    normal tool-message compactor. Keep a bounded host-owned diagnostic projection
    plus a fingerprint of the complete extracted error set instead.
    """

    raw_errors = tuple(state.latest_verifier_errors if errors is None else errors)
    diagnostics: list[dict[str, Any]] = []
    budget = (
        _PHASE_HANDOFF_VERIFIER_DIAGNOSTIC_BYTES
        if budget_bytes is None
        else max(1024, int(budget_bytes))
    )
    target_path = (
        _canonical_mutation_path(state.mutation_context.target_path)
        if errors is not None and state.mutation_context is not None
        else ""
    )
    target_scoped = bool(target_path)
    seen_compact: set[str] = set()
    for raw in raw_errors:
        if not isinstance(raw, Mapping):
            continue
        compact: dict[str, Any] = {}
        for key in ("path", "file", "uri", "line", "severity", "code", "source", "message", "range"):
            value = raw.get(key)
            if value in (None, "", [], {}):
                continue
            if target_scoped and key in {"file", "uri"}:
                continue
            if isinstance(value, str):
                limit = (
                    _PHASE_HANDOFF_DIAGNOSTIC_TEXT_LIMIT
                    if key == "message"
                    else 320
                )
                if len(value) > limit:
                    value = value[: max(0, limit - 1)] + "…"
            compact[key] = value
        if not compact:
            continue
        compact_key = json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if compact_key in seen_compact:
            continue
        seen_compact.add(compact_key)
        trial = {
            "verifier": str(state.latest_verifier_tool or ""),
            "status": "FAIL",
            "diagnostics": [*diagnostics, compact],
        }
        if target_scoped:
            trial["target_path"] = target_path
        if len(
            json.dumps(
                trial,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ) > budget:
            break
        diagnostics.append(compact)

    payload = {
        "schema_version": "mmm/verifier-recovery-handoff-v1",
        "verifier": str(state.latest_verifier_tool or ""),
        "status": "FAIL",
        "diagnostics": diagnostics,
        "diagnostics_fingerprint": evidence_fingerprint(raw_errors),
        "diagnostic_count": len(raw_errors),
        "omitted_diagnostic_count": max(0, len(raw_errors) - len(diagnostics)),
        "policy": (
            "Host-extracted bounded verifier diagnostics. The raw verifier receipt is "
            "not replayed into mandatory model context."
        ),
    }
    if target_scoped:
        payload["target_path"] = target_path
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sync_phase_tool_transcript(
    messages: list[dict[str, Any]],
    *,
    state: HostRunState,
    last_prompt_phase: LoopPhase,
    stage: str,
) -> LoopPhase:
    next_phase = state.phase
    if next_phase == last_prompt_phase:
        return last_prompt_phase
    compacted: list[dict[str, Any]] = []
    observations: list[str] = []
    verifier_recovery_handoff = bool(
        next_phase is LoopPhase.RECOVER
        and state.latest_verifier_errors
        and state.latest_verifier_fingerprint
    )
    for raw in messages:
        message = dict(raw)
        role = str(message.get("role") or "")
        content = message.get("content")
        if (
            role == "system"
            and isinstance(content, str)
            and content.startswith("MMM_PHASE_HANDOFF ")
        ):
            # A phase transition owns exactly one live handoff snapshot. Keeping
            # historical handoffs makes mandatory context grow monotonically.
            continue
        if role == "tool":
            if not verifier_recovery_handoff:
                observations.append(_bounded_phase_tool_observation(message))
            continue
        if role == "assistant" and message.get("tool_calls"):
            if isinstance(content, str) and content.strip():
                compacted.append({"role": "assistant", "content": content})
            continue
        compacted.append(message)
    if verifier_recovery_handoff:
        observations = [_bounded_verifier_recovery_observation(state)]
    compacted.append({
        "role": "system",
        "content": _bounded_phase_handoff_content(
            previous_phase=last_prompt_phase,
            next_phase=next_phase,
            observations=observations,
        ),
    })
    messages[:] = compacted
    emit_root_cause(
        "phase_tool_transcript_handoff",
        stage=stage,
        operation="generate_with_tools",
        gate="phase_boundary",
        result="PASS",
        details={"previous_phase": last_prompt_phase.value, "next_phase": next_phase.value},
    )
    return next_phase


def _verifier_tool(
    schemas: Sequence[Mapping[str, Any]], unavailable: set[str]
) -> str | None:
    names = {_tool_name(schema) for schema in schemas}
    for name in ("java_diagnostics", "jdt_diagnostics", "run_gametest"):
        if name in names and name not in unavailable:
            return name
    return None


def _rollback_non_improving_verifier_repair(
    state: HostRunState,
    runtime: Any,
    *,
    stage: str,
) -> bool:
    """Restore the last verifier-proven source when a repair fails to improve it."""

    if state.last_verifier_quality not in {"NON_IMPROVING", "UNCHANGED"}:
        return False
    path = _canonical_mutation_path(state.repair_previous_path or "")
    source = state.repair_previous_source
    context = state.mutation_context
    rollback_arguments = exact_rollback_arguments(
        path=path,
        previous_source=source,
        context_path=getattr(context, "target_path", None),
        current_source=getattr(context, "source_body", None),
    )
    if rollback_arguments is None:
        return False
    try:
        result = runtime.call(
            stage,
            "apply_source_edit",
            rollback_arguments,
        )
    except Exception as exc:
        raise ModelConfigurationError(
            "VERIFICATION_REPAIR_ROLLBACK_FAILED: host rollback tool failed for "
            f"{path!r}: {type(exc).__name__}: {exc}"
        ) from exc
    applied = mutation_payload_applied(
        "apply_source_edit",
        {"ok": True, "result": result},
    )
    if not applied:
        raise ModelConfigurationError(
            "VERIFICATION_REPAIR_ROLLBACK_FAILED: host could not restore the "
            f"last verifier-proven source for {path!r}"
        )
    with state._lock:
        context = state.mutation_context
        if context is not None and _canonical_mutation_path(context.target_path) == path:
            state.mutation_context = replace(
                context,
                source_body=source,
                evidence_source="verifier_workspace_source",
                is_new_file=False,
            )
        state.validation_status = "FAIL"
        state.latest_verifier_errors = tuple(state.repair_baseline_errors)
        state.repair_target_diagnostics = tuple(
            state.repair_baseline_target_diagnostics
        )
        state.latest_verifier_fingerprint = state.repair_baseline_fingerprint
        state.repair_guidance_fingerprint = None
        state.last_verifier_quality = "NON_IMPROVING"
    emit_root_cause(
        "verifier_repair_rolled_back",
        stage=stage,
        operation="apply_source_edit",
        gate="repair_quality_monotonicity",
        result="PASS",
        reason=(
            "repair did not reduce verifier severity-1 diagnostics; restored "
            "the previous verifier-proven source"
        ),
        details={
            "target_path": path,
            "baseline_error_count": state.repair_baseline_error_count,
        },
    )
    return True


def _fixed_point_diagnostic(state: HostRunState) -> dict[str, Any]:
    context = state.mutation_context
    has_fresh = state.has_fresh_evidence
    has_authoritative = state.has_authoritative_java_evidence
    evidence_ready = evidence_obligation_satisfied(
        require_evidence=state.require_evidence
        or repair_route_requires_retrieval(state.repair_evidence_route),
        semantic_fresh_java_target=bool(
            state.semantic_fresh_java
            or state.repair_evidence_route == "official_api"
        ),
        has_fresh_evidence=has_fresh,
        has_authoritative_java_evidence=has_authoritative,
    )
    blockers: list[str] = []
    if state.require_evidence and state.semantic_fresh_java and not has_authoritative:
        blockers.append("AUTHORITATIVE_JAVA_EVIDENCE_MISSING")
    if any(
        item.get("query_mentions_target") and not item.get("authoritative_accepted")
        for item in state.evidence_adjudications
    ):
        blockers.append("SELF_TARGET_QUERY_NOT_AUTHORITATIVE")
    timeout_count = sum(
        1
        for step in state.trajectory
        for result in step.tool_results
        if str(result.get("failure_code") or "") == "EXTERNAL_MCP_TIMEOUT"
    )
    if timeout_count:
        blockers.append("EXTERNAL_MCP_TIMEOUT_OBSERVED")
    if state.last_failure_reason:
        blockers.append("MODEL_OR_TOOL_REJECTION_OBSERVED")
    mcp_state = recovery_state_snapshot(state, state.repair_evidence_route)
    return {
        "phase": state.phase.value,
        "validation_status": state.validation_status,
        "target_path": context.target_path if context is not None else None,
        "target_symbol": context.target_symbol if context is not None else None,
        "require_evidence": state.require_evidence,
        "semantic_fresh_java": state.semantic_fresh_java,
        "host_grounded": state.host_grounded,
        "compile_backed_java": state.compile_backed_java,
        "has_fresh_evidence": has_fresh,
        "has_authoritative_java_evidence": has_authoritative,
        "evidence_ready": evidence_ready,
        "no_progress_streak": state.no_progress_streak,
        "fixed_point_digest": state.fixed_point_digest,
        "fixed_point_first_seen_step": state.fixed_point_first_seen_step,
        "fixed_point_repeat_step": state.fixed_point_repeat_step,
        "blockers": blockers,
        "external_mcp_timeout_count": timeout_count,
        "last_failure_reason": state.last_failure_reason,
        "attempted_sources": sorted(state.attempted_sources),
        "mcp_capabilities_seen": mcp_state["capabilities_seen"],
        "mcp_available_capabilities": mcp_state["available_capabilities"],
        "mcp_completed_capabilities": mcp_state["completed_capabilities"],
        "mcp_bound_schema_capability": mcp_state["bound_schema_capability"],
        "mcp_next_capability": mcp_state["next_capability"],
        "last_evidence_adjudications": list(state.evidence_adjudications[-6:]),
        "repeated_state": state.fixed_point_snapshot,
    }


def _format_fixed_point_diagnostic(details: Mapping[str, Any]) -> str:
    blockers = ",".join(str(item) for item in details.get("blockers", ())) or "<none>"
    lines = [
        (
            "ROOT_CAUSE_DIAGNOSIS:"
            f" blockers={blockers}"
            f" evidence_required={details.get('require_evidence')}"
            f" semantic_fresh_java={details.get('semantic_fresh_java')}"
            f" evidence_ready={details.get('evidence_ready')}"
            f" fresh_evidence={details.get('has_fresh_evidence')}"
            f" authoritative_java_evidence={details.get('has_authoritative_java_evidence')}"
        ),
        (
            "FIXED_POINT_RECURRENCE:"
            f" digest={details.get('fixed_point_digest')}"
            f" first_seen_step={details.get('fixed_point_first_seen_step')}"
            f" repeated_step={details.get('fixed_point_repeat_step')}"
        ),
        (
            "MCP_FRONTIER:"
            f" completed={details.get('mcp_completed_capabilities')}"
            f" next={details.get('mcp_next_capability')}"
            f" bound_schema={details.get('mcp_bound_schema_capability')}"
            f" timeouts={details.get('external_mcp_timeout_count')}"
        ),
    ]
    evidence_rows = details.get("last_evidence_adjudications")
    if isinstance(evidence_rows, Sequence):
        for row in evidence_rows[-4:]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "EVIDENCE:"
                f" step={row.get('step_index')}"
                f" tool={row.get('tool')}"
                f" capability={row.get('capability')}"
                f" query={row.get('query')!r}"
                f" usable={row.get('semantic_usable')}"
                f" authoritative={row.get('authoritative_accepted')}"
                f" reason={row.get('authoritative_reason')}"
                f" ready={row.get('evidence_ready')}"
            )
    return "\n".join(lines)


def _fixed_point_error(state: HostRunState) -> ModelConfigurationError:
    trajectory = format_trajectory_summary(state.trajectory)
    diagnostic = _fixed_point_diagnostic(state)
    emit_root_cause(
        "agent_fixed_point_diagnosis",
        stage="generation",
        operation="generate_with_tools",
        gate="semantic_convergence",
        result="BLOCKED",
        reason=",".join(diagnostic["blockers"]) or "SEMANTIC_STATE_REPEATED",
        details=diagnostic,
    )
    readable = _format_fixed_point_diagnostic(diagnostic)
    if state.validation_status == "FAIL":
        return ModelConfigurationError(
            "VERIFICATION_REPAIR_FIXED_POINT: the same repair state repeated while "
            "trustworthy verifier diagnostics remain unresolved.\n"
            + readable
            + "\n"
            + trajectory
        )
    return ModelConfigurationError(
        "AGENT_SEMANTIC_FIXED_POINT: the same action/result state repeated without "
        "workspace, localization, evidence, or verification progress.\n"
        + readable
        + "\n"
        + trajectory
    )


def _host_coder_summary(*, verification: str) -> str:
    """Return the fixed coder-summary contract from host-owned terminal state.

    Once mutation/verification state is terminal, asking the model for one more
    formatting-only turn reintroduces tool/prose drift after the host has already
    made the authoritative decision. The host therefore serializes the fixed
    response contract directly.
    """

    if verification == "PASS":
        summary = "Applied the approved source mutation and passed generation-time host verification."
    elif verification == "DEFERRED_TO_TARGET_COMPILE":
        summary = (
            "Applied the approved source mutation; the mandatory target_compile gate is "
            "the canonical Java verifier."
        )
    elif verification == "DEFERRED_TO_PROJECT_BUILD":
        summary = (
            "Applied the approved authored-design mutation; generation-time JDT verification "
            "was unavailable, so the mandatory project build gate must verify the complete "
            "generated workspace."
        )
    else:
        raise ModelConfigurationError(
            f"HOST_SUMMARY_STATE_INVALID: unsupported terminal verification state {verification!r}"
        )
    from .model_response_templates import serialize_response

    return serialize_response("coder_summary", {"summary": summary})



def _call_is_evidence_tool(
    call: Any,
    phase: LoopPhase,
    rag_evidence_tools: frozenset[str],
    external_rag_capability: Any,
) -> bool:
    if call.name in _LOCALIZATION_EVIDENCE_TOOLS:
        return True
    if call.name in rag_evidence_tools:
        return True
    if phase == LoopPhase.RECOVER and call.name in _RECOVERY_EVIDENCE_TOOLS:
        return True
    if call.name != "external_mcp_call":
        return False
    return bool(external_rag_capability(call.arguments))


def _evidence_call_adjudication(
    state: HostRunState,
    call: Any,
    payload: Mapping[str, Any],
    *,
    semantic_usable: bool,
    recorded: bool,
) -> dict[str, Any]:
    arguments = call.arguments if isinstance(call.arguments, Mapping) else {}
    nested = arguments.get("arguments")
    nested_arguments = nested if isinstance(nested, Mapping) else {}
    query = str(
        arguments.get("query")
        or nested_arguments.get("query")
        or ""
    ).strip()
    capability = str(arguments.get("capability") or "").strip()
    context = state.mutation_context
    target_path = context.target_path if context is not None else None
    target_symbol = (
        str(context.target_symbol or "").strip()
        if context is not None
        else ""
    )
    if not target_symbol and target_path:
        target_symbol = (
            str(target_path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        )
    diagnostic = _authoritative_java_evidence_diagnostic(
        payload.get("result"),
        target_path=target_path,
    )
    evidence_ready = evidence_obligation_satisfied(
        require_evidence=state.require_evidence
        or repair_route_requires_retrieval(state.repair_evidence_route),
        semantic_fresh_java_target=bool(
            state.semantic_fresh_java
            or state.repair_evidence_route == "official_api"
        ),
        has_fresh_evidence=state.has_fresh_evidence,
        has_authoritative_java_evidence=state.has_authoritative_java_evidence,
    )
    query_cf = query.casefold()
    target_cf = target_symbol.casefold()
    return {
        "step_index": state.step_index,
        "phase": state.phase.value,
        "tool": str(call.name),
        "capability": capability or None,
        "query": query[:320] or None,
        "query_mentions_target": bool(
            query_cf and target_cf and target_cf in query_cf
        ),
        "runtime_ok": bool(payload.get("ok")),
        "failure_code": payload.get("failure_code"),
        "semantic_usable": bool(semantic_usable),
        "new_evidence_fingerprint": bool(recorded),
        "authoritative_required": bool(
            state.semantic_fresh_java
            or state.repair_evidence_route == "official_api"
        ),
        "authoritative_accepted": bool(diagnostic["accepted"]),
        "authoritative_reason": str(diagnostic["reason"]),
        "evidence_schema": diagnostic.get("schema_version"),
        "api_hit": bool(diagnostic.get("api_hit")),
        "symbol_records": bool(diagnostic.get("symbol_records")),
        "mapping_records": bool(diagnostic.get("mapping_records")),
        "evidence_ready": bool(evidence_ready),
        "has_fresh_evidence": state.has_fresh_evidence,
        "has_authoritative_java_evidence": state.has_authoritative_java_evidence,
        "target_path": target_path,
        "target_symbol": target_symbol or None,
    }


def _message_mapping_payload(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _task_evidence_payload(
    messages: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for message in reversed(messages):
        payload = _message_mapping_payload(message)
        if payload is None:
            continue
        module = payload.get("module")
        if not isinstance(module, Mapping):
            continue
        task = module.get("evidence_task")
        if isinstance(task, Mapping):
            contract = task.get("coder_execution_contract")
            if isinstance(contract, Mapping):
                return contract
            return task
        config = module.get("config")
        if isinstance(config, Mapping):
            task = config.get("evidence_task")
            if isinstance(task, Mapping):
                return task
    return None


def _task_query_fragments(value: Any, result: list[str]) -> None:
    if len(result) >= 24:
        return
    if isinstance(value, str):
        text = " ".join(value.split()).strip()
        if text and not text.casefold().startswith("sha256:") and text not in result:
            result.append(text)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _task_query_fragments(child, result)
            if len(result) >= 24:
                return
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _task_query_fragments(child, result)
            if len(result) >= 24:
                return


def _approved_task_evidence_query(
    messages: Sequence[Mapping[str, Any]],
    *,
    target_path: str | None,
) -> str:
    """Build one bounded retrieval query from the approved task, never its generated name."""

    task = _task_evidence_payload(messages)
    if task is None:
        return ""
    fragments: list[str] = []
    for field_name in (
        "semantic_outcome",
        "acceptance",
        "public_acceptance",
        "provides",
        "engineering_worksheet",
        "implementation_steps",
        "dataflow",
    ):
        if field_name in task:
            _task_query_fragments(task.get(field_name), fragments)
    target = str(target_path or "").replace("\\", "/").strip()
    target_symbol = target.rsplit("/", 1)[-1].rsplit(".", 1)[0] if target else ""
    cleaned: list[str] = []
    for fragment in fragments:
        value = fragment
        if target_symbol:
            value = re.sub(re.escape(target_symbol), " ", value, flags=re.IGNORECASE)
        value = re.sub(r"\bAuthoredFeature\d+\b", " ", value, flags=re.IGNORECASE)
        value = " ".join(value.split()).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return " ".join(cleaned)[:768]


def _normalize_initial_task_evidence_calls(
    calls: Sequence[Any],
    *,
    query: str,
    target_path: str | None,
) -> tuple[Any, ...] | None:
    """Replace generated-self searches with the approved task's semantic query."""

    task_query = str(query or "").strip()
    if not task_query:
        return None
    target = str(target_path or "").replace("\\", "/").strip()
    target_symbol = target.rsplit("/", 1)[-1].rsplit(".", 1)[0] if target else ""
    target_folded = target_symbol.casefold()
    changed = False
    normalized: list[Any] = []
    for call in calls:
        name = str(getattr(call, "name", "") or "").strip()
        arguments = getattr(call, "arguments", None)
        if (
            name not in {"search_code_rag", "search_project_rag"}
            or not isinstance(arguments, Mapping)
        ):
            normalized.append(call)
            continue
        current_query = str(arguments.get("query") or "").strip()
        self_target = bool(
            target_folded
            and current_query
            and target_folded in current_query.casefold()
        )
        if not self_target:
            normalized.append(call)
            continue
        rebound = dict(arguments)
        rebound["query"] = task_query
        normalized.append(
            replace(
                call,
                arguments=rebound,
                raw_arguments=json.dumps(
                    rebound,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
        changed = True
    return tuple(normalized) if changed else None


def _generate_with_tools_impl(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    from .agent_capability_context import (
        project_agent_capability_context,
        reviewed_mcp_servers_for_model_role,
        skills_for_tool,
    )
    from .generation_diagnostic_repair import read_authorized_diagnostic_source
    from .grounding_policy import host_baseline_evidence_ready
    from .model_router import (
        _RAG_EVIDENCE_TOOLS,
        _execute_tool_waves,
        _external_rag_capability,
        _tool_schema_names,
        _usable_external_rag_result,
        _usable_rag_result,
    )

    messages = [dict(message) for message in request.messages]
    all_tools = tuple(request.tools)
    all_names = frozenset(_tool_schema_names(all_tools))
    reviewed_external_servers = reviewed_mcp_servers_for_model_role(stage, role)
    state = HostRunState()
    clear_generation_verification_receipt()
    unavailable_verifiers: set[str] = set()
    implementation_requires_mutation = bool(
        role in {"coder", "coder_safe"}
        and stage == "generation"
        and implementation_requested(request.messages)
    )
    host_grounded = host_baseline_evidence_ready(request.messages)
    mutation_ready = is_mutation_ready(messages, state)
    reconciled_target = _reconcile_materialized_target_from_workspace(state, runtime)
    if reconciled_target is not None:
        mutation_ready = reconciled_target.is_mutation_ready
        messages.append(_existing_target_refresh_message(reconciled_target))
        emit_root_cause(
            "generation_target_materialization_reconciled",
            stage=stage,
            operation="generate_with_tools",
            gate="mutation_target_lifecycle",
            result="PASS",
            reason="host-reserved target already exists in the bound staged workspace",
            details={
                "target_path": reconciled_target.target_path,
                "source_bytes": len((reconciled_target.source_body or "").encode("utf-8")),
            },
        )
    java_target = bool(
        implementation_requires_mutation
        and state.mutation_context
        and _canonical_mutation_path(state.mutation_context.target_path).casefold().endswith(".java")
    )
    from .small_model_task_capsule_contract import (
        current_task_required_gates,
        current_task_reuse_action,
    )
    reuse_action = current_task_reuse_action()
    fresh_java_target = bool(
        java_target
        and state.mutation_context
        and _semantic_fresh_java(
            reuse_action,
            state.mutation_context.target_path,
            materialized_new_file=state.mutation_context.is_new_file,
        )
    )
    state.semantic_fresh_java = fresh_java_target
    approved_task_evidence_query = _approved_task_evidence_query(
        request.messages,
        target_path=(
            state.mutation_context.target_path
            if state.mutation_context is not None
            else None
        ),
    )
    initial_execution_authority = _host_target_execution_authority(state)
    active_mutation_authority = CURRENT_MUTATION_AUTHORITY.get()
    bounded_root_execution_authority = bool(
        active_mutation_authority is not None
        and active_mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
    )
    authored_workspace_refresh = bool(
        bounded_root_execution_authority
        and _authored_workspace_refresh_requested(request.messages)
    )
    state.retrieval_target_binding_enabled = not authored_workspace_refresh
    compile_backed_java = bool(
        java_target
        and state.mutation_context
        and state.mutation_context.target_pinned
        and state.mutation_context.is_mutation_ready
        and "target_compile" in current_task_required_gates()
    )
    router_requires_fresh_evidence = bool(router._agent_require_fresh_evidence)
    require_rag = initial_evidence_required(
        role=role,
        host_grounded=host_grounded,
        router_requires_fresh_evidence=router_requires_fresh_evidence,
        implementation_requires_mutation=implementation_requires_mutation,
        host_target_execution_authority=initial_execution_authority,
        compile_backed_java=compile_backed_java,
        authored_workspace_refresh=authored_workspace_refresh,
        semantic_fresh_java_target=fresh_java_target,
    )
    required_evidence_choice = bool(require_rag)
    state.require_evidence = bool(require_rag)
    state.host_grounded = bool(host_grounded)
    state.compile_backed_java = bool(compile_backed_java)

    if require_rag:
        state.phase = LoopPhase.OBSERVE
    elif implementation_requires_mutation and bounded_root_execution_authority:
        # Saved authored designs intentionally delegate file selection to the coder
        # inside a host-owned bounded root set. Requiring file localization first
        # contradicts that authority, wastes retrieval turns, and can inflate the
        # mandatory conversation until it no longer fits the active llama slot.
        state.phase = LoopPhase.ACT
    elif compile_backed_java or implementation_requires_mutation and mutation_ready and not mutation_history_applied(messages):
        state.phase = LoopPhase.ACT
    else:
        state.phase = LoopPhase.OBSERVE

    last_prompt_phase = state.phase
    emit_root_cause(
        "tool_loop_initialized",
        stage=stage,
        operation="generate_with_tools",
        gate="phase_selection",
        result="PASS",
        details={
            "role": role,
            "host_grounded": host_grounded,
            "fresh_java_target": fresh_java_target,
            "router_requires_fresh_evidence": router_requires_fresh_evidence,
            "require_rag": require_rag,
            "host_target_execution_authority": initial_execution_authority,
            "bounded_root_execution_authority": bounded_root_execution_authority,
            "authored_workspace_refresh": authored_workspace_refresh,
            "implementation_requires_mutation": implementation_requires_mutation,
            "mutation_ready": mutation_ready,
            "compile_backed_java": compile_backed_java,
            "approved_task_evidence_query": approved_task_evidence_query or None,
            "initial_phase": state.phase.value,
        },
    )

    while True:
        if state.phase is LoopPhase.RECOVER and state.validation_status == "FAIL":
            snapshot = read_authorized_diagnostic_source(
                state.latest_verifier_errors,
                getattr(runtime, "workspace_root", None),
                active_mutation_authority,
            )
            if snapshot is not None:
                state.repair_target_diagnostics = tuple(snapshot["diagnostics"])
                state.mutation_context = TargetMutationContext(
                    target_path=snapshot["path"],
                    source_body=snapshot["source"],
                    base_revision_sha=snapshot["sha256"],
                    evidence_source="verifier_workspace_source",
                    writable_paths=(snapshot["path"],),
                    target_pinned=True,
                )
                # This is fresh local evidence and exact repair authority. When the
                # verifier says the platform API itself is wrong, however, local source
                # is not sufficient implementation evidence: remain in RECOVER so the
                # reviewed mappings/source/MCP frontier can resolve the exact API first.
                state.record_evidence(snapshot, usable=True)
                state.phase = _repair_phase_for_route(state.repair_evidence_route)
                emit_root_cause(
                    "verifier_workspace_repair_bound",
                    stage=stage,
                    operation="generate_with_tools",
                    gate="diagnostic_source_binding",
                    result="PASS",
                    details={
                        "target_path": snapshot["path"],
                        "source_sha256": snapshot["sha256"],
                        "source_bytes": len(snapshot["source"].encode("utf-8")),
                        "diagnostics_fingerprint": state.latest_verifier_fingerprint,
                        "repair_evidence_route": state.repair_evidence_route,
                    },
                )
        last_prompt_phase = _sync_phase_tool_transcript(
            messages, state=state, last_prompt_phase=last_prompt_phase, stage=stage
        )
        state.phase = _resume_local_repair(state)

        baseline_ready = _target_evidence_ready(
            state,
            require_rag=require_rag,
            fresh_java_target=fresh_java_target,
            compile_backed_java=compile_backed_java,
        )
        if (
            implementation_requires_mutation
            and _implementation_obligation_has_progress(state)
            and state.validation_status == "PROJECT_BUILD_DEFERRED"
            and bounded_root_execution_authority
        ):
            state.termination_reason = "VERIFICATION_DEFERRED_TO_PROJECT_BUILD"
            emit_root_cause(
                "generation_verifier_deferred_to_project_build",
                stage=stage,
                operation="generate_with_tools",
                gate="generation_verifier",
                result="SKIP",
                reason=(
                    "generation-time JDT unavailable for authored bounded-root generation; "
                    "the outer host must bind all touched paths to the mandatory project build gate"
                ),
                details={"required_gate": "project_build"},
            )
            # Do not emit a per-fragment generation-verification receipt here. The outer
            # custom-module host owns the complete touched-path set across authored fragments
            # and will synthesize the project-scoped deferred receipt after all fragments finish.
            return _host_coder_summary(verification="DEFERRED_TO_PROJECT_BUILD")

        if (
            implementation_requires_mutation
            and _implementation_obligation_has_progress(state)
            and state.validation_status == "DEFERRED"
        ):
            state.termination_reason = "VERIFICATION_DEFERRED_TO_TARGET_COMPILE"
            _record_terminal_generation_verification(
                state,
                terminal_status="DEFERRED_TO_TARGET_COMPILE",
                compile_backed_java=compile_backed_java,
            )
            emit_root_cause(
                "generation_verifier_deferred_to_required_gate",
                stage=stage,
                operation="generate_with_tools",
                gate="generation_verifier",
                result="SKIP",
                reason="generation verifier unavailable; downstream target_compile remains mandatory",
                details={
                    "target_path": (
                        state.mutation_context.target_path
                        if state.mutation_context is not None
                        else None
                    ),
                    "required_gate": "target_compile",
                },
            )
            return _host_coder_summary(verification="DEFERRED_TO_TARGET_COMPILE")

        if (
            implementation_requires_mutation
            and _implementation_obligation_has_progress(state)
            and state.validation_status == "PASS"
        ):
            state.termination_reason = "VERIFICATION_PASSED"
            _record_terminal_generation_verification(
                state,
                terminal_status="PASS",
                compile_backed_java=compile_backed_java,
            )
            return _host_coder_summary(verification="PASS")

        if state.semantic_fixed_point:
            # A repeated semantic state is already the convergence proof. Do not
            # erase it merely because a writable repair target still exists: that
            # was the bug that let ACT -> VERIFY -> FAIL cycle indefinitely.
            raise _fixed_point_error(state)

        if (
            state.phase == LoopPhase.VERIFY
            and compile_backed_java
            and state.validation_status == "COMPILE_REQUIRED"
        ):
            from .generation_compile_state import verify_compile_backed_java

            compile_status, _compile_receipt = verify_compile_backed_java(
                runtime,
                state,
                stage=stage,
            )
            if compile_status == "PASS":
                state.clear_no_progress_result()
                continue
            if compile_status == "FAIL":
                state.record_failure(
                    "target_compile",
                    "target compiler reported task-owned source defects",
                )
                current_compile_errors = tuple(state.latest_verifier_errors)
                current_context = state.mutation_context
                repair_route = repair_evidence_route_for_errors(
                    current_compile_errors,
                    local_source=(
                        current_context.source_body
                        if current_context is not None
                        else None
                    ),
                    target_path=(
                        current_context.target_path
                        if current_context is not None
                        else None
                    ),
                )
                state.repair_evidence_route = str(
                    repair_route.get("route") or "project_local"
                )
                if (
                    repair_route_requires_retrieval(state.repair_evidence_route)
                    and state.begin_recovery_evidence_epoch(
                        state.latest_verifier_fingerprint
                    )
                ):
                    emit_root_cause(
                        "recovery_evidence_epoch_started",
                        stage=stage,
                        operation="generate_with_tools",
                        gate="diagnostic_evidence_frontier",
                        result="PASS",
                        reason=(
                            "a new compiler diagnostic fingerprint reopened the "
                            "reviewed evidence frontier; initial speculative attempts "
                            "do not consume diagnostic-bound recovery"
                        ),
                        details={
                            "verifier_fingerprint": state.latest_verifier_fingerprint,
                            "repair_evidence_route": state.repair_evidence_route,
                            "target_path": (
                                state.mutation_context.target_path
                                if state.mutation_context is not None
                                else None
                            ),
                        },
                    )
                if _compile_recovery.rebase_invalid_api_candidate(
                    state,
                    runtime,
                    stage=stage,
                    fresh_java_target=fresh_java_target,
                ):
                    state.phase = LoopPhase.RECOVER
                    continue
                if state.last_verifier_quality in {"NON_IMPROVING", "UNCHANGED"}:
                    _rollback_non_improving_verifier_repair(
                        state,
                        runtime,
                        stage=stage,
                    )
                    repeated = state.record_no_progress_result(
                        {
                            "phase_before": "VERIFY",
                            "phase_after": (
                                "RECOVER"
                                if repair_route_requires_retrieval(state.repair_evidence_route)
                                else "ACT"
                            ),
                            "validation": "FAIL",
                            "target": _mutation_context_dict(state.mutation_context),
                            "verifier": {
                                "tool": "target_compile",
                                "quality": "NON_IMPROVING",
                                "baseline_error_count": state.repair_baseline_error_count,
                                "target_path": state.repair_previous_path,
                            },
                            "result": "FAIL",
                        }
                    )
                    state.phase = _repair_phase_for_route(state.repair_evidence_route)
                    if repeated:
                        raise _fixed_point_error(state)
                    continue
                state.repair_target_diagnostics = current_compile_errors
                state.clear_no_progress_result()
                state.phase = (
                    LoopPhase.RECOVER
                    if repair_route_requires_retrieval(state.repair_evidence_route)
                    else LoopPhase.ACT
                )
                continue
            state.validation_status = "DEFERRED"
            state.phase = LoopPhase.VERIFY
            continue

        state.step_index += 1
        phase_before = state.phase
        ctx_before = _mutation_context_dict(state.mutation_context)
        loc_before = (
            state.mutation_context.localization_stage.value
            if state.mutation_context else LocalizationStage.NEED_FILE.value
        )

        phase_tools = _filter_tools_for_phase(
            all_tools,
            state.phase,
            role,
            mutation_context=state.mutation_context,
            attempted_sources=state.attempted_sources,
            localization_active=implementation_requires_mutation,
            semantic_retrieval_choice=bool(require_rag and not baseline_ready),
            semantic_fresh_java_target=fresh_java_target,
            repair_evidence_route=state.repair_evidence_route,
        )
        constrained_repair_tools = _constrain_verifier_repair_tools(
            phase_tools,
            state,
        )
        if constrained_repair_tools != tuple(phase_tools):
            phase_tools = constrained_repair_tools
            emit_root_cause(
                "repair_tool_frontier_constrained",
                stage=stage,
                operation="apply_source_edit",
                gate="repair_mutation_schema",
                result="PASS",
                reason=(
                    "live HostRunState projected completion of the exact unimplemented authored source"
                    if _authored_implementation_recovery(state)
                    else "live HostRunState projected verifier repair to one host-bound bounded source window"
                ),
                details={"selected_tools": [
                    _tool_name(schema) for schema in phase_tools if _tool_name(schema)
                ]},
            )
        phase_tools = constrain_recovery_tools(
            phase_tools, state=state, repair_route=state.repair_evidence_route,
            available_tools=all_tools,
        )
        if (
            authored_workspace_refresh
            and state.phase is LoopPhase.OBSERVE
            and not state.has_fresh_evidence
        ):
            workspace_refresh_tools = tuple(
                schema
                for schema in phase_tools
                if _tool_name(schema) == "search_code_rag"
            )
            if workspace_refresh_tools:
                phase_tools = workspace_refresh_tools

        forced_verifier: str | None = None
        if state.phase == LoopPhase.VERIFY:
            forced_verifier = _verifier_tool(phase_tools, unavailable_verifiers)
            if forced_verifier is None:
                raise ModelConfigurationError(
                    "VERIFIER_UNAVAILABLE: no healthy verifier remains for the mutated workspace."
                )
            phase_tools = tuple(schema for schema in phase_tools if _tool_name(schema) == forced_verifier)

        if implementation_requires_mutation and state.phase == LoopPhase.ACT and not phase_tools:
            raise ModelConfigurationError("MUTATION_TOOL_UNAVAILABLE: no reviewed source mutation tool is exposed.")
        if implementation_requires_mutation and state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER} and not phase_tools:
            if (
                bounded_root_execution_authority
                and implementation_requires_mutation
                and state.phase is LoopPhase.OBSERVE
                and state.validation_status != "FAIL"
            ):
                # Bounded-root write authority chooses *where* the coder may write; it
                # is not API evidence.  Never convert exhausted required grounding into
                # permission to guess a fresh implementation.
                if require_rag and not baseline_ready:
                    raise ModelConfigurationError(
                        "IMPLEMENTATION_EVIDENCE_STALLED: bounded-root mutation authority "
                        "is available, but no untried authoritative Java/API evidence route remains."
                    )
                state.phase = LoopPhase.ACT
                emit_root_cause(
                    "bounded_root_localization_exhausted_resume_act",
                    stage=stage,
                    operation="generate_with_tools",
                    gate="mutation_localization",
                    result="PASS",
                    reason=(
                        "bounded-root authored generation exhausted optional localization "
                        "evidence after all required grounding obligations were satisfied"
                    ),
                    details={
                        "attempted_sources": sorted(state.attempted_sources),
                        "validation_status": state.validation_status,
                    },
                )
                continue
            mutation_is_ready = is_mutation_ready(messages, state)
            if mutation_is_ready and baseline_ready:
                state.phase = LoopPhase.ACT
                continue
            if mutation_is_ready and not baseline_ready:
                emit_root_cause(
                    "implementation_evidence_stalled",
                    stage=stage,
                    operation="generate_with_tools",
                    gate="authoritative_evidence",
                    result="FAIL",
                    reason=(
                        "mutation target is host-localized but no untried "
                        "authoritative Java/API evidence route remains"
                    ),
                    details={
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        ),
                        "fresh_java_target": fresh_java_target,
                        "compile_backed_java": compile_backed_java,
                        "validation_status": state.validation_status,
                        "attempted_sources": sorted(state.attempted_sources),
                        "last_evidence_adjudications": list(
                            state.evidence_adjudications[-8:]
                        ),
                        **recovery_state_snapshot(
                            state,
                            state.repair_evidence_route,
                        ),
                    },
                )
                raise ModelConfigurationError(
                    "IMPLEMENTATION_EVIDENCE_STALLED: the mutation target is host-localized, "
                    "but no untried authoritative Java/API evidence route remains."
                )
            raise ModelConfigurationError(
                "MUTATION_LOCALIZATION_STALLED: no untried relevant source-evidence route remains."
            )

        phase_names = frozenset(_tool_name(schema) for schema in phase_tools if _tool_name(schema))
        tool_choice = request.tool_choice
        parallel = request.parallel_tool_calls

        forced_evidence_tool: str | None = None
        forced_evidence_arguments: dict[str, Any] = {}
        if (
            state.phase is LoopPhase.RECOVER
            or (
                required_evidence_choice
                and require_rag
                and not baseline_ready
                and state.phase is LoopPhase.OBSERVE
            )
        ):
            if len(phase_names) == 1:
                forced_evidence_tool = next(iter(phase_names))
                tool_choice = {
                    "type": "function",
                    "function": {"name": forced_evidence_tool},
                }
                parallel = False
            elif len(phase_names) > 1:
                # Require evidence while preserving semantic route choice when multiple
                # reviewed retrieval tools remain available.
                tool_choice = "required"
                parallel = False
            if forced_evidence_tool in {"external_mcp_schema", "external_mcp_call"}:
                for schema in phase_tools:
                    if _tool_name(schema) != forced_evidence_tool:
                        continue
                    function = schema.get("function")
                    parameters = function.get("parameters") if isinstance(function, Mapping) else None
                    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
                    capability = properties.get("capability") if isinstance(properties, Mapping) else None
                    enum = capability.get("enum") if isinstance(capability, Mapping) else None
                    if (
                        isinstance(enum, Sequence)
                        and not isinstance(enum, (str, bytes, bytearray))
                        and len(enum) == 1
                    ):
                        forced_evidence_arguments = {"capability": str(enum[0])}
                    break
        elif state.phase == LoopPhase.ACT:
            mutation_names = [name for name in phase_names if name in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel = False
        elif forced_verifier:
            tool_choice = {"type": "function", "function": {"name": forced_verifier}}
            parallel = False
        elif state.phase == LoopPhase.RECOVER:
            tool_choice = "required"
            parallel = False

        if state.phase == LoopPhase.ACT:
            guidance = state.take_verifier_repair_guidance()
            if guidance:
                messages[:] = [
                    message for message in messages
                    if not (
                        message.get("role") == "system"
                        and str(message.get("content") or "").startswith(_REPAIR_CONTEXT_PREFIX)
                    )
                ]
                messages.append({"role": "system", "content": guidance})

        if (
            state.phase == LoopPhase.OBSERVE
            and state.mutation_context
            and fresh_java_target
            and state.mutation_context.is_mutation_ready
            and require_rag
            and not baseline_ready
        ):
            messages.append({
                "role": "system",
                "content": (
                    "The host target needs a NEW Java implementation; an existing scaffold "
                    "is not API evidence. Do not search for that generated filename. "
                    "Before writing code, retrieve task-relevant "
                    "symbols from the actual Java workspace when available, plus project conventions "
                    "or version-pinned API/mapping evidence needed to implement the requested behavior. "
                    "Do not guess Minecraft/Fabric package names from memory."
                ),
            })

        emit_root_cause(
            "phase_tools_selected",
            stage=stage,
            operation="generate_with_tools",
            gate="phase_tool_allowlist",
            result="PASS" if phase_tools else "SKIP",
            details={
                "step_index": state.step_index,
                "phase": state.phase.value,
                "selected_tools": sorted(phase_names),
                "forced_verifier": forced_verifier,
                "target": ctx_before,
            },
        )

        turn_metadata = (
            dict(request.metadata) if isinstance(request.metadata, Mapping) else {}
        )
        if (
            state.phase is LoopPhase.ACT and state.validation_status == "FAIL"
            and not _authored_implementation_recovery(state)
        ):
            try:
                existing_ceiling = int(
                    turn_metadata.get("mmm_output_token_ceiling")
                    or _VERIFIER_REPAIR_OUTPUT_TOKENS
                )
            except (TypeError, ValueError):
                existing_ceiling = _VERIFIER_REPAIR_OUTPUT_TOKENS
            turn_metadata["mmm_output_token_ceiling"] = min(
                max(1, existing_ceiling),
                _VERIFIER_REPAIR_OUTPUT_TOKENS,
            )

        turn_request = replace(
            request,
            tools=phase_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
            metadata=turn_metadata,
        )
        verifier_relative_files = target_scoped_verifier_files(
            forced_verifier, state.mutation_context
        )
        # Derive every phase from the original routing snapshot. A projection
        # must never narrow the persistent snapshot needed by a later phase.
        phase_messages = [dict(message) for message in messages]
        if role in {"coder", "coder_safe"}:
            for message in phase_messages:
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    message["content"] = project_agent_capability_context(
                        message["content"], phase_tools
                    )
        if phase_messages != messages:
            emit_root_cause(
                "phase_capability_context_projected",
                stage=stage,
                operation="generate_with_tools",
                gate="coder_input",
                result="PASS",
                details={
                    "phase": state.phase.value,
                    "selected_tools": sorted(phase_names),
                    "content_bytes_before": sum(
                        len(str(item.get("content") or "").encode("utf-8")) for item in messages
                    ),
                    "content_bytes_after": sum(
                        len(str(item.get("content") or "").encode("utf-8")) for item in phase_messages
                    ),
                },
            )
        turn_messages = _forced_act_messages(
            phase_messages,
            state=state,
            require_rag=require_rag,
            phase_names=phase_names,
            evidence_ready=_target_evidence_ready(
                state,
                require_rag=require_rag,
                fresh_java_target=fresh_java_target,
                compile_backed_java=compile_backed_java,
            ),
        )
        if len(turn_messages) != len(messages):
            emit_root_cause(
                "forced_act_context_projected",
                stage=stage,
                operation="generate_with_tools",
                gate="coder_input",
                result="PASS",
                reason="removed research/routing context after the canonical loop selected one forced ACT tool",
                details={
                    "step_index": state.step_index,
                    "selected_tools": sorted(phase_names),
                    "messages_before": len(messages),
                    "messages_after": len(turn_messages),
                    "target": ctx_before,
                },
            )
        turn = _generate_turn_with_context_recovery(
            router,
            config=config,
            adapter=adapter,
            request=turn_request,
            messages=turn_messages,
            media_paths=request.media_paths if state.step_index == 1 else (),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
            verifier_relative_files=verifier_relative_files,
        )

        recovered_repair_calls = recover_schema_rejected_verifier_repair_calls(
            turn.tool_calls,
            phase=state.phase.value,
            validation_status=str(state.validation_status or ""),
            context=state.mutation_context,
            repair_window=_repair_source_window(state),
        )
        if recovered_repair_calls is not None:
            turn = replace(turn, tool_calls=recovered_repair_calls)
            emit_root_cause(
                "verifier_repair_rejection_downprojected",
                stage=stage,
                operation="apply_source_edit",
                gate="tool_admission",
                result="PASS",
                reason="schema-rejected whole-source repair was safely down-projected to the host-selected span",
            )
        recovered_existing_calls = recover_schema_rejected_host_bound_existing_calls(
            turn.tool_calls,
            phase=state.phase.value,
            validation_status=str(state.validation_status or ""),
            context=state.mutation_context,
        )
        if recovered_existing_calls is not None:
            turn = replace(turn, tool_calls=recovered_existing_calls)
            emit_root_cause(
                "existing_source_rejection_host_normalized",
                stage=stage,
                operation="apply_source_edit",
                gate="tool_admission",
                result="PASS",
                reason=(
                    "small coder supplied valid updated source plus redundant host-owned "
                    "mutation fields; host removed those fields before exact binding"
                ),
                details={
                    "target_path": (
                        state.mutation_context.target_path
                        if state.mutation_context is not None
                        else None
                    ),
                },
            )
        normalized_evidence_calls = normalize_forced_evidence_rejection_calls(
            turn.tool_calls,
            phase_tools=phase_tools,
            forced_evidence_tool=forced_evidence_tool,
        )
        if normalized_evidence_calls is not None:
            rejected_before_normalization = tuple(turn.tool_calls)
            turn = replace(turn, tool_calls=normalized_evidence_calls)
            normalized_call = (
                normalized_evidence_calls[0]
                if normalized_evidence_calls
                else None
            )
            rejected_call = (
                rejected_before_normalization[0]
                if rejected_before_normalization
                else None
            )
            rejected_payload = (
                rejected_call.arguments
                if rejected_call is not None
                and isinstance(getattr(rejected_call, "arguments", None), Mapping)
                else {}
            )
            emit_root_cause(
                "forced_evidence_tool_host_normalized",
                stage=stage,
                operation=forced_evidence_tool or "evidence",
                gate="tool_admission",
                result="PASS",
                reason=(
                    "host rebound the rejected model tool call to the exact "
                    "host-selected evidence tool and capability"
                ),
                details={
                    "step_index": state.step_index,
                    "forced_tool": forced_evidence_tool,
                    "original_tool": rejected_payload.get("original_tool"),
                    "original_raw_arguments": str(
                        rejected_payload.get("raw_arguments") or ""
                    )[:512]
                    or None,
                    "normalized_tool": (
                        str(getattr(normalized_call, "name", "") or "")
                        if normalized_call is not None
                        else None
                    ),
                    "normalized_capability": (
                        str(
                            getattr(normalized_call, "arguments", {}).get(
                                "capability"
                            )
                            or ""
                        )
                        if normalized_call is not None
                        and isinstance(
                            getattr(normalized_call, "arguments", None),
                            Mapping,
                        )
                        else None
                    ),
                },
            )
        if (
            state.phase is LoopPhase.OBSERVE
            and fresh_java_target
            and require_rag
            and not baseline_ready
        ):
            target_path = (
                state.mutation_context.target_path
                if state.mutation_context is not None
                else None
            )
            normalized_initial_calls = _normalize_initial_task_evidence_calls(
                turn.tool_calls,
                query=approved_task_evidence_query,
                target_path=target_path,
            )
            if normalized_initial_calls is not None:
                original_queries = [
                    str(
                        call.arguments.get("query")
                        if isinstance(call.arguments, Mapping)
                        else ""
                    )
                    for call in turn.tool_calls
                ]
                turn = replace(turn, tool_calls=normalized_initial_calls)
                emit_root_cause(
                    "initial_evidence_query_host_bound",
                    stage=stage,
                    operation=(
                        str(normalized_initial_calls[0].name)
                        if normalized_initial_calls
                        else "evidence"
                    ),
                    gate="authoritative_evidence",
                    result="PASS",
                    reason=(
                        "generated-target self-search was replaced by the approved "
                        "task semantic outcome and acceptance contract"
                    ),
                    details={
                        "step_index": state.step_index,
                        "target_path": target_path,
                        "original_queries": original_queries,
                        "task_query": approved_task_evidence_query,
                    },
                )
        if state.phase is LoopPhase.RECOVER:
            recovery_diagnostics = tuple(
                state.repair_target_diagnostics or state.latest_verifier_errors
            )
            normalized_recovery_calls = normalize_recovery_evidence_calls(
                turn.tool_calls,
                errors=recovery_diagnostics,
                target_path=(
                    state.mutation_context.target_path
                    if state.mutation_context is not None
                    else None
                ),
                repair_route=state.repair_evidence_route,
            )
            if normalized_recovery_calls is not None:
                turn = replace(turn, tool_calls=normalized_recovery_calls)
                emit_root_cause(
                    "recovery_evidence_query_host_bound",
                    stage=stage,
                    operation=forced_evidence_tool or "evidence",
                    gate="tool_admission",
                    result="PASS",
                    reason=(
                        "host rebound verifier recovery search intent to the concrete "
                        "failed Java/API symbols instead of the generated target class"
                    ),
                    details={
                        "step_index": state.step_index,
                        "repair_evidence_route": state.repair_evidence_route,
                        "diagnostic_count": len(recovery_diagnostics),
                    },
                )
        turn = reject_noop_repair(
            turn, state=state, binder=_bind_existing_verifier_repair_call
        )
        rejection = _model_tool_rejection_feedback(turn.tool_calls)
        if rejection is not None:
            feedback, rejection_payloads = rejection
            for payload in rejection_payloads:
                state.record_failure(
                    str(
                        payload.get("original_tool")
                        or payload.get("rejected_name")
                        or "model_tool_call"
                    ),
                    str(
                        payload.get("error")
                        or payload.get("failure_code")
                        or "model tool call rejected"
                    ),
                )
            repeated = state.record_no_progress_result(
                _model_rejection_progress_key(
                    state,
                    rejection_payloads,
                    forced_evidence_tool=forced_evidence_tool,
                )
            )
            if forced_evidence_tool is not None:
                feedback += (
                    " The host has selected exactly one admissible evidence function for "
                    f"this turn: {forced_evidence_tool!r}. Call exactly that function; do "
                    "not replay a tool from an earlier turn."
                )
                capability = str(forced_evidence_arguments.get("capability") or "").strip()
                if capability:
                    feedback += (
                        f" Its capability is host-selected as {capability!r}; use exactly "
                        "that capability and do not substitute another one."
                    )
            messages.append({"role": "assistant", "content": turn.content or None})
            messages.append({"role": "system", "content": feedback})
            emit_root_cause(
                "model_tool_call_rejected",
                stage=stage,
                operation="generate_with_tools",
                gate="tool_admission",
                result="RETRY",
                reason="model tool call rejected before phase checks/runtime execution",
                details={
                    "step_index": state.step_index,
                    "phase": state.phase.value,
                    "rejections": rejection_payloads,
                },
            )
            if require_rag and not baseline_ready:
                required_evidence_choice = True
            if repeated:
                if (
                    state.phase is LoopPhase.OBSERVE
                    and implementation_requires_mutation
                    and is_mutation_ready(messages, state)
                    and (not require_rag or baseline_ready)
                ):
                    state.phase = LoopPhase.ACT
                    state.clear_no_progress_result()
                    continue
                raise _fixed_point_error(state)
            continue

        if (
            state.phase is LoopPhase.ACT
            and state.validation_status == "FAIL"
            and turn.tool_calls
        ):
            repair_call_names = tuple(call.name for call in turn.tool_calls)
            if (
                len(turn.tool_calls) != 1
                or repair_call_names != ("apply_source_edit",)
            ):
                repeated = state.record_no_progress_result(
                    {
                        "phase": "ACT",
                        "validation": "FAIL",
                        "repair_protocol_violation": repair_call_names,
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "REPAIR_PROTOCOL_VIOLATION: emit exactly one "
                            "apply_source_edit call for the host-pinned verifier repair. "
                            "Do not batch, parallelize, or emit any additional tool call."
                        ),
                    }
                )
                emit_root_cause(
                    "verifier_repair_protocol_rejected",
                    stage=stage,
                    operation="generate_with_tools",
                    gate="repair_mutation_schema",
                    result="RETRY",
                    reason="repair turn must contain exactly one apply_source_edit call",
                    details={
                        "tool_calls": list(repair_call_names),
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        ),
                    },
                )
                if repeated:
                    raise _fixed_point_error(state)
                continue
            bound_calls = tuple(
                _bind_existing_verifier_repair_call(call, state)
                for call in turn.tool_calls
            )
            if bound_calls != tuple(turn.tool_calls):
                turn = replace(turn, tool_calls=bound_calls)
                emit_root_cause(
                    "verifier_repair_call_host_bound",
                    stage=stage,
                    operation="apply_source_edit",
                    gate="repair_mutation_binding",
                    result="PASS",
                    reason=(
                        "model authored one bounded replacement; host bound exact old span, "
                        "target path, count, and live-SHA edit semantics"
                    ),
                    details={
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        )
                    },
                )
        host_bound_existing_calls = tuple(
            _bind_host_owned_existing_source_call(call, state)
            for call in turn.tool_calls
        )
        if host_bound_existing_calls != tuple(turn.tool_calls):
            turn = replace(turn, tool_calls=host_bound_existing_calls)
            emit_root_cause(
                "existing_source_call_host_bound",
                stage=stage,
                operation="apply_source_edit",
                gate="mutation_precondition_binding",
                result="PASS",
                reason=(
                    "recovered authored existing target uses host-owned exact live source "
                    "precondition; small coder supplied only updated source"
                ),
                details={
                    "target_path": (
                        state.mutation_context.target_path
                        if state.mutation_context is not None
                        else None
                    ),
                    "source_bytes": (
                        len((state.mutation_context.source_body or "").encode("utf-8"))
                        if state.mutation_context is not None
                        else 0
                    ),
                },
            )

        if not turn.tool_calls:
            content = turn.content.strip()
            if not content:
                raise ModelConfigurationError("Tool-capable model returned an empty final response.")
            if forced_evidence_tool is not None:
                raise ModelConfigurationError(
                    "Production coder did not honor host-forced RAG tool choice "
                    f"{forced_evidence_tool!r}; received no tool call."
                )
            if require_rag and not baseline_ready:
                repeated = state.record_no_progress_result({
                    "phase": state.phase.value,
                    "validation": state.validation_status,
                    "verifier": state.latest_verifier_fingerprint,
                    "missing_required_evidence": True,
                    "prose": content,
                })
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": (
                            "Reviewed production evidence is still required. Select exactly one "
                            "currently exposed evidence function that best matches the information "
                            "need and call it. Do not answer in prose and do not invent tool names."
                        ),
                    },
                ])
                required_evidence_choice = True
                if repeated:
                    raise _fixed_point_error(state)
                continue
            if implementation_requires_mutation and state.phase in {LoopPhase.ACT, LoopPhase.VERIFY, LoopPhase.RECOVER}:
                repeated = state.record_no_progress_result({
                    "phase": state.phase.value,
                    "validation": state.validation_status,
                    "verifier": state.latest_verifier_fingerprint,
                    "prose": content,
                })
                if repeated:
                    raise _fixed_point_error(state)
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": (
                            f"Phase {state.phase.value} requires the exposed tool action. "
                            "Prose cannot complete this implementation_requires_mutation."
                        ),
                    },
                ])
                continue
            if implementation_requires_mutation and not _implementation_obligation_has_progress(state):
                if is_mutation_ready(messages, state) and baseline_ready:
                    state.phase = LoopPhase.ACT
                    messages.append({"role": "assistant", "content": content})
                    continue
                raise ModelConfigurationError(
                    "Writable coder returned prose before a reviewed source mutation was applied."
                )
            return content

        for call in turn.tool_calls:
            if call.name not in _MUTATION_ACT_TOOLS:
                continue
            semantic_target_error = _mutation_target_error(
                call.name,
                call.arguments,
                state.mutation_context,
                state=state,
            )
            if (
                semantic_target_error
                and semantic_target_error.startswith("MUTATION_TARGET_DRIFT:")
            ):
                raise ModelConfigurationError(
                    "POST_ARGUMENT_SEMANTIC_FAILURE: " + semantic_target_error
                )

        if forced_evidence_tool is not None and (
            len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_evidence_tool
        ):
            called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
            raise ModelConfigurationError(
                "Production coder did not honor host-forced RAG tool choice "
                f"{forced_evidence_tool!r}; received {called}."
            )

        if forced_verifier and (
            len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_verifier
        ):
            called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
            raise ModelConfigurationError(
                f"VERIFIER_PROTOCOL_VIOLATION: expected {forced_verifier!r}; received {called}."
            )

        messages.append({
            "role": "assistant",
            "content": turn.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or json.dumps(
                            dict(call.arguments), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
                for call in turn.tool_calls
            ],
        })

        def is_evidence_tool(call: Any) -> bool:
            return _call_is_evidence_tool(
                call,
                state.phase,
                _RAG_EVIDENCE_TOOLS,
                _external_rag_capability,
            )

        def execute(
            call: Any, *, allowed_names: frozenset[str] = frozenset(phase_names)
        ) -> tuple[Any, Mapping[str, Any]]:
            metadata = {"skills": list(skills_for_tool(stage, call.name, model_role=role))}
            if call.name not in allowed_names:
                error = (
                    f"PHASE_PROTOCOL_VIOLATION: {call.name!r} is not allowed in "
                    f"{state.phase.value}; allowed={sorted(allowed_names)}"
                )
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": "PHASE_PROTOCOL_VIOLATION",
                    "error": error,
                }

            if is_evidence_tool(call) and state.is_query_attempted(call.name, call.arguments):
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": "DUPLICATE_QUERY",
                    "error": "RetrievalNoProgress: equivalent query already attempted",
                }

            target_error = _mutation_target_error(
                call.name, call.arguments, state.mutation_context, state=state
            )
            if target_error:
                failure_code = target_error.partition(":")[0]
                if (
                    failure_code == "REPAIR_SEMANTIC_FOOTPRINT_VIOLATION"
                    and state.mutation_context is not None
                    and existing_java_structurally_subsumes_candidate(
                        _canonical_mutation_path(state.mutation_context.target_path),
                        state.mutation_context.source_body,
                        call.arguments.get("new"),
                    )
                ):
                    return call, {
                        "ok": True,
                        "tool": call.name,
                        **metadata,
                        "semantic_noop": True,
                        "result": {
                            "status": "PRESERVED_EXISTING_SUPERSET",
                            "target_path": _canonical_mutation_path(
                                state.mutation_context.target_path
                            ),
                        },
                    }
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": failure_code,
                    "error": target_error,
                }

            try:
                if call.name.startswith("external_mcp_"):
                    state.record_source_attempt(call.name, call.arguments)
                    scoped = getattr(runtime, "call_scoped", None)
                    if not callable(scoped):
                        raise ModelConfigurationError(
                            "External MCP execution requires a role-scoped runtime."
                        )
                    result = scoped(
                        stage,
                        call.name,
                        call.arguments,
                        external_server_ids=reviewed_external_servers,
                    )
                else:
                    result = runtime.call(stage, call.name, call.arguments)
                if is_evidence_tool(call):
                    state.record_query(call.name, call.arguments)
                    state.record_source_attempt(call.name, call.arguments)
                return call, {
                    "ok": True,
                    "tool": call.name,
                    **metadata,
                    "result": result,
                }
            except Exception as exc:  # noqa: BLE001 - tool failures become typed recovery observations
                if is_evidence_tool(call):
                    state.record_query(call.name, call.arguments)
                    state.record_source_attempt(call.name, call.arguments)
                error = f"{type(exc).__name__}: {exc}"
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": _runtime_failure_code(call.name, error),
                    "error": error,
                }

        executed = _execute_tool_waves(tuple(turn.tool_calls), execute)
        progress = False
        tentative_repair_applied = False

        for call, payload in executed:
            record_discovery(
                state,
                call,
                payload,
                external_rag_capability=_external_rag_capability,
            )
            if str(call.name).startswith("external_mcp_"):
                mcp_state = recovery_state_snapshot(
                    state,
                    state.repair_evidence_route,
                )
                emit_root_cause(
                    "external_mcp_frontier_state",
                    stage=stage,
                    operation=str(call.name),
                    gate="external_mcp_recovery_frontier",
                    result="PASS" if bool(payload.get("ok")) else "SKIP",
                    reason=(
                        "external MCP frontier after this tool observation; "
                        "completed capabilities are ineligible for reselection"
                    ),
                    details={
                        "step_index": state.step_index,
                        "requested_capability": (
                            str(call.arguments.get("capability") or "")
                            if isinstance(call.arguments, Mapping)
                            else ""
                        )
                        or None,
                        **mcp_state,
                    },
                )
            messages.append(dict(bounded_tool_message(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                },
                config=config,
                tools=request.tools,
            )))

            if call.name in _MUTATION_ACT_TOOLS:
                if bool(payload.get("semantic_noop")):
                    state.preserved_existing_source = True
                    state.unchanged_mutation_fingerprints.clear()
                    state.unapplied_mutation_fixed_point = False
                    state.semantic_fixed_point = False
                    state.clear_failure()
                    state.clear_no_progress_result()
                    progress = True
                    if compile_backed_java:
                        state.validation_status = "COMPILE_REQUIRED"
                        state.phase = LoopPhase.VERIFY
                    else:
                        state.phase = (
                            LoopPhase.VERIFY
                            if all_names & _VERIFY_TOOLS
                            else LoopPhase.OBSERVE
                        )
                        if not all_names & _VERIFY_TOOLS:
                            state.validation_status = "PASS"
                    emit_root_cause(
                        "authored_existing_source_preserved",
                        stage=stage,
                        operation=call.name,
                        gate="cumulative_authored_semantics",
                        result="PASS",
                        reason=(
                            "candidate would only reduce an already-materialized Java "
                            "structure; existing cumulative source was preserved for verification"
                        ),
                        details={
                            "target_path": (
                                state.mutation_context.target_path
                                if state.mutation_context is not None
                                else None
                            ),
                            "workspace_changed": state.workspace_changed,
                        },
                    )
                    continue
                applied = state.record_mutation(call.name, call.arguments, payload)
                if applied:
                    repair_candidate_pending_verification = (
                        state.repair_baseline_error_count is not None
                    )
                    if not repair_candidate_pending_verification:
                        progress = True
                    else:
                        tentative_repair_applied = True
                        emit_root_cause(
                            "verifier_repair_candidate_applied",
                            stage=stage,
                            operation=call.name,
                            gate="repair_quality_monotonicity",
                            result="SKIP",
                            reason=(
                                "repair mutation is tentative until the verifier proves "
                                "strict diagnostic improvement"
                            ),
                            details={
                                "target_path": state.repair_previous_path,
                                "baseline_error_count": state.repair_baseline_error_count,
                            },
                        )
                    state.clear_failure()
                    operation = str(call.arguments.get("operation") or "").strip().casefold()
                    if operation in _SOURCE_CREATE_OPERATIONS:
                        materialized_path = ""
                        for key in _SOURCE_EDIT_PATH_KEYS:
                            value = call.arguments.get(key)
                            if isinstance(value, str) and value.strip():
                                materialized_path = _canonical_mutation_path(value)
                                break
                        if materialized_path:
                            messages.append({
                                "role": "system",
                                "content": (
                                    "MMM_TARGET_MATERIALIZED_V1\n"
                                    f"The host has materialized {materialized_path!r}. It is now an existing "
                                    "workspace file. Any earlier host_reserved/fresh creation status describes "
                                    "only the pre-create lifecycle. Future edits must follow the current "
                                    "host-selected target and exposed tool schema. For existing-source "
                                    "repairs, use an admitted non-create edit such as replace_exact."
                                ),
                            })
                    if compile_backed_java:
                        state.validation_status = "COMPILE_REQUIRED"
                        state.phase = LoopPhase.VERIFY
                    else:
                        state.phase = (
                            LoopPhase.VERIFY if all_names & _VERIFY_TOOLS else LoopPhase.OBSERVE
                        )
                        if not all_names & _VERIFY_TOOLS:
                            state.validation_status = "PASS"
                else:
                    code = str(payload.get("failure_code") or "")
                    error = str(payload.get("error") or "MUTATION_UNCHANGED: no source-byte change")
                    state.record_failure(call.name, error)
                    if code == "MUTATION_STALE_PRECONDITION":
                        refreshed = _recover_stale(
                            state, getattr(runtime, "workspace_root", None), _source_edit_path(call.arguments),
                            _reconcile_materialized_target_from_workspace(state, runtime), TargetMutationContext,
                        )
                        if refreshed is not None:
                            state.repair_guidance_fingerprint = None
                            messages.append(_existing_target_refresh_message(refreshed))
                        state.phase = (
                            LoopPhase.ACT
                            if state.mutation_context and state.mutation_context.is_mutation_ready
                            else LoopPhase.OBSERVE
                        )
                    elif code in {
                        "MUTATION_TARGET_CREATION_CONFLICT",
                        "MUTATION_TARGET_ALREADY_EXISTS",
                    }:
                        refreshed = _recover_creation_conflict_target(
                            state,
                            runtime,
                            call.arguments,
                        )
                        if refreshed is not None:
                            state.repair_guidance_fingerprint = None
                            state.clear_failure()
                            messages.append(_existing_target_refresh_message(refreshed))
                            state.phase = LoopPhase.ACT
                            progress = True
                            emit_root_cause(
                                "mutation_creation_conflict_rebound",
                                stage=stage,
                                operation=call.name,
                                gate="mutation_target_lifecycle",
                                result="PASS",
                                reason=(
                                    "failed create target already exists; host rebound the exact "
                                    "live source as an existing-file edit target"
                                ),
                                details={
                                    "target_path": refreshed.target_path,
                                    "source_bytes": len(
                                        (refreshed.source_body or "").encode("utf-8")
                                    ),
                                },
                            )
                        elif state.mutation_context and state.mutation_context.is_mutation_ready:
                            state.phase = LoopPhase.ACT
                        else:
                            state.phase = LoopPhase.OBSERVE
                    elif code in {
                        "MUTATION_TARGET_DRIFT",
                        "MUTATION_TARGET_UNBOUND",
                        "REPAIR_ATOMIC_REPLACEMENT_TOO_LARGE",
                        "REPAIR_ATOMIC_SCOPE_VIOLATION",
                        "PHASE_PROTOCOL_VIOLATION",
                    }:
                        if state.mutation_context and state.mutation_context.is_mutation_ready:
                            state.phase = LoopPhase.ACT
                        else:
                            state.phase = LoopPhase.OBSERVE
                    elif bool(payload.get("ok")):
                        state.phase = LoopPhase.ACT
                continue

            if call.name in _VERIFY_TOOLS:
                status = _verification_outcome(call.name, payload)
                emit_root_cause(
                    "verification_adjudicated",
                    stage=stage,
                    operation=call.name,
                    gate="generation_verifier",
                    result=(
                        "PASS"
                        if status == "PASS"
                        else "FAIL"
                        if status in {"FAIL", "UNAVAILABLE"}
                        else "SKIP"
                    ),
                    reason=status,
                    details={
                        "status": status,
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        ),
                    },
                )
                if status in {"VERIFIER_ARGUMENT_INVALID", "VERIFIER_TARGET_INVALID"}:
                    raise ModelConfigurationError(
                        f"{status}: {payload.get('error', '')}"
                    )
                if status == "UNAVAILABLE":
                    state.record_failure(call.name, payload.get("error", "verifier unavailable"))
                    unavailable_verifiers.add(call.name)
                    if (
                        bounded_root_execution_authority
                        and implementation_requires_mutation
                        and _implementation_obligation_has_progress(state)
                    ):
                        # Authored-design generation intentionally has no exact task target.
                        # A JDT infrastructure failure (for example Loom dependency download)
                        # must not erase already-authored source. Defer verification to the
                        # project-scoped build gate, which the outer host binds to every
                        # touched path after all authored fragments finish.
                        state.validation_status = "PROJECT_BUILD_DEFERRED"
                    else:
                        state.validation_status = "UNAVAILABLE"
                    state.phase = LoopPhase.VERIFY
                    continue
                verification_progress = state.record_verification(
                    call.name,
                    payload,
                    status,
                )
                emit_root_cause(
                    "verification_quality_adjudicated",
                    stage=stage,
                    operation=call.name,
                    gate="repair_quality_monotonicity",
                    result="PASS" if verification_progress else "SKIP",
                    reason=state.last_verifier_quality or "UNKNOWN",
                    details={
                        "quality": state.last_verifier_quality,
                        "baseline_error_count": state.repair_baseline_error_count,
                        "current_error_count": len(state.latest_verifier_errors),
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        ),
                    },
                )
                if verification_progress:
                    progress = True
                if status == "FAIL" and implementation_requires_mutation:
                    state.record_failure(call.name, "verification reported source defects")
                    current_context = state.mutation_context
                    repair_route = repair_evidence_route_for_errors(
                        state.latest_verifier_errors,
                        local_source=(
                            current_context.source_body
                            if current_context is not None
                            else None
                        ),
                        target_path=(
                            current_context.target_path
                            if current_context is not None
                            else None
                        ),
                    )
                    state.repair_evidence_route = str(
                        repair_route.get("route") or "project_local"
                    )
                    if (
                        repair_route_requires_retrieval(state.repair_evidence_route)
                        and state.begin_recovery_evidence_epoch(
                            state.latest_verifier_fingerprint
                        )
                    ):
                        emit_root_cause(
                            "recovery_evidence_epoch_started",
                            stage=stage,
                            operation="generate_with_tools",
                            gate="diagnostic_evidence_frontier",
                            result="PASS",
                            reason=(
                                "a new verifier diagnostic fingerprint reopened the "
                                "reviewed evidence frontier; initial speculative attempts "
                                "do not consume diagnostic-bound recovery"
                            ),
                            details={
                                "verifier_fingerprint": state.latest_verifier_fingerprint,
                                "repair_evidence_route": state.repair_evidence_route,
                                "target_path": (
                                    state.mutation_context.target_path
                                    if state.mutation_context is not None
                                    else None
                                ),
                            },
                        )
                    if state.last_verifier_quality in {"NON_IMPROVING", "UNCHANGED"}:
                        _rollback_non_improving_verifier_repair(
                            state,
                            runtime,
                            stage=stage,
                        )
                    state.phase = LoopPhase.RECOVER
                continue

            if is_evidence_tool(call):
                if not bool(payload.get("ok")):
                    state.record_failure(call.name, payload.get("error", "evidence tool failed"))
                    continue
                if call.name in _RAG_EVIDENCE_TOOLS:
                    usable = _usable_rag_result(payload.get("result"))
                elif call.name == "external_mcp_call":
                    usable = _usable_external_rag_result(call.arguments, payload.get("result"))
                else:
                    usable = bool(payload.get("result"))
                before = _mutation_context_dict(state.mutation_context)
                recorded = state.record_evidence(
                    payload.get("result"),
                    usable=usable,
                )
                adjudication = _evidence_call_adjudication(
                    state,
                    call,
                    payload,
                    semantic_usable=usable,
                    recorded=recorded,
                )
                state.record_evidence_adjudication(adjudication)
                emit_root_cause(
                    "evidence_adjudicated",
                    stage=stage,
                    operation=call.name,
                    gate="authoritative_evidence",
                    result=(
                        "PASS"
                        if adjudication["evidence_ready"]
                        else "SKIP"
                    ),
                    reason=(
                        "EVIDENCE_OBLIGATION_SATISFIED"
                        if adjudication["evidence_ready"]
                        else str(adjudication["authoritative_reason"])
                    ),
                    details=adjudication,
                )
                after = _mutation_context_dict(state.mutation_context)
                localization_progress = before != after
                evidence_progress = bool(recorded and usable)
                if localization_progress or evidence_progress:
                    progress = True
                    if (
                        authored_workspace_refresh
                        and bounded_root_execution_authority
                        and evidence_progress
                    ):
                        # This refresh is evidence only; ACT retains bounded-root file choice.
                        state.retrieval_target_binding_enabled = True
                        state.phase = LoopPhase.ACT
                    elif (
                        implementation_requires_mutation
                        and context_is_localized(state.mutation_context)
                        and _target_evidence_ready(
                            state,
                            require_rag=require_rag,
                            fresh_java_target=fresh_java_target,
                            compile_backed_java=compile_backed_java,
                        )
                    ):
                        state.phase = LoopPhase.ACT
                continue

        phase_after = state.phase.value
        ctx_after = _mutation_context_dict(state.mutation_context)
        loc_after = (
            state.mutation_context.localization_stage.value
            if state.mutation_context else LocalizationStage.NEED_FILE.value
        )
        call_info = [{"name": call.name, "arguments": dict(call.arguments)} for call in turn.tool_calls]
        fixed_point_calls = _fixed_point_tool_calls(turn.tool_calls)
        result_info = _fixed_point_tool_results(executed)

        if progress:
            state.clear_no_progress_result()
            if _target_evidence_ready(
                state,
                require_rag=require_rag,
                fresh_java_target=fresh_java_target,
                compile_backed_java=compile_backed_java,
            ):
                required_evidence_choice = False
        elif tentative_repair_applied:
            # A repair write is neither success nor no-progress until VERIFY measures
            # it. Preserve prior no-progress fingerprints so repeated verifier states
            # can converge, but never terminate on an unverified candidate source.
            pass
        else:
            verifier_progress_key: Any = state.latest_verifier_fingerprint
            if (
                phase_before is LoopPhase.VERIFY
                and state.last_verifier_quality == "NON_IMPROVING"
            ):
                verifier_progress_key = {
                    "quality": "NON_IMPROVING",
                    "baseline_error_count": state.repair_baseline_error_count,
                    "target_path": (
                        state.repair_previous_path
                        or (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        )
                    ),
                }
            state.record_no_progress_result({
                "phase_before": phase_before.value,
                "phase_after": phase_after,
                "localization_before": loc_before,
                "localization_after": loc_after,
                "target": ctx_after,
                "validation": state.validation_status,
                "verifier": verifier_progress_key,
                "calls": fixed_point_calls,
                "results": result_info,
            })
            if require_rag and not _target_evidence_ready(
                state,
                require_rag=require_rag,
                fresh_java_target=fresh_java_target,
                compile_backed_java=compile_backed_java,
            ):
                # Required grounding is a hard gate.  Multiple empty/control-plane
                # retrieval turns must advance to another evidence route or fail
                # closed; they must never silently downgrade into ACT.
                required_evidence_choice = True

        trace = ExecutionStepTrace(
            step_index=state.step_index,
            phase_before=phase_before.value,
            localization_stage_before=loc_before,
            mutation_context_before=ctx_before,
            exposed_tools=sorted(phase_names),
            tool_choice=tool_choice,
            input_messages_count=len(messages),
            model_response_content=turn.content or None,
            model_tool_calls=call_info,
            query_signatures=[retrieval_query_signature(call.name, call.arguments) for call in turn.tool_calls],
            tool_results=result_info,
            mutation_context_after=ctx_after,
            localization_stage_after=loc_after,
            phase_after=phase_after,
            turn_made_progress=progress,
            no_progress_streak_after=state.no_progress_streak,
            action_decision="tool_wave_executed",
        )
        state.trajectory.append(trace)
        emit_root_cause(
            "tool_loop_step_result",
            stage=stage,
            operation="generate_with_tools",
            gate="progress_adjudication",
            result="PASS" if progress else "SKIP",
            reason=(
                "progress"
                if progress
                else "pending_verification"
                if tentative_repair_applied
                else "no_progress"
            ),
            details=trace.to_dict(),
        )


@task_capsule_tool_loop
def generate_with_tools(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    runtime: Any,
    stage: str,
    role: str,
) -> str:
    with trace_scope("tool_loop"):
        emit_root_cause(
            "pipeline_boundary_start",
            stage=stage,
            operation="generate_with_tools",
            gate="coder_tool_loop",
            result="START",
            details={"role": role, "request_tools": request.tools},
        )
        try:
            value = _generate_with_tools_impl(
                router,
                config=config,
                adapter=adapter,
                request=request,
                runtime=runtime,
                stage=stage,
                role=role,
            )
        except BaseException as exc:
            emit_root_cause(
                "pipeline_boundary_failure",
                stage=stage,
                operation="generate_with_tools",
                gate="coder_tool_loop",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                exc=exc,
            )
            raise
        emit_root_cause(
            "pipeline_boundary_result",
            stage=stage,
            operation="generate_with_tools",
            gate="coder_tool_loop",
            result="PASS",
            reason="coder/tool loop completed",
            details={"output": value},
        )
        return value


_mmm_planir_mutation_authority_v1 = True
_mmm_repair_mutation_recovery_v1 = True
_mmm_mutation_authority_final_guard_v1 = True
_mmm_post_argument_semantic_boundary_v1 = True

__all__ = [
    "_LOCALIZATION_EVIDENCE_TOOLS",
    "_MUTATION_ACT_TOOLS",
    "_READ_OBSERVE_TOOLS",
    "_RECOVERY_EVIDENCE_TOOLS",
    "_VERIFY_TOOLS",
    "ExecutionStepTrace",
    "HostRunState",
    "LocalizationStage",
    "LoopPhase",
    "RetrievalDecision",
    "RetrievalNoProgressError",
    "RetrievalObservation",
    "RetrievalProgress",
    "TargetMutationContext",
    "_atomic_output_recovery_instruction",
    "_fixed_point_tool_results",
    "_model_tool_rejection_feedback",
    "clear_generation_verification_receipt",
    "current_generation_verification_receipt",
    "evidence_fingerprint",
    "format_trajectory_summary",
    "generate_with_tools",
    "is_mutation_ready",
    "normalize_retrieval_query",
    "retrieval_query_signature",
    "retrieval_source_key",
]
