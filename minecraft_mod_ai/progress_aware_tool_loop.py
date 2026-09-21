from __future__ import annotations

"""Single host-owned coder execution state machine.

This module deliberately owns target localization, source mutation, verifier repair,
retrieval progress, and convergence in one place. Runtime bootstrap must not monkey-patch
these boundaries from separate contract modules.
"""

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
from .owned_target_contract import (
    normalize_target_status,
    target_is_creatable,
    target_is_existing,
    target_is_writable,
)
from .root_cause_trace import emit_root_cause, trace_scope
from .small_model_task_capsule_contract import task_capsule_tool_loop
from .source_mutation_contract import mutation_history_applied, mutation_payload_applied
from .value_shapes import as_sequence as _sequence
from .value_shapes import structured_payload as _structured_payload


class LoopPhase(str, Enum):
    OBSERVE = "OBSERVE"
    ACT = "ACT"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"


class RetrievalDecision(str, Enum):
    EXECUTE = "EXECUTE"
    DUPLICATE_QUERY = "DUPLICATE_QUERY"


class RetrievalObservation(str, Enum):
    FRESH = "FRESH"
    DUPLICATE_EVIDENCE = "DUPLICATE_EVIDENCE"
    WEAK = "WEAK"


class LocalizationStage(str, Enum):
    NEED_FILE = "NEED_FILE"
    NEED_SYMBOL = "NEED_SYMBOL"
    NEED_BODY = "NEED_BODY"
    READY = "READY"


_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "where", "which", "with",
})
_VOLATILE_EVIDENCE_KEYS = frozenset({
    "coverage_score", "correction", "elapsed_ms", "generated_at", "latency_ms",
    "normalized_query", "query", "relevance_score", "request_id", "result_count",
    "timestamp", "trace_id",
})
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
_MUTATION_ACT_TOOLS = frozenset({
    "apply_source_edit",
    "apply_source_patch",
    "apply_java_operations",
    "repair_project",
})
_VERIFY_TOOLS = frozenset({
    "java_diagnostics",
    "jdt_diagnostics",
    "run_gametest",
})
_VERIFIER_UNAVAILABLE_STATUSES = frozenset({
    "UNAVAILABLE", "NOT_RUN", "TIMEOUT", "TIMED_OUT", "UNHEALTHY",
})
_VERIFIER_FAIL_STATUSES = frozenset({"FAIL", "FAILED", "ERROR", "INVALID"})
_VERIFIER_PASS_STATUSES = frozenset({
    "PASS", "PASSED", "OK", "SUCCESS", "SUCCEEDED", "AVAILABLE",
})
_SOURCE_EDIT_PATH_KEYS = ("path", "file", "target_path", "target_file")
_SOURCE_CREATE_OPERATIONS = frozenset({
    "create", "create_file", "create_java_type", "create_class", "create_type",
    "write", "write_file",
})
_SOURCE_ATOMIC_REWRITE_OPERATIONS = frozenset({"create", "create_file", "write", "write_file"})
_REPAIR_CONTEXT_PREFIX = "MMM_CORE_VERIFIER_REPAIR_"
_MODEL_REJECTION_TOOL_NAME = "__mmm_rejected_tool_call__"
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


_EXISTING_TARGET_EVIDENCE_SOURCES = frozenset({
    "host_exact_source",
    "mutation_receipt",
    "search_code_rag",
    "workspace_existing_target",
    "verifier_workspace_source",
})
_CODE_MARKERS = frozenset({
    "class ", "interface ", "enum ", "record ", "public ", "private ", "protected ",
    "package ", "import ", "void ", "return ", "final ", "static ", "new ",
    "extends ", "implements ", "override", "{", "}", ";", "(", ")",
})
def _tool_name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


def _canonical_mutation_path(value: Any) -> str:
    clean = str(value or "").strip().replace("\\", "/")
    return re.sub(r"^(?:\./)+", "", clean)


def _normalized_target_path(value: Any) -> str:
    return _canonical_mutation_path(value)


def _is_workspace_file_path(path: str) -> bool:
    clean = _canonical_mutation_path(path)
    if not clean or clean.startswith("/") or ":" in clean:
        return False
    if ".." in Path(clean).parts:
        return False
    return "/" in clean or Path(clean).suffix.casefold() in {
        ".java", ".json", ".toml", ".gradle", ".properties", ".txt", ".md",
        ".kt", ".groovy",
    }


def _is_code_bearing_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 15:
        return False
    lower = stripped.casefold()
    return any(marker in lower for marker in _CODE_MARKERS)


def normalize_retrieval_query(value: Any) -> str:
    text = str(value or "").casefold()
    tokens = re.findall(r"[a-z0-9_.$:/+-]+", text)
    return " ".join(sorted({token for token in tokens if token not in _STOPWORDS}))


def retrieval_source_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    if name in {"external_mcp_schema", "external_mcp_call"}:
        capability = str(arguments.get("capability", "")).strip()
        return f"{name}:{capability}" if capability else name
    return name


def retrieval_query_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    parts = [name]
    if name in {"external_mcp_schema", "external_mcp_call"}:
        capability = str(arguments.get("capability", "")).strip().casefold()
        if capability:
            parts.append(f"capability={capability}")
    query = normalize_retrieval_query(arguments.get("query"))
    for key in ("index_path", "path", "file", "target_path", "symbol", "symbol_name"):
        value = str(arguments.get(key) or "").strip().casefold()
        if value:
            label = "target" if key in {"index_path", "path", "file", "target_path"} else key
            parts.append(f"{label}={value}")
    if query:
        parts.append(f"q={query}")
    cursor = arguments.get("cursor") or arguments.get("offset_bytes")
    if cursor not in (None, "", 0, "0"):
        parts.append(f"cursor={cursor}")
    return ":".join(parts)


def _stable_value(value: Any, *, drop_volatile: bool = True) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if drop_volatile and key.casefold() in _VOLATILE_EVIDENCE_KEYS:
                continue
            stable = _stable_value(child, drop_volatile=drop_volatile)
            if stable not in (None, "", [], {}):
                out[key] = stable
        return out
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item, drop_volatile=drop_volatile) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item, drop_volatile=drop_volatile) for item in value]
    return value


def evidence_fingerprint(value: Any) -> str | None:
    stable = _stable_value(value)
    if stable in (None, "", [], {}):
        return None
    canonical = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_task_from_module(module: Any) -> Mapping[str, Any] | None:
    if not isinstance(module, Mapping):
        return None
    direct = module.get("evidence_task")
    if isinstance(direct, Mapping):
        return direct
    config = module.get("config")
    if isinstance(config, Mapping):
        nested = config.get("evidence_task")
        if isinstance(nested, Mapping):
            return nested
    return None


def _anchor_path_and_symbol(anchor: Mapping[str, Any]) -> tuple[str, str]:
    locator = str(anchor.get("locator") or "").replace("\\", "/").strip()
    raw_path, sep, symbol = locator.partition("#")
    path = _canonical_mutation_path(raw_path)
    if not _is_workspace_file_path(path):
        return "", ""
    return path, symbol.strip() if sep else ""


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _anchor_candidate(
    anchor: Mapping[str, Any],
    *,
    reuse: str,
) -> tuple[str, str, bool, bool] | None:
    path, symbol = _anchor_path_and_symbol(anchor)
    if not path:
        return None
    status = normalize_target_status(anchor.get("status"))
    if status and not target_is_writable(status):
        return None
    if target_is_existing(status):
        fresh = False
    elif reuse:
        fresh = reuse == "fresh"
    else:
        fresh = target_is_creatable(status)
    symbolic = str(anchor.get("kind") or "").strip().casefold() == "symbol"
    return path, symbol, fresh, symbolic


def _collect_task_anchors(
    anchors: Sequence[Any],
    *,
    reuse: str,
    candidates: list[tuple[str, str, bool]],
    writable: list[str],
    creatable: list[str],
) -> None:
    for raw_anchor in anchors:
        if not isinstance(raw_anchor, Mapping):
            continue
        candidate = _anchor_candidate(raw_anchor, reuse=reuse)
        if candidate is None:
            continue
        path, symbol, fresh, symbolic = candidate
        _append_unique(writable, path)
        reserved = target_is_creatable(raw_anchor.get("status"))
        if fresh or reserved:
            _append_unique(creatable, path)
        if symbolic:
            candidates.append((path, symbol, fresh))


def _collect_authority_candidates(
    task: Mapping[str, Any] | None,
    *,
    direct_reuse: str,
    writable: list[str],
    creatable: list[str],
) -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    if not isinstance(task, Mapping):
        return candidates
    task_reuse = str(task.get("reuse_action") or "").strip().casefold()
    for raw_binding in _sequence(task.get("production_bindings")):
        if not isinstance(raw_binding, Mapping):
            continue
        reuse = str(
            raw_binding.get("reuse_action") or task_reuse or direct_reuse
        ).strip().casefold()
        _collect_task_anchors(
            _sequence(raw_binding.get("owned_anchors")),
            reuse=reuse,
            candidates=candidates,
            writable=writable,
            creatable=creatable,
        )
    task_candidates = candidates if not candidates else []
    _collect_task_anchors(
        _sequence(task.get("owned_anchors")),
        reuse=task_reuse or direct_reuse,
        candidates=task_candidates,
        writable=writable,
        creatable=creatable,
    )
    return candidates


def _choose_authority_candidate(
    direct_primary: str,
    direct_reuse: str,
    candidates: Sequence[tuple[str, str, bool]],
) -> tuple[str, str, bool] | None:
    if direct_primary:
        matching = next((item for item in candidates if item[0] == direct_primary), None)
        return matching or (direct_primary, "", direct_reuse == "fresh")
    unique = tuple(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _task_anchor_status(task: Mapping[str, Any] | None, path: str) -> str:
    if not isinstance(task, Mapping):
        return ""
    groups: list[Any] = []
    for binding in _sequence(task.get("production_bindings")):
        if isinstance(binding, Mapping):
            groups.extend(_sequence(binding.get("owned_anchors")))
    groups.extend(_sequence(task.get("owned_anchors")))
    for anchor in groups:
        if not isinstance(anchor, Mapping):
            continue
        anchor_path, _symbol = _anchor_path_and_symbol(anchor)
        if anchor_path == path:
            status = normalize_target_status(anchor.get("status"))
            if status:
                return status
    return ""


def _task_authority_context(payload: Mapping[str, Any]) -> TargetMutationContext | None:
    task = _evidence_task_from_module(payload.get("module"))
    direct_primary = _canonical_mutation_path(payload.get("primary_path"))
    direct_reuse = str(payload.get("reuse_action") or "").strip().casefold()
    writable = [
        path
        for item in _sequence(payload.get("writable_paths"))
        if _is_workspace_file_path(path := _canonical_mutation_path(item))
    ]
    creatable: list[str] = []
    candidates = _collect_authority_candidates(
        task,
        direct_reuse=direct_reuse,
        writable=writable,
        creatable=creatable,
    )
    if direct_primary:
        _append_unique(writable, direct_primary)
    chosen = _choose_authority_candidate(direct_primary, direct_reuse, candidates)
    if chosen is None:
        return None
    path, symbol, fresh = chosen
    _append_unique(writable, path)
    if fresh:
        _append_unique(creatable, path)
    status = _task_anchor_status(task, path)
    if isinstance(task, Mapping) and target_is_existing(status):
        evidence_source = "evidence_existing_owned_anchor"
    elif fresh and isinstance(task, Mapping):
        evidence_source = "evidence_fresh_owned_anchor"
    elif isinstance(task, Mapping) and target_is_creatable(status):
        evidence_source = "evidence_host_reserved_owned_anchor"
    else:
        evidence_source = "host_task_authority"
    return TargetMutationContext(
        target_path=path,
        target_symbol=symbol or None,
        is_new_file=fresh,
        evidence_source=evidence_source,
        writable_paths=tuple(writable),
        creatable_paths=tuple(creatable),
        target_pinned=True,
    )


def _planir_owned_anchor_sets(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the exact host-owned writable/creatable PlanIR paths."""

    context = _task_authority_context(payload)
    if context is None:
        return (), ()
    return context.writable_paths, context.creatable_paths


def _approved_donor_source_authority(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    from .donor_source_authority import approved_donor_authority

    return approved_donor_authority(messages)


def _filter_donor_tool_schemas(schemas: Sequence[Any]) -> tuple[Any, ...]:
    from .donor_source_authority import filter_donor_tool_schemas

    return filter_donor_tool_schemas(schemas)


def _constrain_existing_repair_schema(
    schema: Mapping[str, Any],
    *,
    target_path: str,
) -> Mapping[str, Any]:
    """Project verifier repair onto one host-bound atomic whole-file rewrite.

    The model authors only the corrected source body. Operation, path, and optimistic
    concurrency are host-owned so a small model cannot replay a stale whole-file
    precondition.
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
    new_schema["type"] = "string"
    new_schema["description"] = (
        "Complete corrected contents of the host-pinned existing source file. "
        f"The host binds operation=replace_exact and path={target_path!r}, reads the "
        "live file at execution time, and applies this content with its live SHA."
    )
    parameters["properties"] = {"new": new_schema}
    parameters["required"] = ["new"]
    parameters["additionalProperties"] = False
    function["description"] = (
        "Repair the verifier-selected existing source file atomically. Emit only the "
        "complete corrected source in new; operation, path, old text, and SHA are host-owned."
    )
    return cloned


def _bind_existing_verifier_repair_call(
    call: Any,
    state: Any,
) -> Any:
    """Bind a model-authored repair body to the live host-selected target."""

    if str(getattr(call, "name", "") or "").strip() != "apply_source_edit":
        return call
    if str(getattr(state, "validation_status", "") or "") != "FAIL":
        return call
    context = getattr(state, "mutation_context", None)
    if context is None or context.is_new_file or not context.is_mutation_ready:
        return call
    target = _canonical_mutation_path(context.target_path)
    if not target:
        return call
    raw_arguments = getattr(call, "arguments", None)
    if not isinstance(raw_arguments, Mapping):
        return call
    new_source = raw_arguments.get("new")
    if not isinstance(new_source, str):
        return call
    bound = {
        "operation": "replace_exact",
        "path": target,
        "new": new_source,
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
    context = getattr(state, "mutation_context", None)
    if context is None or context.is_new_file or not context.is_mutation_ready:
        return tuple(tools)
    target_path = _canonical_mutation_path(context.target_path)
    if not target_path:
        return tuple(tools)

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
            _constrain_existing_repair_schema(schema, target_path=target_path)
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


@dataclass(frozen=True)
class TargetMutationContext:
    target_path: str | None = None
    target_symbol: str | None = None
    source_body: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    is_new_file: bool = False
    evidence_source: str | None = None
    base_revision_sha: str | None = None
    writable_paths: tuple[str, ...] = ()
    creatable_paths: tuple[str, ...] = ()
    target_pinned: bool = False

    @property
    def localization_stage(self) -> LocalizationStage:
        if self.is_new_file:
            return LocalizationStage.READY if _canonical_mutation_path(self.target_path) else LocalizationStage.NEED_FILE
        if not _canonical_mutation_path(self.target_path):
            return LocalizationStage.NEED_FILE
        if self.source_body and _is_code_bearing_text(self.source_body):
            return LocalizationStage.READY
        if not self.target_symbol and self.start_line is None:
            return LocalizationStage.NEED_SYMBOL
        return LocalizationStage.NEED_BODY

    @property
    def is_mutation_ready(self) -> bool:
        return self.localization_stage == LocalizationStage.READY

    def merge(self, other: TargetMutationContext) -> TargetMutationContext:
        return _merge_target_context(self, other)



def _paths_conflict(left: str, right: str) -> bool:
    return bool(left and right and left != right)


def _conflict_resolution(
    current: TargetMutationContext,
    other: TargetMutationContext,
    left: str,
    right: str,
) -> TargetMutationContext | None:
    if not _paths_conflict(left, right):
        return None
    if current.target_pinned:
        return current
    return other


def _existing_target_context(
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> TargetMutationContext | None:
    if not other.is_new_file and str(other.evidence_source or "").strip() in _EXISTING_TARGET_EVIDENCE_SOURCES:
        return other
    if not current.is_new_file and str(current.evidence_source or "").strip() in _EXISTING_TARGET_EVIDENCE_SOURCES:
        return current
    return None


def _preferred_context_value(
    existing: TargetMutationContext | None,
    existing_value: Any,
    other_value: Any,
    current_value: Any,
) -> Any:
    if existing is not None and existing_value is not None:
        return existing_value
    if other_value is not None:
        return other_value
    return current_value


def _without_target_path(paths: Sequence[str], target: str) -> tuple[str, ...]:
    return tuple(item for item in paths if _canonical_mutation_path(item) != target)


def _merged_is_new_file(
    existing: TargetMutationContext | None,
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> bool:
    if existing is not None:
        return False
    return any((current.is_new_file, other.is_new_file))


def _merged_evidence_source(
    existing: TargetMutationContext | None,
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> str | None:
    if existing is not None:
        return existing.evidence_source
    return other.evidence_source or current.evidence_source


def _merge_target_context(
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> TargetMutationContext:
    if other is None:
        return current
    left = _canonical_mutation_path(current.target_path)
    right = _canonical_mutation_path(other.target_path)
    conflict = _conflict_resolution(current, other, left, right)
    if conflict is not None:
        return conflict
    writable = tuple(dict.fromkeys((*current.writable_paths, *other.writable_paths)))
    creatable = tuple(dict.fromkeys((*current.creatable_paths, *other.creatable_paths)))
    existing = _existing_target_context(current, other)
    target_path = other.target_path or current.target_path
    target = _canonical_mutation_path(target_path)
    if existing is not None and target:
        creatable = _without_target_path(creatable, target)
    return TargetMutationContext(
        target_path=target_path,
        target_symbol=other.target_symbol or current.target_symbol,
        source_body=_preferred_context_value(
            existing,
            existing.source_body if existing is not None else None,
            other.source_body,
            current.source_body,
        ),
        start_line=_preferred_context_value(
            existing,
            existing.start_line if existing is not None else None,
            other.start_line,
            current.start_line,
        ),
        end_line=_preferred_context_value(
            existing,
            existing.end_line if existing is not None else None,
            other.end_line,
            current.end_line,
        ),
        is_new_file=_merged_is_new_file(existing, current, other),
        evidence_source=_merged_evidence_source(existing, current, other),
        base_revision_sha=_preferred_context_value(
            existing,
            existing.base_revision_sha if existing is not None else None,
            other.base_revision_sha,
            current.base_revision_sha,
        ),
        writable_paths=writable,
        creatable_paths=creatable,
        target_pinned=any((current.target_pinned, other.target_pinned)),
    )


def _first_mapping_value(
    mappings: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return None


def _search_hit_text(hit: Mapping[str, Any]) -> str | None:
    raw = _first_mapping_value(
        (hit,),
        ("text", "snippet", "code", "content", "source"),
    )
    if isinstance(raw, (list, tuple)):
        return "\n".join(str(item) for item in raw)
    return raw if isinstance(raw, str) else None


def _search_hit_context(hit: Mapping[str, Any]) -> TargetMutationContext | None:
    raw_meta = hit.get("metadata")
    meta = raw_meta if isinstance(raw_meta, Mapping) else {}
    path = _canonical_mutation_path(
        _first_mapping_value(
            (hit, meta),
            ("source_path", "path", "file", "uri"),
        )
    )
    if not _is_workspace_file_path(path):
        return None
    text = _search_hit_text(hit)
    symbol = str(
        _first_mapping_value((hit, meta), ("symbol", "function", "name")) or ""
    ).strip()
    start_line = hit.get("start_line")
    end_line = hit.get("end_line")
    return TargetMutationContext(
        target_path=path,
        target_symbol=symbol or None,
        source_body=text if _is_code_bearing_text(text) else None,
        start_line=start_line if isinstance(start_line, int) else None,
        end_line=end_line if isinstance(end_line, int) else None,
        evidence_source="search_code_rag",
    )


def _extract_search_context(payload: Mapping[str, Any]) -> TargetMutationContext | None:
    hits = payload.get("hits") or payload.get("results") or payload.get("sources")
    for hit in _sequence(hits):
        if not isinstance(hit, Mapping):
            continue
        context = _search_hit_context(hit)
        if context is not None:
            return context
    return None


def _extract_mutation_context_from_payload(payload: Any) -> TargetMutationContext | None:
    if not isinstance(payload, Mapping):
        for item in _sequence(payload):
            context = _extract_mutation_context_from_payload(item)
            if context is not None:
                return context
        return None

    authority = _task_authority_context(payload)
    if authority is not None:
        initial = payload.get("initial_exact_source_context")
        if isinstance(initial, Mapping):
            exact_candidates: list[TargetMutationContext] = []
            direct_exact = _extract_mutation_context_from_payload(initial)
            if direct_exact is not None:
                exact_candidates.append(direct_exact)
            for record in _sequence(initial.get("records")):
                exact = _extract_mutation_context_from_payload(record)
                if exact is not None:
                    exact_candidates.append(exact)
            expected_path = _canonical_mutation_path(authority.target_path)
            exact = next(
                (
                    candidate
                    for candidate in exact_candidates
                    if _canonical_mutation_path(candidate.target_path) == expected_path
                    and candidate.source_body
                ),
                None,
            )
            if exact is not None:
                return replace(
                    authority,
                    source_body=exact.source_body,
                    start_line=exact.start_line,
                    end_line=exact.end_line,
                    is_new_file=False,
                    evidence_source="host_exact_source",
                    creatable_paths=_without_target_path(
                        authority.creatable_paths,
                        expected_path,
                    ),
                )
        return authority

    for key in ("structured_content", "result", "data", "body", "_mmm_observation", "raw_result", "structured", "observation"):
        wrapped = payload.get(key)
        if isinstance(wrapped, (Mapping, list, tuple)) and wrapped is not payload:
            context = _extract_mutation_context_from_payload(wrapped)
            if context is not None:
                return context

    context = _extract_search_context(payload)
    if context is not None:
        return context

    symbols = payload.get("symbols")
    for symbol in _sequence(symbols):
        if not isinstance(symbol, Mapping):
            continue
        location = symbol.get("location")
        if not isinstance(location, Mapping):
            continue
        uri = str(location.get("uri") or "")
        path = _canonical_mutation_path(uri.replace("file:///", "").replace("file://", ""))
        if not path:
            continue
        name = str(symbol.get("name") or "").strip()
        return TargetMutationContext(
            target_path=path,
            target_symbol=name or None,
            evidence_source="java_workspace_symbols",
        )

    files = payload.get("files")
    if isinstance(files, Mapping):
        for raw_path, raw_content in files.items():
            path = _canonical_mutation_path(raw_path)
            if _is_workspace_file_path(path):
                return TargetMutationContext(
                    target_path=path,
                    source_body=str(raw_content) if _is_code_bearing_text(str(raw_content)) else None,
                    evidence_source="files_map",
                )

    target = _canonical_mutation_path(payload.get("target_file") or payload.get("path"))
    if target and _is_workspace_file_path(target):
        source = payload.get("source") or payload.get("content") or payload.get("code")
        return TargetMutationContext(
            target_path=target,
            source_body=source if _is_code_bearing_text(source) else None,
            evidence_source="target_path_field",
        )
    return None


def _mutation_context_dict(ctx: TargetMutationContext | None) -> dict[str, Any] | None:
    if ctx is None:
        return None
    return {
        "target_path": ctx.target_path,
        "target_symbol": ctx.target_symbol,
        "source_body_len": len(ctx.source_body) if ctx.source_body else 0,
        "start_line": ctx.start_line,
        "end_line": ctx.end_line,
        "is_new_file": ctx.is_new_file,
        "localization_stage": ctx.localization_stage.value,
        "evidence_source": ctx.evidence_source,
        "writable_paths": list(ctx.writable_paths),
        "creatable_paths": list(ctx.creatable_paths),
        "target_pinned": ctx.target_pinned,
    }


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
    if context is None:
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
    if supplied in set(context.creatable_paths):
        return True
    return supplied == pinned and context.is_new_file


_JAVA_PACKAGE_DECLARATION_RE = re.compile(
    r"(?m)^\s*package\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*;"
)
_JAVA_PUBLIC_TOP_LEVEL_TYPE_RE = re.compile(
    r"\bpublic\s+(?:(?:abstract|final|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)\b"
)


def _java_whole_file_identity_error(
    path: str,
    current_source: str | None,
    new_source: Any,
) -> str | None:
    """Reject whole-file Java repairs that change the host-selected source identity."""

    if not path.casefold().endswith(".java") or not isinstance(new_source, str):
        return None
    expected_type = Path(path).stem
    current = current_source if isinstance(current_source, str) else ""

    current_package_match = _JAVA_PACKAGE_DECLARATION_RE.search(current)
    new_package_match = _JAVA_PACKAGE_DECLARATION_RE.search(new_source)
    if current_package_match is not None:
        current_package = current_package_match.group(1)
        new_package = new_package_match.group(1) if new_package_match is not None else ""
        if new_package != current_package:
            return (
                "REPAIR_SEMANTIC_IDENTITY_VIOLATION: whole-file Java repair changed "
                f"package identity for {path!r}: expected {current_package!r}, got "
                f"{new_package or '<missing>'!r}"
            )

    public_types = tuple(_JAVA_PUBLIC_TOP_LEVEL_TYPE_RE.findall(new_source))
    if public_types and expected_type not in public_types:
        return (
            "REPAIR_SEMANTIC_IDENTITY_VIOLATION: whole-file Java repair changed "
            f"primary type identity for {path!r}: expected public type "
            f"{expected_type!r}, got {public_types!r}"
        )

    current_public_types = tuple(_JAVA_PUBLIC_TOP_LEVEL_TYPE_RE.findall(current))
    if expected_type in current_public_types and expected_type not in public_types:
        return (
            "REPAIR_SEMANTIC_IDENTITY_VIOLATION: whole-file Java repair removed "
            f"the existing public type {expected_type!r} from {path!r}"
        )
    return None


def _mutation_target_error(
    tool_name: str,
    arguments: Mapping[str, Any],
    context: TargetMutationContext | None,
) -> str | None:
    if tool_name != "apply_source_edit":
        return None
    authority = CURRENT_MUTATION_AUTHORITY.get()
    if authority is not None:
        error = authority.mutation_error(
            _source_edit_path(arguments),
            operation=arguments.get("operation"),
        )
        if error is not None:
            return error
        if authority.mode is MutationAuthorityMode.BOUNDED_ROOTS and not (
            context is not None and context.evidence_source == "verifier_workspace_source"
        ):
            return None
    if context is None:
        return "MUTATION_TARGET_UNBOUND: no host-pinned mutation target is READY"
    if not context.is_mutation_ready:
        return "MUTATION_TARGET_UNBOUND: no host-pinned mutation target is READY"
    supplied = _source_edit_path(arguments)
    pinned = _canonical_mutation_path(context.target_path)
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
    operation = str(arguments.get("operation") or "").strip().casefold()
    if (
        operation == "replace_exact"
        and supplied == pinned
        and not context.is_new_file
        and "old" not in arguments
    ):
        identity_error = _java_whole_file_identity_error(
            pinned,
            context.source_body,
            arguments.get("new"),
        )
        if identity_error is not None:
            return identity_error
    if operation not in _SOURCE_CREATE_OPERATIONS:
        return None
    if _creation_authorized(supplied, pinned, context):
        return None
    if (
        supplied == pinned
        and not context.is_new_file
        and operation in _SOURCE_ATOMIC_REWRITE_OPERATIONS
    ):
        # The scalar source-edit core lowers same-path create_file/create on an
        # existing exact target into an expected-SHA replace. It is a rewrite,
        # not authority to create another path.
        return None
    return (
        "MUTATION_TARGET_CREATION_CONFLICT: create operation is not authorized "
        f"for existing target {supplied!r}"
    )


def _verification_outcome(tool_name: str, payload: Mapping[str, Any]) -> str:
    if not bool(payload.get("ok")):
        code = str(payload.get("failure_code") or "").strip().upper()
        if code in {"VERIFIER_ARGUMENT_INVALID", "VERIFIER_TARGET_INVALID"}:
            return code
        return "UNAVAILABLE"
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return "PASS"
    status = str(result.get("status") or result.get("state") or result.get("outcome") or "").strip().upper()
    if status in _VERIFIER_UNAVAILABLE_STATUSES:
        return "UNAVAILABLE"
    if result.get("available") is False:
        return "UNAVAILABLE"
    if tool_name in {"java_diagnostics", "jdt_diagnostics"}:
        from .validation_diagnostic_contract import diagnostic_errors
        errors = diagnostic_errors(result)
        unavailable_codes = {
            "JDT_DIAGNOSTICS_UNAVAILABLE",
            "JDT_WORKSPACE_NOT_READY",
        }
        if any(str(item.get("code") or "").strip().upper() in unavailable_codes for item in errors):
            return "UNAVAILABLE"
        if errors:
            return "FAIL"
        return "FAIL" if status in _VERIFIER_FAIL_STATUSES else "PASS"
    if status in _VERIFIER_FAIL_STATUSES:
        return "FAIL"
    if status in _VERIFIER_PASS_STATUSES:
        return "PASS"
    for key in ("success", "ok"):
        if isinstance(result.get(key), bool):
            return "PASS" if result[key] else "FAIL"
    if "exit_code" in result:
        try:
            return "PASS" if int(result["exit_code"]) == 0 else "FAIL"
        except (TypeError, ValueError, OverflowError):
            return "UNAVAILABLE"
    return "PASS"


def _fixed_point_tool_results(
    executed: Sequence[tuple[Any, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Use semantic outcomes, not volatile transport text, as loop identity."""
    stable: list[dict[str, Any]] = []
    for call, payload in executed:
        if call.name in _VERIFY_TOOLS:
            stable.append({
                "name": call.name,
                "verification_outcome": _verification_outcome(call.name, payload),
            })
        else:
            stable.append({
                "name": call.name,
                "ok": bool(payload.get("ok")),
                "failure_code": payload.get("failure_code"),
            })
    return stable


def _fixed_point_tool_calls(calls: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip source payload bytes while preserving semantic action identity."""

    stable: list[dict[str, Any]] = []
    for call in calls:
        arguments = call.arguments if isinstance(call.arguments, Mapping) else {}
        item: dict[str, Any] = {"name": call.name}
        for key in (
            "operation",
            "path",
            "target_path",
            "file_path",
            "query",
            "capability",
        ):
            value = arguments.get(key)
            if value not in (None, "", [], {}, ()):
                item[key] = _stable_value(value)
        stable.append(item)
    return stable


def _runtime_failure_code(tool_name: str, error: str) -> str:
    lowered = str(error or "").casefold()
    if tool_name in _MUTATION_ACT_TOOLS:
        if "exact source-edit precondition failed" in lowered:
            return "MUTATION_STALE_PRECONDITION"
        if "target already exists" in lowered or "creation_conflict" in lowered:
            return "MUTATION_TARGET_CREATION_CONFLICT"
        if "source edit requires an existing regular file" in lowered:
            return "MUTATION_TARGET_UNBOUND"
    if tool_name in _VERIFY_TOOLS:
        if any(
            marker in lowered
            for marker in ("no such file", "not found", "does not exist", "outside", "unsafe path")
        ):
            return "VERIFIER_TARGET_INVALID"
        if any(marker in lowered for marker in ("argument", "schema", "invalid")):
            return "VERIFIER_ARGUMENT_INVALID"
    return "TOOL_RUNTIME_UNAVAILABLE"


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
                    "The preceding repair output exceeded the bounded allowance and is discarded. "
                    "Call apply_source_edit exactly once with no prose. Emit only the complete corrected "
                    "existing source file in the new argument. Do not emit operation, path, old text, "
                    "anchors, partial edits, or any additional tool call; the host binds those details."
                )
            break
        return (
            "The preceding assistant action exceeded the bounded output allowance and is discarded. "
            "Do not continue, reproduce, or complete that oversized payload. The host will preserve the "
            "same mutation target and workspace state. Use exactly one visible source-mutation tool: "
            "call apply_source_edit exactly once with no prose and make one small semantic edit. "
            "For a fresh host-pinned Java target, use operation=create_file and provide one complete Java "
            "file that is minimal and compilable at the already-authorized path. For an existing target, "
            "use one bounded replace/insert operation, or create_file only as an atomic whole-file rewrite "
            "of that exact same path. Do not invent Java mutation tools that are not visible in this turn."
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
    active_authority = CURRENT_MUTATION_AUTHORITY.get()
    bounded_roots = (
        tuple(active_authority.roots)
        if (
            active_authority is not None
            and active_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
        )
        else ()
    )
    if context is not None and context.evidence_source in {
        "verifier_workspace_source", "workspace_existing_target"
    } and getattr(state, "validation_status", "") == "FAIL":
        directive = (
            f"HOST FORCED ACT: repair the verified defect in {target!r} using the fresh "
            "workspace source and diagnostics supplied by the host. Emit the complete corrected "
            "contents in the single visible new argument only. Operation, path, old text, and SHA "
            "are host-owned. Preserve approved behavior and do not restart generation or retrieve."
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


def _model_tool_rejection_feedback(
    calls: Sequence[Any],
) -> tuple[str, list[Mapping[str, Any]]] | None:
    """Convert adapter admission rejections into retryable model feedback."""
    rejections: list[Mapping[str, Any]] = []
    for call in calls:
        if str(getattr(call, "name", "") or "").strip() != _MODEL_REJECTION_TOOL_NAME:
            continue
        arguments = getattr(call, "arguments", None)
        if isinstance(arguments, Mapping):
            rejections.append(dict(arguments))
        else:
            rejections.append({
                "failure_code": "MODEL_TOOL_CALL_REJECTED",
                "error": "invalid rejection payload",
            })
    if not rejections:
        return None

    details: list[str] = []
    for payload in rejections:
        code = str(payload.get("failure_code") or "MODEL_TOOL_CALL_REJECTED").strip()
        name = str(
            payload.get("original_tool")
            or payload.get("rejected_name")
            or ""
        ).strip()
        error = str(payload.get("error") or "").strip()
        line = code + (f" for {name!r}" if name else "")
        if error:
            line += f": {error}"
        details.append(line)
    return (
        "The previous model tool call was rejected during host admission and was not executed. "
        "Correct the tool name/arguments to match the currently exposed schema and try the required "
        "phase action again. Rejections: " + " | ".join(details),
        rejections,
    )


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


_JAVA_API_EVIDENCE_RE = re.compile(
    r"(?:\b(?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)\.[A-Za-z0-9_.$]+"
    r"|\b(?:package|import)\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+"
    r"|\b(?:class|interface|record|enum)\s+[A-Za-z_$][\w$]*)"
)
_ATOMIC_OUTPUT_RECOVERY_MARKER = "MMM_ATOMIC_OUTPUT_RECOVERY_V1"
_ATOMIC_SOURCE_EDIT_OUTPUT_TOKENS = 4096


def _fresh_java_context(context: TargetMutationContext | None) -> bool:
    if context is None or not context.is_new_file:
        return False
    return _canonical_mutation_path(context.target_path).casefold().endswith(".java")


def _mapping_schema(value: Any, schema: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if str(value.get("schema_version") or "").strip() == schema:
        return True
    for key in ("structured_content", "result", "data"):
        child = value.get(key)
        if isinstance(child, Mapping) and _mapping_schema(child, schema):
            return True
    return False


def _append_java_text_field(texts: list[str], raw: Any) -> None:
    if isinstance(raw, str):
        if raw.strip():
            texts.append(raw)
        return
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        texts.extend(str(item) for item in raw if isinstance(item, str) and item.strip())


def _collect_java_mapping_texts(value: Mapping[str, Any], texts: list[str]) -> None:
    for key in ("parsed_text", "text", "content", "snippet", "code", "source", "source_text", "body"):
        _append_java_text_field(texts, value.get(key))
    for key in ("hits", "results", "records", "documents", "chunks", "resources", "symbols", "evidence"):
        raw = value.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for item in raw:
                texts.extend(_java_evidence_texts(item))
    for key in ("structured_content", "result", "data"):
        child = value.get(key)
        if child is not None:
            texts.extend(_java_evidence_texts(child))


def _java_evidence_texts(value: Any) -> tuple[str, ...]:
    texts: list[str] = []
    if isinstance(value, Mapping):
        _collect_java_mapping_texts(value, texts)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            texts.extend(_java_evidence_texts(item))
    return tuple(texts)


def _has_symbol_records(item: Any) -> bool:
    if isinstance(item, Mapping):
        symbols = item.get("symbols")
        if (
            isinstance(symbols, Sequence)
            and not isinstance(symbols, (str, bytes, bytearray))
            and any(isinstance(symbol, Mapping) and bool(symbol) for symbol in symbols)
        ):
            return True
        return any(_has_symbol_records(child) for child in item.values())
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        return any(_has_symbol_records(child) for child in item)
    return False


def _has_mapping_records(item: Any) -> bool:
    if isinstance(item, Mapping):
        mappings = item.get("mappings")
        if isinstance(mappings, Mapping) and bool(mappings):
            return True
        if (
            isinstance(mappings, Sequence)
            and not isinstance(mappings, (str, bytes, bytearray))
            and any(isinstance(entry, Mapping) and bool(entry) for entry in mappings)
        ):
            return True
        return any(_has_mapping_records(child) for child in item.values())
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        return any(_has_mapping_records(child) for child in item)
    return False


def _contains_java_api_evidence(value: Any) -> bool:
    return any(_JAVA_API_EVIDENCE_RE.search(text) for text in _java_evidence_texts(value))


def _authoritative_java_evidence(value: Any) -> bool:
    """Return whether evidence is strong enough to authorize a fresh Java mutation."""
    if not isinstance(value, Mapping):
        return False
    if not value:
        return False
    if _mapping_schema(value, "mmm/rag-result-v2"):
        return False
    if _mapping_schema(value, "mmm/java-symbols-v1"):
        return _has_symbol_records(value)
    if _mapping_schema(value, "mmm/code-rag-result-v1"):
        return _contains_java_api_evidence(value)
    if _contains_java_api_evidence(value):
        return True
    return _has_mapping_records(value)


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


def _requires_rag_evidence(
    *,
    role: str,
    host_grounded: bool,
    router_requires_fresh_evidence: bool,
    implementation_requires_mutation: bool,
    initial_execution_authority: bool,
) -> bool:
    """Separate write-location authority from implementation-evidence authority.

    An exact host-reserved target proves where the coder may write. It does not prove
    which Minecraft/Fabric API is valid for the approved version. Explicit fresh-evidence
    policy therefore remains authoritative even for an executable target, while host
    target authority still suppresses retrieval that would exist only to localize the file.
    """

    if role not in {"coder", "coder_safe"} or host_grounded:
        return False
    if router_requires_fresh_evidence:
        return True
    return bool(implementation_requires_mutation and not initial_execution_authority)


def _target_evidence_ready(
    state: HostRunState,
    *,
    require_rag: bool,
    fresh_java_target: bool,
    compile_backed_java: bool = False,
) -> bool:
    """Decide whether implementation may proceed.

    target_compile is the canonical verifier, not permission to guess an exact
    Minecraft/Fabric API. When the active coding policy requires fresh evidence, the
    small coder must ground the implementation before mutating even though compile is
    guaranteed downstream.
    """

    del compile_backed_java
    if not require_rag:
        return True
    return (
        state.has_authoritative_java_evidence
        if fresh_java_target
        else state.has_fresh_evidence
    )

def _record_evidence_locked(state: Any, value: Any, fingerprint: str) -> bool:
    if fingerprint in state.evidence_fingerprints:
        return False
    state.evidence_fingerprints.add(fingerprint)
    if _fresh_java_context(state.mutation_context) and _authoritative_java_evidence(value):
        state.authoritative_java_evidence_fingerprints.add(fingerprint)
    context = _extract_mutation_context_from_payload(value)
    if context is not None and state.mutation_context is not None:
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
        # replace_exact without old is the model-facing atomic whole-file rewrite.
        # Keep host state identical to the file content that the scalar protocol writes.
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
        and "old" not in arguments
    ):
        state.repair_baseline_error_count = len(state.latest_verifier_errors)
        state.repair_baseline_errors = tuple(state.latest_verifier_errors)
        state.repair_baseline_fingerprint = state.latest_verifier_fingerprint
        state.repair_previous_source = context.source_body
        state.repair_previous_path = path
        state.last_verifier_quality = None
    else:
        state.repair_baseline_error_count = None
        state.repair_baseline_errors = ()
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
    state.latest_verifier_tool = None
    state.latest_verifier_errors = ()
    state.repair_target_diagnostics = ()
    state.latest_verifier_fingerprint = None
    state.repair_guidance_fingerprint = None
    if path and operation in _SOURCE_CREATE_OPERATIONS:
        state.created_paths.add(path)
    _update_mutation_context_after_edit(state, path, operation, arguments)
    return True


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
    diagnostic_snapshot = (
        json.loads(_bounded_verifier_recovery_observation(
            state, errors=state.repair_target_diagnostics,
        ))
        if context and context.evidence_source == "verifier_workspace_source"
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
            "other_file_diagnostic_count": (
                len(state.latest_verifier_errors) - len(state.repair_target_diagnostics)
            ),
        } if diagnostic_snapshot else {}),
        "target_path": context.target_path if context else None,
        "target_is_new_file": context.is_new_file if context else None,
        "writable_paths": list(context.writable_paths) if context else [],
        "current_source": source,
    }


@dataclass
class HostRunState:
    phase: LoopPhase = LoopPhase.OBSERVE
    step_index: int = 0
    no_progress_streak: int = 0
    seen_no_progress_digests: set[str] = field(default_factory=set)
    semantic_fixed_point: bool = False
    attempted_queries: set[str] = field(default_factory=set)
    attempted_sources: set[str] = field(default_factory=set)
    evidence_fingerprints: set[str] = field(default_factory=set)
    authoritative_java_evidence_fingerprints: set[str] = field(default_factory=set)
    mutation_context: TargetMutationContext | None = None
    applied_mutations: list[str] = field(default_factory=list)
    mutation_fingerprints: set[str] = field(default_factory=set)
    unchanged_mutation_fingerprints: set[str] = field(default_factory=set)
    unapplied_mutation_fixed_point: bool = False
    created_paths: set[str] = field(default_factory=set)
    workspace_changed: bool = False
    validation_status: str = "PENDING"
    latest_verifier_tool: str | None = None
    latest_verifier_errors: tuple[dict[str, Any], ...] = ()
    repair_target_diagnostics: tuple[dict[str, Any], ...] = ()
    latest_verifier_fingerprint: str | None = None
    repair_guidance_fingerprint: str | None = None
    repair_baseline_error_count: int | None = None
    repair_baseline_errors: tuple[dict[str, Any], ...] = ()
    repair_baseline_fingerprint: str | None = None
    repair_previous_source: str | None = None
    repair_previous_path: str | None = None
    last_verifier_quality: str | None = None
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
                self.repair_baseline_fingerprint = None
                self.repair_previous_source = None
                self.repair_previous_path = None
            return progress

    def take_verifier_repair_guidance(self) -> str | None:
        with self._lock:
            payload = _repair_guidance_payload(self)
        if payload is None:
            return None
        return (
            "MMM_CORE_VERIFIER_REPAIR_V5\n"
            "The verifier failure is the active repair obligation. Do not restart generation, "
            "do not search unrelated ecosystem candidates, and never write a different path. "
            "The payload includes the exact host-tracked current source. "
            "Any earlier host_reserved/fresh metadata is pre-materialization history only. "
            "This repair turn exposes only the complete corrected source body in new. "
            "Operation, path, old text, and optimistic-concurrency SHA are host-owned and must not "
            "be emitted by the model. The host binds operation=replace_exact and new to an atomic "
            "live-SHA whole-file rewrite. Preserve the file package and primary Java type identity. "
            "Use the diagnostics below against the host-pinned target and make one materially "
            "different source edit that reduces severity-1 diagnostics. An equal or worse verifier "
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
        canonical = json.dumps(
            _stable_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            repeated = digest in self.seen_no_progress_digests
            self.seen_no_progress_digests.add(digest)
            self.semantic_fixed_point = repeated
            self.no_progress_streak += 1
            return repeated

    def clear_no_progress_result(self) -> None:
        with self._lock:
            self.seen_no_progress_digests.clear()
            self.semantic_fixed_point = self.unapplied_mutation_fixed_point
            self.no_progress_streak = 0

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


class RetrievalNoProgressError(ModelConfigurationError):
    pass


class RetrievalProgress:
    def __init__(self, state: HostRunState | None = None, *, no_progress_limit: int | None = None) -> None:
        self._state = state or HostRunState()
        self.attempted_queries = self._state.attempted_queries
        self.attempted_sources = self._state.attempted_sources
        self.evidence_fingerprints = self._state.evidence_fingerprints
        self.no_progress_observations = 0
        self._no_progress_limit = no_progress_limit

    def begin(self, tool_name: str, arguments: Mapping[str, Any]) -> RetrievalDecision:
        return (
            RetrievalDecision.EXECUTE
            if self._state.record_query(tool_name, arguments)
            else RetrievalDecision.DUPLICATE_QUERY
        )

    def observe(self, *args: Any, usable: bool = True, **kwargs: Any) -> RetrievalObservation:
        value = args[2] if len(args) >= 3 else (args[0] if args else kwargs.get("value"))
        if not usable:
            self.no_progress_observations += 1
            if self._no_progress_limit is not None and self.no_progress_observations >= self._no_progress_limit:
                raise RetrievalNoProgressError("no novel usable evidence")
            return RetrievalObservation.WEAK
        if self._state.record_evidence(value, usable=True):
            self.no_progress_observations = 0
            return RetrievalObservation.FRESH
        return RetrievalObservation.DUPLICATE_EVIDENCE

    @property
    def has_fresh_evidence(self) -> bool:
        return self._state.has_fresh_evidence

    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
    ) -> str | None:
        return self._state.next_untried_internal_tool(exposed_tools, preferred=preferred)


def _source_edit_schema_for_context(
    schema: Mapping[str, Any],
    context: TargetMutationContext | None,
) -> Mapping[str, Any]:
    """Project source-edit choices onto the exact live target mutation state.

    A fresh Java target exposes only create_file/path/content; aliases and repair-only
    operations remain host-side compatibility rather than model-facing choices.
    """
    if _tool_name(schema) != "apply_source_edit" or context is None:
        return schema
    cloned = deepcopy(schema)
    if not isinstance(cloned, dict):
        return schema
    function = cloned.get("function")
    if not isinstance(function, dict):
        return cloned
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
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
                    if (
                        str(value).strip().casefold() not in _SOURCE_CREATE_OPERATIONS
                        or str(value).strip().casefold() in _SOURCE_ATOMIC_REWRITE_OPERATIONS
                    )
                ]

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
    elif not context.is_new_file:
        suffix = (
            "Existing host-pinned target: create_file/create on this exact same path means "
            "an atomic whole-file rewrite and is host-lowered to a SHA-bound replace; "
            "create_java_type or any different path remains forbidden. Exact bounded edits "
            "remain available when a smaller repair is sufficient."
        )
    else:
        suffix = ""
    function["description"] = f"{description} {suffix}".strip()
    return cloned


def _unattempted_tools(
    by_name: Mapping[str, Mapping[str, Any]],
    attempted: set[str],
    preferred: Sequence[str],
) -> list[str]:
    return [name for name in preferred if name in by_name and name not in attempted]


def _fresh_observe_names(
    by_name: Mapping[str, Mapping[str, Any]],
    attempted: set[str],
    mutation_context: TargetMutationContext,
    *,
    semantic_retrieval_choice: bool,
) -> list[str]:
    del mutation_context
    preferred = (
        "search_code_rag",
        "java_workspace_symbols",
        "search_project_rag",
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
        "inspect_modrinth_project",
    )
    names = _unattempted_tools(by_name, attempted, preferred)
    return names if semantic_retrieval_choice else names[:1]


def _localized_observe_names(
    by_name: Mapping[str, Mapping[str, Any]],
    attempted: set[str],
    mutation_context: TargetMutationContext | None,
    *,
    semantic_retrieval_choice: bool,
) -> list[str]:
    if mutation_context and mutation_context.is_new_file and mutation_context.is_mutation_ready:
        return _fresh_observe_names(
            by_name, attempted, mutation_context,
            semantic_retrieval_choice=semantic_retrieval_choice,
        )
    stage = mutation_context.localization_stage if mutation_context else LocalizationStage.NEED_FILE
    if stage == LocalizationStage.NEED_FILE:
        preferred = ("search_code_rag", "search_project_rag")
    elif stage == LocalizationStage.NEED_SYMBOL:
        preferred = ("java_workspace_symbols", "search_code_rag", "search_project_rag")
    elif stage == LocalizationStage.NEED_BODY:
        preferred = ("search_code_rag", "java_workspace_symbols", "search_project_rag")
    else:
        preferred = ("search_project_rag", "search_code_rag")
    names = _unattempted_tools(by_name, attempted, preferred)
    return names if semantic_retrieval_choice else names[:1]


def _filter_tools_for_phase(
    exposed_tools: Sequence[Mapping[str, Any]],
    phase: LoopPhase,
    role: str,
    *,
    mutation_context: TargetMutationContext | None = None,
    attempted_sources: Sequence[str] | set[str] | frozenset[str] = frozenset(),
    localization_active: bool | None = None,
    semantic_retrieval_choice: bool = False,
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
        pinned_source = bool(
            mutation_context
            and mutation_context.is_mutation_ready
            and isinstance(mutation_context.source_body, str)
        )
        recovery_frontier = (
            (
                "search_code_rag",
                "java_workspace_symbols",
                "search_project_rag",
                "external_mcp_capabilities",
                "external_mcp_schema",
                "external_mcp_call",
            )
            if pinned_source
            else (
                "search_code_rag",
                "java_workspace_symbols",
                "search_project_rag",
                "external_mcp_capabilities",
                "external_mcp_schema",
                "external_mcp_call",
                "inspect_modrinth_project",
                "read_reuse_source",
            )
        )
        names = [name for name in recovery_frontier if name in by_name and name not in attempted]
    else:
        active = mutation_context is not None if localization_active is None else localization_active
        if not active:
            names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
            if mutation_context is None and "search_code_rag" in names and "java_workspace_symbols" in names:
                names.remove("java_workspace_symbols")
        else:
            names = _localized_observe_names(
                by_name, attempted, mutation_context,
                semantic_retrieval_choice=semantic_retrieval_choice,
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
    state: HostRunState, *, errors: Sequence[Mapping[str, Any]] | None = None,
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
    budget = _PHASE_HANDOFF_VERIFIER_DIAGNOSTIC_BYTES
    for raw in raw_errors:
        if not isinstance(raw, Mapping):
            continue
        compact: dict[str, Any] = {}
        for key in ("path", "file", "uri", "line", "severity", "code", "source", "message", "range"):
            value = raw.get(key)
            if value in (None, "", [], {}):
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
        trial = {
            "verifier": str(state.latest_verifier_tool or ""),
            "status": "FAIL",
            "diagnostics": [*diagnostics, compact],
        }
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
        last_prompt_phase is LoopPhase.VERIFY
        and next_phase is LoopPhase.RECOVER
        and state.validation_status == "FAIL"
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
    if not path or not isinstance(source, str):
        return False
    result = runtime.call(
        stage,
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": path,
            "new": source,
        },
    )
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


def _fixed_point_error(state: HostRunState) -> ModelConfigurationError:
    trajectory = format_trajectory_summary(state.trajectory)
    if state.validation_status == "FAIL":
        return ModelConfigurationError(
            "VERIFICATION_REPAIR_FIXED_POINT: the same repair state repeated while "
            "trustworthy verifier diagnostics remain unresolved.\n" + trajectory
        )
    return ModelConfigurationError(
        "AGENT_SEMANTIC_FIXED_POINT: the same action/result state repeated without "
        "workspace, localization, evidence, or verification progress.\n" + trajectory
    )


def _consume_rejected_evidence_fixed_point(
    state: HostRunState,
    rejection_payloads: Sequence[Mapping[str, Any]],
    phase_names: Collection[str],
    *,
    forced_evidence_tool: str | None = None,
    forced_evidence_arguments: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Consume the reviewed evidence route that cannot produce an admissible call.

    Adapter rejection payloads name the model-emitted tool. When the host has forced
    exactly one reviewed evidence route, a non-visible model tool is evidence that the
    forced route failed to yield an admissible call, not that the invented tool became
    part of the reviewed frontier. After the same rejection state repeats, consume the
    forced route so the frontier can advance without granting authority to the invented
    tool name.
    """

    if state.phase not in {LoopPhase.OBSERVE, LoopPhase.RECOVER}:
        return ()
    allowed = {
        str(name).strip()
        for name in phase_names
        if str(name).strip()
    }
    routes = {
        name
        for payload in rejection_payloads
        if (name := str(payload.get("original_tool") or "").strip())
        and name in allowed
    }
    forced = str(forced_evidence_tool or "").strip()
    if not routes and forced and forced in allowed and len(allowed) == 1:
        routes.add(forced)
    consumed = tuple(sorted(routes))
    if not consumed:
        return ()
    forced_arguments = dict(forced_evidence_arguments or {})
    for route in consumed:
        state.record_source_attempt(
            route,
            forced_arguments if route == forced else {},
        )
    state.clear_no_progress_result()
    return consumed


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
    fresh_java_target = bool(
        java_target
        and state.mutation_context
        and state.mutation_context.is_new_file
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
    from .small_model_task_capsule_contract import current_task_required_gates
    compile_backed_java = bool(
        java_target
        and state.mutation_context
        and state.mutation_context.target_pinned
        and state.mutation_context.is_mutation_ready
        and "target_compile" in current_task_required_gates()
    )
    router_requires_fresh_evidence = bool(router._agent_require_fresh_evidence)
    require_rag = bool(
        authored_workspace_refresh
        or _requires_rag_evidence(
            role=role,
            host_grounded=host_grounded,
            router_requires_fresh_evidence=router_requires_fresh_evidence,
            implementation_requires_mutation=implementation_requires_mutation,
            initial_execution_authority=initial_execution_authority,
        )
    )
    required_evidence_choice = bool(require_rag)

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
                # This is fresh local evidence, not permission inferred from a
                # search hit. It satisfies recovery without consuming external
                # discovery routes for an already-known workspace defect.
                state.record_evidence(snapshot, usable=True)
                state.phase = LoopPhase.ACT
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
                    },
                )
        last_prompt_phase = _sync_phase_tool_transcript(
            messages, state=state, last_prompt_phase=last_prompt_phase, stage=stage
        )

        baseline_ready = _target_evidence_ready(
            state,
            require_rag=require_rag,
            fresh_java_target=fresh_java_target,
            compile_backed_java=compile_backed_java,
        )
        if (
            implementation_requires_mutation
            and state.workspace_changed
            and state.validation_status == "PROJECT_BUILD_DEFERRED"
            and bounded_root_execution_authority
            and baseline_ready
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
            and state.workspace_changed
            and state.validation_status == "DEFERRED"
            and baseline_ready
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

        if implementation_requires_mutation and state.workspace_changed and state.validation_status == "PASS" and baseline_ready:
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
                continue
            if compile_status == "FAIL":
                state.record_failure(
                    "target_compile",
                    "target compiler reported task-owned source defects",
                )
                state.phase = LoopPhase.ACT
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
                reason="live HostRunState projected verifier repair to a host-bound whole-file rewrite",
                details={"selected_tools": [
                    _tool_name(schema) for schema in phase_tools if _tool_name(schema)
                ]},
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
            mutation_is_ready = is_mutation_ready(messages, state)
            if mutation_is_ready and baseline_ready:
                state.phase = LoopPhase.ACT
                continue
            if mutation_is_ready and require_rag and not baseline_ready:
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
            required_evidence_choice
            and require_rag
            and not baseline_ready
            and state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER}
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
            if forced_evidence_tool == "external_mcp_call":
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
            and state.mutation_context.is_new_file
            and state.mutation_context.is_mutation_ready
            and require_rag
            and not baseline_ready
        ):
            messages.append({
                "role": "system",
                "content": (
                    "The host target is a NEW reserved Java file and does not exist yet. "
                    "Do not search for that filename. Before writing code, retrieve task-relevant "
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

        turn_request = replace(
            request,
            tools=phase_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
        )
        verifier_relative_files = (
            (state.mutation_context.target_path,)
            if (
                forced_verifier == "java_diagnostics"
                and not bounded_root_execution_authority
                and state.mutation_context is not None
                and state.mutation_context.target_path.casefold().endswith(".java")
            )
            else ()
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
                rejected_routes = _consume_rejected_evidence_fixed_point(
                    state,
                    rejection_payloads,
                    phase_names,
                    forced_evidence_tool=forced_evidence_tool,
                    forced_evidence_arguments=forced_evidence_arguments,
                )
                if rejected_routes:
                    # A repeated schema/protocol rejection means this evidence route
                    # has reached a semantic fixed point. Consume only that route and
                    # continue through the remaining evidence frontier instead of
                    # aborting the whole generation task.
                    emit_root_cause(
                        "rejected_evidence_route_exhausted",
                        stage=stage,
                        operation="generate_with_tools",
                        gate="semantic_fixed_point",
                        result="SKIP",
                        reason="repeated rejected evidence call; advancing to next reviewed route",
                        details={
                            "step_index": state.step_index,
                            "phase": state.phase.value,
                            "routes": list(rejected_routes),
                        },
                    )
                    continue
            if repeated:
                raise _fixed_point_error(state)
            continue

        if (
            state.phase is LoopPhase.ACT
            and state.validation_status == "FAIL"
            and turn.tool_calls
        ):
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
                        "model authored corrected source; host bound operation, target path, "
                        "and live-SHA whole-file rewrite semantics"
                    ),
                    details={
                        "target_path": (
                            state.mutation_context.target_path
                            if state.mutation_context is not None
                            else None
                        )
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
            if implementation_requires_mutation and not state.workspace_changed:
                if is_mutation_ready(messages, state) and baseline_ready:
                    state.phase = LoopPhase.ACT
                    messages.append({"role": "assistant", "content": content})
                    continue
                raise ModelConfigurationError(
                    "Writable coder returned prose before a reviewed source mutation was applied."
                )
            return content

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
                call.name, call.arguments, state.mutation_context
            )
            if target_error:
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": target_error.partition(":")[0],
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
                return call, {
                    "ok": True,
                    "tool": call.name,
                    **metadata,
                    "result": result,
                }
            except Exception as exc:  # noqa: BLE001 - tool failures become typed recovery observations
                if is_evidence_tool(call):
                    state.record_query(call.name, call.arguments)
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
                        refreshed = _reconcile_materialized_target_from_workspace(state, runtime)
                        if refreshed is not None:
                            # The verifier obligation is still valid, but the model's exact-match
                            # precondition was stale. Re-issue one repair contract against the live
                            # workspace snapshot instead of replaying the stale edit or duplicating
                            # the full source in a second refresh message.
                            state.repair_guidance_fingerprint = None
                        state.phase = (
                            LoopPhase.ACT
                            if state.mutation_context and state.mutation_context.is_mutation_ready
                            else LoopPhase.OBSERVE
                        )
                    elif code in {
                        "MUTATION_TARGET_DRIFT",
                        "MUTATION_TARGET_UNBOUND",
                        "MUTATION_TARGET_CREATION_CONFLICT",
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
                        and state.workspace_changed
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
                recorded = state.record_evidence(payload.get("result"), usable=usable)
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
                        # Later authored fragments retain model-owned file selection inside
                        # bounded roots, but they must first observe the live staged code
                        # produced by earlier fragments. Fresh workspace evidence satisfies
                        # that refresh without falsely pinning the fragment to one file.
                        state.phase = LoopPhase.ACT
                    elif (
                        implementation_requires_mutation
                        and state.mutation_context
                        and state.mutation_context.is_mutation_ready
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
            reason="progress" if progress else "no_progress",
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
