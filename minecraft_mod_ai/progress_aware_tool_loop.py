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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .agent_intent import implementation_requested
from .model_adapters import GenerationRequest, ModelConfigurationError
from .model_context_budget import bounded_tool_message, fit_messages_to_context
from .root_cause_trace import emit_root_cause, trace_scope
from .source_mutation_contract import mutation_history_applied, mutation_payload_applied
from .value_shapes import as_sequence as _sequence, structured_payload as _structured_payload


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
    "run_gradle_build",
    "gradle_build",
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
_MODEL_REJECTION_TOOL_NAME = "__mmm_rejected_tool_call__"
_HOST_AUTHORITY_ROLES = frozenset({"system", "developer", "tool"})
_EXISTING_TARGET_EVIDENCE_SOURCES = frozenset({"host_exact_source", "mutation_receipt"})
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
    while clean.startswith("./"):
        clean = clean[2:]
    return clean


def _normalized_target_path(value: Any) -> str:
    return _canonical_mutation_path(value)


def _is_workspace_file_path(path: str) -> bool:
    clean = _canonical_mutation_path(path)
    if not clean or clean.startswith("/") or "://" in clean:
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
    if name == "external_mcp_call":
        capability = str(arguments.get("capability", "")).strip()
        return f"{name}:{capability}" if capability else name
    return name


def retrieval_query_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    parts = [str(tool_name or "").strip()]
    query = normalize_retrieval_query(arguments.get("query"))
    for key in ("index_path", "path", "file", "target_path", "symbol", "symbol_name"):
        value = str(arguments.get(key) or "").strip().casefold()
        if value:
            parts.append(f"{key}={value}")
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


def _task_authority_context(payload: Mapping[str, Any]) -> TargetMutationContext | None:
    module = payload.get("module")
    task = _evidence_task_from_module(module)
    direct_primary = _canonical_mutation_path(payload.get("primary_path"))
    direct_writable = tuple(
        path for path in (
            _canonical_mutation_path(item) for item in _sequence(payload.get("writable_paths"))
        ) if _is_workspace_file_path(path)
    )
    direct_reuse = str(payload.get("reuse_action") or "").strip().casefold()

    candidates: list[tuple[str, str, bool]] = []
    writable: list[str] = list(direct_writable)
    creatable: list[str] = []

    if isinstance(task, Mapping):
        task_reuse = str(task.get("reuse_action") or "").strip().casefold()
        bindings = [
            item for item in _sequence(task.get("production_bindings"))
            if isinstance(item, Mapping)
        ]
        for binding in bindings:
            reuse = str(binding.get("reuse_action") or task_reuse or direct_reuse).strip().casefold()
            for anchor in _sequence(binding.get("owned_anchors")):
                if not isinstance(anchor, Mapping):
                    continue
                path, symbol = _anchor_path_and_symbol(anchor)
                if not path:
                    continue
                if path not in writable:
                    writable.append(path)
                fresh = (
                    str(anchor.get("status") or "").strip().casefold() == "host_reserved"
                    or reuse == "fresh"
                )
                if fresh and path not in creatable:
                    creatable.append(path)
                if str(anchor.get("kind") or "").strip().casefold() == "symbol":
                    candidates.append((path, symbol, fresh))
        for anchor in _sequence(task.get("owned_anchors")):
            if not isinstance(anchor, Mapping):
                continue
            path, symbol = _anchor_path_and_symbol(anchor)
            if not path:
                continue
            if path not in writable:
                writable.append(path)
            fresh = str(anchor.get("status") or "").strip().casefold() == "host_reserved"
            if fresh and path not in creatable:
                creatable.append(path)
            if str(anchor.get("kind") or "").strip().casefold() == "symbol":
                candidates.append((path, symbol, fresh))

    if direct_primary and direct_primary not in writable:
        writable.insert(0, direct_primary)

    chosen: tuple[str, str, bool] | None = None
    if direct_primary:
        for item in candidates:
            if item[0] == direct_primary:
                chosen = item
                break
        if chosen is None:
            chosen = (direct_primary, "", direct_reuse == "fresh")
    else:
        unique = []
        for item in candidates:
            if item not in unique:
                unique.append(item)
        if len(unique) == 1:
            chosen = unique[0]

    if chosen is None:
        return None

    path, symbol, fresh = chosen
    if path not in writable:
        writable.insert(0, path)
    if fresh and path not in creatable:
        creatable.append(path)
    return TargetMutationContext(
        target_path=path,
        target_symbol=symbol or None,
        is_new_file=fresh,
        evidence_source="host_task_authority",
        writable_paths=tuple(writable),
        creatable_paths=tuple(creatable),
        target_pinned=True,
    )


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
        if other is None:
            return self
        left = _canonical_mutation_path(self.target_path)
        right = _canonical_mutation_path(other.target_path)
        if self.target_pinned and left and right and left != right:
            return self
        if other.target_pinned and right and left and left != right:
            return other
        if left and right and left != right:
            return other

        writable = tuple(dict.fromkeys((*self.writable_paths, *other.writable_paths)))
        creatable = tuple(dict.fromkeys((*self.creatable_paths, *other.creatable_paths)))
        existing: TargetMutationContext | None = None
        if (
            not other.is_new_file
            and str(other.evidence_source or "").strip() in _EXISTING_TARGET_EVIDENCE_SOURCES
        ):
            existing = other
        elif (
            not self.is_new_file
            and str(self.evidence_source or "").strip() in _EXISTING_TARGET_EVIDENCE_SOURCES
        ):
            existing = self

        target_path = other.target_path or self.target_path
        target = _canonical_mutation_path(target_path)
        if existing is not None and target:
            creatable = tuple(
                item for item in creatable
                if _canonical_mutation_path(item) != target
            )

        return TargetMutationContext(
            target_path=target_path,
            target_symbol=other.target_symbol or self.target_symbol,
            source_body=(
                existing.source_body
                if existing is not None and existing.source_body is not None
                else other.source_body or self.source_body
            ),
            start_line=(
                existing.start_line
                if existing is not None and existing.start_line is not None
                else other.start_line if other.start_line is not None else self.start_line
            ),
            end_line=(
                existing.end_line
                if existing is not None and existing.end_line is not None
                else other.end_line if other.end_line is not None else self.end_line
            ),
            is_new_file=False if existing is not None else (other.is_new_file or self.is_new_file),
            evidence_source=(
                existing.evidence_source
                if existing is not None
                else other.evidence_source or self.evidence_source
            ),
            base_revision_sha=(
                existing.base_revision_sha
                if existing is not None and existing.base_revision_sha is not None
                else other.base_revision_sha or self.base_revision_sha
            ),
            writable_paths=writable,
            creatable_paths=creatable,
            target_pinned=self.target_pinned or other.target_pinned,
        )


def _extract_search_context(payload: Mapping[str, Any]) -> TargetMutationContext | None:
    hits = payload.get("hits") or payload.get("results")
    for hit in _sequence(hits):
        if not isinstance(hit, Mapping):
            continue
        meta = hit.get("metadata") if isinstance(hit.get("metadata"), Mapping) else {}
        path = _canonical_mutation_path(
            hit.get("source_path") or hit.get("path") or hit.get("file") or hit.get("uri")
            or meta.get("path") or meta.get("source_path")
        )
        if not _is_workspace_file_path(path):
            continue
        text = hit.get("text") or hit.get("snippet") or hit.get("code") or hit.get("content") or hit.get("source")
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(item) for item in text)
        symbol = str(hit.get("symbol") or hit.get("function") or hit.get("name") or meta.get("symbol") or "").strip()
        return TargetMutationContext(
            target_path=path,
            target_symbol=symbol or None,
            source_body=text if _is_code_bearing_text(text) else None,
            start_line=hit.get("start_line") if isinstance(hit.get("start_line"), int) else None,
            end_line=hit.get("end_line") if isinstance(hit.get("end_line"), int) else None,
            evidence_source="search_code_rag",
        )
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
            exact = _extract_mutation_context_from_payload(initial)
            if (
                exact is not None
                and _canonical_mutation_path(exact.target_path)
                == _canonical_mutation_path(authority.target_path)
                and exact.source_body
            ):
                return replace(
                    authority,
                    source_body=exact.source_body,
                    start_line=exact.start_line,
                    end_line=exact.end_line,
                    is_new_file=False,
                    evidence_source="host_exact_source",
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


def is_mutation_ready(messages: Sequence[Mapping[str, Any]], state: HostRunState) -> bool:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().casefold()
        payload = _structured_payload(message.get("content"))
        if not isinstance(payload, Mapping):
            continue
        authority_allowed = role in _HOST_AUTHORITY_ROLES or (
            role == "user" and _trusted_internal_user_payload(payload)
        )
        if not authority_allowed:
            continue
        context = _task_authority_context(payload)
        if context is None:
            continue
        with state._lock:
            if state.mutation_context is None or not state.mutation_context.target_pinned:
                state.mutation_context = context
            elif (
                _canonical_mutation_path(state.mutation_context.target_path)
                == _canonical_mutation_path(context.target_path)
            ):
                state.mutation_context = state.mutation_context.merge(context)

    for message in messages:
        if not isinstance(message, Mapping):
            continue
        payload = _structured_payload(message.get("content"))
        if payload is None:
            continue
        context = _extract_mutation_context_from_payload(payload)
        if context is None:
            continue
        with state._lock:
            if state.mutation_context is None:
                role = str(message.get("role") or "").strip().casefold()
                if role not in _HOST_AUTHORITY_ROLES:
                    continue
                state.mutation_context = context
            else:
                state.mutation_context = state.mutation_context.merge(context)

    with state._lock:
        return bool(state.mutation_context and state.mutation_context.is_mutation_ready)


def _mutation_target_error(
    tool_name: str,
    arguments: Mapping[str, Any],
    context: TargetMutationContext | None,
) -> str | None:
    if tool_name != "apply_source_edit":
        return None
    if context is None or not context.is_mutation_ready:
        return "MUTATION_TARGET_UNBOUND: no host-pinned mutation target is READY"
    supplied = ""
    for key in _SOURCE_EDIT_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            supplied = _canonical_mutation_path(value)
            break
    pinned = _canonical_mutation_path(context.target_path)
    allowed = set(context.writable_paths) or ({pinned} if pinned else set())
    if not supplied or not pinned:
        return "MUTATION_TARGET_UNBOUND: apply_source_edit requires an explicit host-bound path"
    if supplied not in allowed:
        return (
            f"MUTATION_TARGET_DRIFT: writable exact-set {sorted(allowed)!r} "
            f"does not authorize {supplied!r}"
        )
    operation = str(arguments.get("operation") or "").strip().casefold()
    if operation in _SOURCE_CREATE_OPERATIONS:
        can_create = supplied in set(context.creatable_paths) or (
            supplied == pinned and context.is_new_file
        )
        if not can_create:
            return (
                "MUTATION_TARGET_CREATION_CONFLICT: create operation is not authorized "
                f"for existing target {supplied!r}"
            )
    return None


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
    if status in _VERIFIER_FAIL_STATUSES:
        return "FAIL"
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
        return "FAIL" if errors else "PASS"
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


def _atomic_output_recovery_instruction(request: GenerationRequest) -> str:
    names = frozenset(_tool_name(schema) for schema in request.tools if _tool_name(schema))
    if names & _MUTATION_ACT_TOOLS:
        return (
            "The preceding assistant action exceeded the bounded output allowance and is discarded. "
            "Do not continue, reproduce, or complete that oversized payload. Call exactly one visible "
            "source-mutation tool now with exactly one small semantic edit and no prose. For a new Java "
            "file, the first action must be create_java_type with only package_name and an empty type "
            "declaration; never create a complete Java file with create_file. After each tool observation, "
            "add at most one import with add_java_import or one field/constructor/method/nested declaration "
            "with insert_java_member. For an existing file, use one bounded replace_exact/insert action. "
            "The host will preserve the same mutation target and workspace state between actions."
        )
    return (
        "The preceding assistant action exceeded the bounded output allowance and is discarded. "
        "Do not continue that oversized payload. Produce exactly one concise visible tool call or one "
        "concise final answer using the already-grounded state; do not emit a long reconstruction."
    )


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
        name = str(payload.get("rejected_name") or "").strip()
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
            f"Step {item.step_index} {item.phase_before}->{item.phase_after} "
            f"{item.localization_stage_before}->{item.localization_stage_after} "
            f"calls={calls} results={results} progress={item.turn_made_progress}"
        )
    return "\n".join(lines)


_JAVA_API_EVIDENCE_RE = re.compile(
    r"(?:\b(?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)\.[A-Za-z0-9_.$]+"
    r"|\b(?:package|import)\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+"
    r"|\b(?:class|interface|record|enum)\s+[A-Za-z_$][\w$]*)"
)
_ATOMIC_OUTPUT_RECOVERY_MARKER = "MMM_ATOMIC_OUTPUT_RECOVERY_V1"


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


def _java_evidence_texts(value: Any) -> tuple[str, ...]:
    texts: list[str] = []
    if isinstance(value, Mapping):
        for key in ("parsed_text", "text", "content", "snippet", "code", "source", "source_text", "body"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                texts.append(raw)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                texts.extend(str(item) for item in raw if isinstance(item, str) and item.strip())
        for key in ("hits", "results", "records", "documents", "chunks", "resources", "symbols", "evidence"):
            raw = value.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                for item in raw:
                    texts.extend(_java_evidence_texts(item))
        for key in ("structured_content", "result", "data"):
            child = value.get(key)
            if child is not None:
                texts.extend(_java_evidence_texts(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            texts.extend(_java_evidence_texts(item))
    return tuple(texts)


def _authoritative_java_evidence(value: Any) -> bool:
    """Return whether evidence is strong enough to authorize a fresh Java mutation."""
    if not isinstance(value, Mapping) or not value:
        return False

    if _mapping_schema(value, "mmm/rag-result-v2"):
        return False

    if _mapping_schema(value, "mmm/java-symbols-v1"):
        def has_symbols(item: Any) -> bool:
            if isinstance(item, Mapping):
                symbols = item.get("symbols")
                if isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
                    if any(isinstance(symbol, Mapping) and bool(symbol) for symbol in symbols):
                        return True
                return any(has_symbols(child) for child in item.values())
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                return any(has_symbols(child) for child in item)
            return False
        return has_symbols(value)

    if _mapping_schema(value, "mmm/code-rag-result-v1"):
        return any(_JAVA_API_EVIDENCE_RE.search(text) for text in _java_evidence_texts(value))

    if any(_JAVA_API_EVIDENCE_RE.search(text) for text in _java_evidence_texts(value)):
        return True

    def has_mapping_records(item: Any) -> bool:
        if isinstance(item, Mapping):
            mappings = item.get("mappings")
            if isinstance(mappings, Mapping) and bool(mappings):
                return True
            if isinstance(mappings, Sequence) and not isinstance(mappings, (str, bytes, bytearray)):
                if any(isinstance(entry, Mapping) and bool(entry) for entry in mappings):
                    return True
            return any(has_mapping_records(child) for child in item.values())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return any(has_mapping_records(child) for child in item)
        return False

    return has_mapping_records(value)


def _target_evidence_ready(
    state: "HostRunState",
    *,
    require_rag: bool,
    fresh_java_target: bool,
) -> bool:
    if not require_rag:
        return True
    if fresh_java_target:
        return state.has_authoritative_java_evidence
    return state.has_fresh_evidence


def _completion_boundary_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        text = str(current).casefold()
        if name == "LlamaCompletionBoundaryError":
            return True
        if (
            "completion boundary" in text
            or "completion token limit" in text
            or "maximum completion" in text
            or ("finish_reason" in text and "length" in text)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


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
    created_paths: set[str] = field(default_factory=set)
    workspace_changed: bool = False
    validation_status: str = "PENDING"
    latest_verifier_tool: str | None = None
    latest_verifier_errors: tuple[dict[str, Any], ...] = ()
    latest_verifier_fingerprint: str | None = None
    repair_guidance_fingerprint: str | None = None
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
            self.attempted_sources.add(retrieval_source_key(tool_name, arguments))
            return True

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
            if fp in self.evidence_fingerprints:
                return False
            self.evidence_fingerprints.add(fp)
            if _fresh_java_context(self.mutation_context) and _authoritative_java_evidence(value):
                self.authoritative_java_evidence_fingerprints.add(fp)
            context = _extract_mutation_context_from_payload(value)
            if context is not None and self.mutation_context is not None:
                self.mutation_context = self.mutation_context.merge(context)
            return True

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
                if signature:
                    repeated = signature in self.unchanged_mutation_fingerprints
                    self.unchanged_mutation_fingerprints.add(signature)
                    if repeated:
                        self.semantic_fixed_point = True
                return False
            if signature:
                self.mutation_fingerprints.add(signature)
            self.applied_mutations.append(tool_name)
            self.workspace_changed = True
            self.validation_status = "PENDING"
            self.repair_guidance_fingerprint = None
            operation = str(arguments.get("operation") or "").strip().casefold()
            path = ""
            for key in _SOURCE_EDIT_PATH_KEYS:
                value = arguments.get(key)
                if isinstance(value, str) and value.strip():
                    path = _canonical_mutation_path(value)
                    break
            if path and operation in _SOURCE_CREATE_OPERATIONS:
                self.created_paths.add(path)
            if self.mutation_context is not None and path == _canonical_mutation_path(self.mutation_context.target_path):
                body = self.mutation_context.source_body
                if operation in _SOURCE_CREATE_OPERATIONS and isinstance(arguments.get("content"), str):
                    body = str(arguments["content"])
                elif operation == "replace_exact" and isinstance(body, str):
                    old, new = arguments.get("old"), arguments.get("new")
                    if isinstance(old, str) and isinstance(new, str) and old and body.count(old) == 1:
                        body = body.replace(old, new, 1)
                remaining_creatable = tuple(
                    item for item in self.mutation_context.creatable_paths
                    if _canonical_mutation_path(item) != path
                )
                self.mutation_context = replace(
                    self.mutation_context,
                    source_body=body,
                    is_new_file=False,
                    evidence_source="mutation_receipt",
                    creatable_paths=remaining_creatable,
                )
            return True

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
            changed = status != self.validation_status or fp != self.latest_verifier_fingerprint
            self.validation_status = status
            self.latest_verifier_tool = tool_name
            self.latest_verifier_errors = tuple(errors)
            self.latest_verifier_fingerprint = fp
            if status != "FAIL":
                self.repair_guidance_fingerprint = None
            return changed

    def take_verifier_repair_guidance(self) -> str | None:
        with self._lock:
            if (
                self.validation_status != "FAIL"
                or not self.latest_verifier_fingerprint
                or self.latest_verifier_fingerprint == self.repair_guidance_fingerprint
            ):
                return None
            self.repair_guidance_fingerprint = self.latest_verifier_fingerprint
            context = self.mutation_context
            source = context.source_body if context and isinstance(context.source_body, str) else None
            payload = {
                "verifier": self.latest_verifier_tool,
                "diagnostics": list(self.latest_verifier_errors),
                "target_path": context.target_path if context else None,
                "target_is_new_file": context.is_new_file if context else None,
                "writable_paths": list(context.writable_paths) if context else [],
                "current_source_sha256": (
                    hashlib.sha256(source.encode("utf-8")).hexdigest() if source is not None else None
                ),
                "current_source": source,
            }
        return (
            "MMM_CORE_VERIFIER_REPAIR_V5\n"
            "The verifier failure is the active repair obligation. Do not restart generation, "
            "do not search unrelated ecosystem candidates, and do not recreate an existing path. "
            "The payload includes the exact host-tracked current source and its SHA-256. "
            "Any earlier host_reserved/fresh metadata is pre-materialization history only and "
            "does not authorize a second create after target_is_new_file becomes false. "
            "Edit that existing source with a non-create operation when target_is_new_file is false. "
            "Use the diagnostics below against the host-pinned target and make one materially "
            "different source edit. The next successful mutation goes directly back to VERIFY.\n"
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
            self.no_progress_streak = int(repeated)
            return repeated

    def clear_no_progress_result(self) -> None:
        with self._lock:
            self.seen_no_progress_digests.clear()
            self.semantic_fixed_point = False
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
    if _tool_name(schema) != "apply_source_edit" or context is None or context.is_new_file:
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
    if isinstance(operation, dict):
        enum = operation.get("enum")
        if isinstance(enum, list):
            operation["enum"] = [
                value for value in enum
                if str(value).strip().casefold() not in _SOURCE_CREATE_OPERATIONS
            ]
    description = str(function.get("description") or "").strip()
    suffix = "Existing host-pinned target: create/write operations are not permitted; edit the current file."
    function["description"] = f"{description} {suffix}".strip()
    return cloned


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
    del role, localization_active
    by_name = {_tool_name(schema): schema for schema in exposed_tools if _tool_name(schema)}
    attempted = set(attempted_sources)
    if phase == LoopPhase.ACT:
        names = ["apply_source_edit"] if "apply_source_edit" in by_name else [
            name for name in by_name if name in _MUTATION_ACT_TOOLS
        ]
    elif phase == LoopPhase.VERIFY:
        names = [name for name in by_name if name in _VERIFY_TOOLS]
    elif phase == LoopPhase.RECOVER:
        names = [
            name for name in (
                "search_code_rag", "java_workspace_symbols", "search_project_rag",
                "external_mcp_call", "read_reuse_source",
            )
            if name in by_name and name not in attempted
        ]
    else:
        stage = mutation_context.localization_stage if mutation_context else LocalizationStage.NEED_FILE
        if mutation_context and mutation_context.is_new_file and mutation_context.is_mutation_ready:
            preferred = (
                "search_code_rag", "java_workspace_symbols", "external_mcp_call"
            )
        elif stage == LocalizationStage.NEED_FILE:
            preferred = ("search_code_rag", "search_project_rag")
        elif stage == LocalizationStage.NEED_SYMBOL:
            preferred = ("java_workspace_symbols", "search_code_rag", "search_project_rag")
        elif stage == LocalizationStage.NEED_BODY:
            preferred = ("search_code_rag", "java_workspace_symbols", "search_project_rag")
        else:
            preferred = ("search_project_rag", "search_code_rag")
        names = [name for name in preferred if name in by_name and name not in attempted]
        if names:
            fresh_reserved = bool(
                mutation_context
                and mutation_context.is_new_file
                and mutation_context.is_mutation_ready
            )
            if (
                semantic_retrieval_choice
                and fresh_reserved
                and "search_code_rag" in names
                and "search_code_rag" not in attempted
            ):
                # Fresh Java starts with current-project code evidence, but once that
                # route is exhausted the model must be allowed to choose among the
                # remaining reviewed evidence routes instead of being force-fed one.
                names = ["search_code_rag"]
            elif not semantic_retrieval_choice:
                names = [names[0]]
    return tuple(
        _source_edit_schema_for_context(by_name[name], mutation_context)
        for name in names if name in by_name
    )


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
) -> Any:
    del router
    fitted = fit_messages_to_context(messages, config=config, tools=request.tools)
    if tuple(messages) != tuple(fitted):
        messages[:] = [dict(message) for message in fitted]
    turn_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    try:
        return adapter.generate_turn(turn_request)
    except Exception as exc:
        if not _completion_boundary_error(exc):
            raise
        already_recovered = any(
            isinstance(message.get("content"), str)
            and _ATOMIC_OUTPUT_RECOVERY_MARKER in str(message.get("content"))
            for message in messages
            if isinstance(message, Mapping)
        )
        if already_recovered:
            raise
        recovery_instruction = (
            _ATOMIC_OUTPUT_RECOVERY_MARKER
            + "\n"
            + _atomic_output_recovery_instruction(turn_request)
        )
        recovery_messages = [*messages, {"role": "system", "content": recovery_instruction}]
        fitted_recovery = fit_messages_to_context(
            recovery_messages, config=config, tools=request.tools
        )
        messages[:] = [dict(message) for message in fitted_recovery]
        recovery_request = replace(
            turn_request,
            messages=tuple(messages),
            media_paths=(),
        )
        emit_root_cause(
            "atomic_output_boundary_recovery",
            operation="generate_with_tools",
            gate="completion_boundary",
            result="RETRY",
            reason=f"{type(exc).__name__}: {exc}",
            details={"tool_choice": tool_choice},
        )
        return adapter.generate_turn(recovery_request)


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
    for raw in messages:
        message = dict(raw)
        role = str(message.get("role") or "")
        if role == "tool":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                observations.append(content)
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                compacted.append({"role": "assistant", "content": content})
            continue
        compacted.append(message)
    compacted.append({
        "role": "system",
        "content": (
            f"MMM_PHASE_HANDOFF {last_prompt_phase.value}->{next_phase.value}\n"
            "Prior-phase tool calls are closed and cannot be replayed.\n"
            + "\n".join(f"Observation {i}:\n{value}" for i, value in enumerate(observations, 1))
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
    for name in ("java_diagnostics", "jdt_diagnostics", "run_gradle_build", "gradle_build", "run_gametest"):
        if name in names and name not in unavailable:
            return name
    return None


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


def _finalize_without_tools(
    router: Any,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    *,
    instruction: str,
    empty_error: str,
) -> str:
    final_messages = [*messages, {"role": "system", "content": instruction}]
    final_request = replace(
        request,
        messages=tuple(final_messages),
        media_paths=(),
        tools=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )
    mutable = [dict(message) for message in final_messages]
    turn = _generate_turn_with_context_recovery(
        router,
        config=config,
        adapter=adapter,
        request=final_request,
        messages=mutable,
        media_paths=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )
    if turn.tool_calls:
        raise ModelConfigurationError("Agent emitted tool calls after the host disabled tools.")
    content = turn.content.strip()
    if not content:
        raise ModelConfigurationError(empty_error)
    return content


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
    from .agent_capability_context import reviewed_mcp_servers_for_model_role, skills_for_tool
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
    unavailable_verifiers: set[str] = set()
    required_evidence_choice = False
    implementation = bool(
        role in {"coder", "coder_safe"}
        and stage == "generation"
        and implementation_requested(request.messages)
    )
    host_grounded = host_baseline_evidence_ready(request.messages)
    mutation_ready = is_mutation_ready(messages, state)
    fresh_java_target = bool(
        implementation
        and state.mutation_context
        and state.mutation_context.is_new_file
        and _canonical_mutation_path(state.mutation_context.target_path).casefold().endswith(".java")
    )
    require_rag = bool(
        role in {"coder", "coder_safe"}
        and all_names & _RAG_EVIDENCE_TOOLS
        and (
            (router._agent_require_fresh_evidence and not host_grounded)
            or fresh_java_target
        )
    )

    if require_rag:
        state.phase = LoopPhase.OBSERVE
    elif implementation and mutation_ready and not mutation_history_applied(messages):
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
            "require_rag": require_rag,
            "implementation_requires_mutation": implementation,
            "mutation_ready": mutation_ready,
            "initial_phase": state.phase.value,
        },
    )

    while True:
        last_prompt_phase = _sync_phase_tool_transcript(
            messages, state=state, last_prompt_phase=last_prompt_phase, stage=stage
        )

        if state.semantic_fixed_point:
            raise _fixed_point_error(state)

        baseline_ready = _target_evidence_ready(
            state, require_rag=require_rag, fresh_java_target=fresh_java_target
        )
        if implementation and state.workspace_changed and state.validation_status == "PASS" and baseline_ready:
            state.termination_reason = "VERIFICATION_PASSED"
            return _finalize_without_tools(
                router,
                config,
                adapter,
                request,
                messages,
                instruction=(
                    "Host verification passed after the applied source mutation. "
                    "Do not call more tools. Return only the verified implementation summary."
                ),
                empty_error="Agent returned an empty final response after verification passed.",
            )

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
            localization_active=implementation,
            semantic_retrieval_choice=bool(require_rag and not baseline_ready),
        )

        forced_verifier: str | None = None
        if state.phase == LoopPhase.VERIFY:
            forced_verifier = _verifier_tool(phase_tools, unavailable_verifiers)
            if forced_verifier is None:
                raise ModelConfigurationError(
                    "VERIFIER_UNAVAILABLE: no healthy verifier remains for the mutated workspace."
                )
            phase_tools = tuple(schema for schema in phase_tools if _tool_name(schema) == forced_verifier)

        if implementation and state.phase == LoopPhase.ACT and not phase_tools:
            raise ModelConfigurationError("MUTATION_TOOL_UNAVAILABLE: no reviewed source mutation tool is exposed.")
        if implementation and state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER} and not phase_tools:
            if is_mutation_ready(messages, state) and baseline_ready:
                state.phase = LoopPhase.ACT
                continue
            raise ModelConfigurationError(
                "MUTATION_LOCALIZATION_STALLED: no untried relevant source-evidence route remains."
            )

        phase_names = frozenset(_tool_name(schema) for schema in phase_tools if _tool_name(schema))
        tool_choice = request.tool_choice
        parallel = request.parallel_tool_calls

        if (
            required_evidence_choice
            and require_rag
            and not baseline_ready
            and state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER}
            and len(phase_names) > 1
        ):
            # The host requires evidence, not a particular semantic route. Force one
            # visible tool call while leaving route selection to the model.
            tool_choice = "required"
            parallel = False
        elif state.phase == LoopPhase.ACT:
            mutation_names = [name for name in phase_names if name in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel = False
        elif forced_verifier:
            tool_choice = {"type": "function", "function": {"name": forced_verifier}}
            parallel = False
        elif state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER} and len(phase_names) == 1:
            name = next(iter(phase_names))
            tool_choice = {"type": "function", "function": {"name": name}}
            parallel = False

        if state.phase == LoopPhase.ACT:
            guidance = state.take_verifier_repair_guidance()
            if guidance:
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
        turn = _generate_turn_with_context_recovery(
            router,
            config=config,
            adapter=adapter,
            request=turn_request,
            messages=messages,
            media_paths=request.media_paths if state.step_index == 1 else (),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
        )

        rejection = _model_tool_rejection_feedback(turn.tool_calls)
        if rejection is not None:
            feedback, rejection_payloads = rejection
            for payload in rejection_payloads:
                state.record_failure(
                    str(payload.get("rejected_name") or "model_tool_call"),
                    str(
                        payload.get("error")
                        or payload.get("failure_code")
                        or "model tool call rejected"
                    ),
                )
            repeated = state.record_no_progress_result({
                "phase": state.phase.value,
                "validation": state.validation_status,
                "verifier": state.latest_verifier_fingerprint,
                "model_tool_rejections": rejection_payloads,
            })
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
                raise _fixed_point_error(state)
            continue

        if not turn.tool_calls:
            content = turn.content.strip()
            if not content:
                raise ModelConfigurationError("Tool-capable model returned an empty final response.")
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
            if implementation and state.phase in {LoopPhase.ACT, LoopPhase.VERIFY, LoopPhase.RECOVER}:
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
                            "Prose cannot complete this implementation."
                        ),
                    },
                ])
                continue
            if implementation and not state.workspace_changed:
                if is_mutation_ready(messages, state) and baseline_ready:
                    state.phase = LoopPhase.ACT
                    messages.append({"role": "assistant", "content": content})
                    continue
                raise ModelConfigurationError(
                    "Writable coder returned prose before a reviewed source mutation was applied."
                )
            return content

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
            return (
                call.name in _LOCALIZATION_EVIDENCE_TOOLS
                or call.name in _RAG_EVIDENCE_TOOLS
                or (
                    state.phase == LoopPhase.RECOVER
                    and call.name in _RECOVERY_EVIDENCE_TOOLS
                )
                or (
                    call.name == "external_mcp_call"
                    and bool(_external_rag_capability(call.arguments))
                )
            )

        def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
            metadata = {"skills": list(skills_for_tool(stage, call.name, model_role=role))}
            if call.name not in phase_names:
                error = (
                    f"PHASE_PROTOCOL_VIOLATION: {call.name!r} is not allowed in "
                    f"{state.phase.value}; allowed={sorted(phase_names)}"
                )
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": "PHASE_PROTOCOL_VIOLATION",
                    "error": error,
                }

            if is_evidence_tool(call):
                if state.is_query_attempted(call.name, call.arguments):
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
            except Exception as exc:
                if is_evidence_tool(call):
                    state.record_query(call.name, call.arguments)
                error = f"{type(exc).__name__}: {exc}"
                lowered = error.casefold()
                failure_code = "TOOL_RUNTIME_UNAVAILABLE"
                if call.name in _VERIFY_TOOLS:
                    if any(marker in lowered for marker in ("no such file", "not found", "does not exist", "outside", "unsafe path")):
                        failure_code = "VERIFIER_TARGET_INVALID"
                    elif any(marker in lowered for marker in ("argument", "schema", "invalid")):
                        failure_code = "VERIFIER_ARGUMENT_INVALID"
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **metadata,
                    "failure_code": failure_code,
                    "error": error,
                }

        executed = _execute_tool_waves(tuple(turn.tool_calls), execute)
        progress = False

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
                    progress = True
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
                                    "only the pre-create lifecycle and no longer authorizes create/write operations "
                                    "for this path. Future repairs must edit the current file in place."
                                ),
                            })
                    state.phase = LoopPhase.VERIFY if all_names & _VERIFY_TOOLS else LoopPhase.OBSERVE
                    if not all_names & _VERIFY_TOOLS:
                        state.validation_status = "PASS"
                else:
                    code = str(payload.get("failure_code") or "")
                    error = str(payload.get("error") or "MUTATION_UNCHANGED: no source-byte change")
                    state.record_failure(call.name, error)
                    if code in {
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
                if status in {"VERIFIER_ARGUMENT_INVALID", "VERIFIER_TARGET_INVALID"}:
                    raise ModelConfigurationError(
                        f"{status}: {payload.get('error', '')}"
                    )
                if status == "UNAVAILABLE":
                    unavailable_verifiers.add(call.name)
                    state.validation_status = "UNAVAILABLE"
                    state.record_failure(call.name, payload.get("error", "verifier unavailable"))
                    state.phase = LoopPhase.VERIFY
                    continue
                if state.record_verification(call.name, payload, status):
                    progress = True
                if status == "FAIL":
                    state.record_failure(call.name, "verification reported source defects")
                    if state.mutation_context and state.mutation_context.is_mutation_ready:
                        state.phase = LoopPhase.ACT
                    else:
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
                baseline_progress = bool(
                    state.phase == LoopPhase.OBSERVE
                    and require_rag
                    and recorded
                    and usable
                    and (
                        not fresh_java_target
                        or state.has_authoritative_java_evidence
                    )
                )
                if localization_progress or baseline_progress:
                    progress = True
                    if (
                        implementation
                        and state.mutation_context
                        and state.mutation_context.is_mutation_ready
                        and _target_evidence_ready(
                            state,
                            require_rag=require_rag,
                            fresh_java_target=fresh_java_target,
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
        result_info = _fixed_point_tool_results(executed)

        if progress:
            state.clear_no_progress_result()
            if _target_evidence_ready(
                state, require_rag=require_rag, fresh_java_target=fresh_java_target
            ):
                required_evidence_choice = False
        else:
            state.record_no_progress_result({
                "phase_before": phase_before.value,
                "phase_after": phase_after,
                "localization_before": loc_before,
                "localization_after": loc_after,
                "target": ctx_after,
                "validation": state.validation_status,
                "verifier": state.latest_verifier_fingerprint,
                "calls": call_info,
                "results": result_info,
            })
            if require_rag and not _target_evidence_ready(
                state, require_rag=require_rag, fresh_java_target=fresh_java_target
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
    "_atomic_output_recovery_instruction",
    "_fixed_point_tool_results",
    "_model_tool_rejection_feedback",
    "ExecutionStepTrace",
    "HostRunState",
    "LocalizationStage",
    "LoopPhase",
    "RetrievalDecision",
    "RetrievalNoProgressError",
    "RetrievalObservation",
    "RetrievalProgress",
    "TargetMutationContext",
    "evidence_fingerprint",
    "format_trajectory_summary",
    "generate_with_tools",
    "is_mutation_ready",
    "normalize_retrieval_query",
    "retrieval_query_signature",
    "retrieval_source_key",
]
