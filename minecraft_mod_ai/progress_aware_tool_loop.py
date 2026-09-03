from __future__ import annotations

"""Single host-owned execution loop for retrieve/act/observe/verify."""

import hashlib
import json
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import getattr_static
from pathlib import Path
from typing import Any

from .agent_intent import implementation_requested
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
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
from .source_mutation_contract import (
    mutation_history_applied,
    mutation_payload_applied,
)


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
    "inspect_existing_mod",
    "read_reuse_source",
})

_READ_OBSERVE_TOOLS = frozenset({
    "search_code_rag",
    "search_project_rag",
    "discover_ecosystem_resources",
    "inspect_modrinth_project",
    "inspect_github_repository",
    "inspect_huggingface_model",
    "inspect_existing_mod",
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


def normalize_retrieval_query(value: Any) -> str:
    """Canonicalize retrieval intent without relying on incidental model phrasing."""
    text = str(value or "").casefold()
    tokens = re.findall(r"[a-z0-9_.$:/+-]+", text)
    material = sorted({token for token in tokens if token not in _STOPWORDS})
    return " ".join(material)


def retrieval_source_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    if name == "external_mcp_call":
        capability = str(arguments.get("capability", "")).strip()
        return f"{name}:{capability}" if capability else name
    return name


def _is_workspace_file_path(path: str) -> bool:
    if not path or not isinstance(path, str):
        return False
    clean = path.strip().replace("\\", "/")
    if clean.startswith(("http://", "https://")) or "://" in clean:
        return False
    suffix = Path(clean).suffix.casefold()
    return "/" in clean or suffix in {
        ".java", ".json", ".toml", ".gradle", ".properties", ".txt", ".md", ".kt", ".groovy"
    }


def retrieval_query_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    name = str(tool_name or "").strip()
    norm_query = normalize_retrieval_query(arguments.get("query"))
    target = str(
        arguments.get("index_path")
        or arguments.get("path")
        or arguments.get("file")
        or arguments.get("target_path")
        or ""
    ).strip().casefold()
    symbol = str(
        arguments.get("symbol")
        or arguments.get("symbol_name")
        or arguments.get("function")
        or ""
    ).strip().casefold()
    parts = [name]
    if target:
        parts.append(f"target={target}")
    if symbol:
        parts.append(f"symbol={symbol}")
    if norm_query:
        parts.append(f"q={norm_query}")
    cursor = arguments.get("cursor") or arguments.get("offset_bytes")
    if cursor not in (None, "", 0, "0"):
        parts.append(f"cursor={cursor}")
    return ":".join(parts)


def _stable_value(value: Any, *, drop_volatile: bool) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if drop_volatile and key.casefold() in _VOLATILE_EVIDENCE_KEYS:
                continue
            stable_child = _stable_value(child, drop_volatile=drop_volatile)
            if stable_child not in (None, "", [], {}):
                result[key] = stable_child
        return result
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item, drop_volatile=drop_volatile) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item, drop_volatile=drop_volatile) for item in value]
    return value


def evidence_fingerprint(value: Any) -> str | None:
    """Compute a canonical stable digest for novel evidence checking."""
    stable = _stable_value(value, drop_volatile=True)
    if stable in (None, "", [], {}):
        return None
    try:
        canonical = json.dumps(
            stable,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        canonical = repr(stable)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



_EXISTING_TARGET_EVIDENCE_PREFIXES = (
    "search_code_rag",
    "sources_",
    "java_workspace_symbols",
    "observation_page_",
    "files_",
)


def _normalized_target_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


_SOURCE_EDIT_PATH_KEYS = ("path", "file", "target_path", "target_file")
_SOURCE_CREATE_OPERATIONS = frozenset(
    {
        "create",
        "create_file",
        "create_java_type",
        "create_class",
        "create_type",
        "write",
        "write_file",
    }
)


def _canonical_mutation_path(value: Any) -> str:
    clean = _normalized_target_path(value)
    while clean.startswith("./"):
        clean = clean[2:]
    return clean


def _mutation_target_error(
    tool_name: str,
    arguments: Mapping[str, Any],
    context: TargetMutationContext | None,
) -> str | None:
    """Reject source edits that escape the repository-localized target."""

    if tool_name != "apply_source_edit":
        return None
    if context is None or not context.is_mutation_ready:
        return (
            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "
            "repository-localized target context."
        )

    pinned = _canonical_mutation_path(context.target_path)
    supplied = ""
    for key in _SOURCE_EDIT_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            supplied = _canonical_mutation_path(value)
            break

    if not pinned or not supplied:
        return (
            "MUTATION_TARGET_UNBOUND: apply_source_edit requires the pinned target "
            "path in its model payload."
        )
    if supplied != pinned:
        return (
            f"MUTATION_TARGET_DRIFT: pinned target {pinned!r} but "
            f"apply_source_edit requested {supplied!r}."
        )

    operation = str(arguments.get("operation", "")).strip().casefold()
    if not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:
        return (
            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "
            f"{pinned!r} cannot be recreated by {operation!r}."
        )
    return None


def _proves_existing_target(context: Any) -> bool:
    if bool(getattr(context, "is_new_file", False)):
        return False
    if not _normalized_target_path(getattr(context, "target_path", None)):
        return False
    if getattr(context, "source_body", None):
        return True
    source = str(getattr(context, "evidence_source", "") or "")
    return source.startswith(_EXISTING_TARGET_EVIDENCE_PREFIXES)


class LocalizationStage(str, Enum):
    NEED_FILE = "NEED_FILE"
    NEED_SYMBOL = "NEED_SYMBOL"
    NEED_BODY = "NEED_BODY"
    READY = "READY"


@dataclass(frozen=True)
class TargetMutationContext:
    """Target-bound mutation context enforcing hierarchical localization before ACT phase entry.

    Adheres to Agentless (file -> symbol -> edit location -> concrete source span)
    and repository-scale function-level repair findings.
    """

    target_path: str | None = None
    target_symbol: str | None = None
    source_body: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    is_new_file: bool = False
    evidence_source: str | None = None
    base_revision_sha: str | None = None

    @property
    def is_mutation_ready(self) -> bool:
        """Return True only when the target is localized and bound to concrete source code/spans."""
        return self.localization_stage == LocalizationStage.READY

    @property
    def localization_stage(self) -> LocalizationStage:
        """Calculate the current hierarchical localization stage."""
        if self.is_new_file:
            return (
                LocalizationStage.READY
                if (self.target_path and self.target_path.strip())
                else LocalizationStage.NEED_FILE
            )
        if not self.target_path or not self.target_path.strip():
            return LocalizationStage.NEED_FILE
        if not self.source_body or not _is_code_bearing_text(self.source_body):
            if not self.target_symbol and self.start_line is None:
                return LocalizationStage.NEED_SYMBOL
            return LocalizationStage.NEED_BODY
        return LocalizationStage.READY

    def merge(self, other: TargetMutationContext) -> TargetMutationContext:
        """Merge localization evidence without crossing target-file identity."""

        self_path = _normalized_target_path(self.target_path)
        other_path = _normalized_target_path(other.target_path)
        if self_path and other_path and self_path != other_path:
            return other

        merged = TargetMutationContext(
            target_path=other.target_path or self.target_path,
            target_symbol=other.target_symbol or self.target_symbol,
            source_body=other.source_body or self.source_body,
            start_line=other.start_line if other.start_line is not None else self.start_line,
            end_line=other.end_line if other.end_line is not None else self.end_line,
            is_new_file=other.is_new_file or self.is_new_file,
            evidence_source=other.evidence_source or self.evidence_source,
            base_revision_sha=other.base_revision_sha or self.base_revision_sha,
        )
        if other.is_new_file:
            return merged
        if merged.is_new_file and _proves_existing_target(other):
            return replace(merged, is_new_file=False)
        return merged


@dataclass(frozen=True)
class ExecutionStepTrace:
    """Immutable per-turn trajectory record inspired by SWE-agent (.traj) and OpenHands."""

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
        return {
            "step_index": self.step_index,
            "phase_before": self.phase_before,
            "localization_stage_before": self.localization_stage_before,
            "mutation_context_before": self.mutation_context_before,
            "exposed_tools": self.exposed_tools,
            "tool_choice": self.tool_choice,
            "input_messages_count": self.input_messages_count,
            "model_response_content": self.model_response_content,
            "model_tool_calls": self.model_tool_calls,
            "query_signatures": self.query_signatures,
            "tool_results": self.tool_results,
            "mutation_context_after": self.mutation_context_after,
            "localization_stage_after": self.localization_stage_after,
            "phase_after": self.phase_after,
            "turn_made_progress": self.turn_made_progress,
            "no_progress_streak_after": self.no_progress_streak_after,
            "action_decision": self.action_decision,
            "timestamp": self.timestamp,
        }


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
    }


def format_trajectory_summary(trajectory: Sequence[ExecutionStepTrace]) -> str:
    """Format recent trajectory steps into a compact, human-readable trace."""
    lines = []
    for t in trajectory[-6:]:
        tool_call_desc = (
            ", ".join(
                f"{c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)[:60]})"
                for c in t.model_tool_calls
            )
            or (t.model_response_content[:50] if t.model_response_content else "<empty>")
        )
        results_desc = ", ".join(
            f"{r.get('name')}->{'OK' if r.get('ok') else str(r.get('error', 'FAIL'))[:40]}"
            for r in t.tool_results
        ) or "<none>"
        lines.append(
            f"  Step {t.step_index} [{t.phase_before}:{t.localization_stage_before} -> {t.phase_after}:{t.localization_stage_after}]: "
            f"exposed={t.exposed_tools} | model={tool_call_desc} | res={results_desc} | progress={t.turn_made_progress} streak={t.no_progress_streak_after}"
        )
    return "\n".join(lines)


@dataclass
class HostRunState:
    """Unified single-owner state tracking execution, progress deltas, and bounds."""

    phase: LoopPhase = LoopPhase.OBSERVE
    step_index: int = 0
    no_progress_streak: int = 0
    attempted_queries: set[str] = field(default_factory=set)
    attempted_sources: set[str] = field(default_factory=set)
    localization_attempted_sources: set[str] = field(default_factory=set)
    evidence_fingerprints: set[str] = field(default_factory=set)
    mutation_context: TargetMutationContext | None = None
    applied_mutations: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    validation_status: str = "PENDING"
    last_failure_digest: str | None = None
    last_failure_reason: str | None = None
    last_result_digest: str | None = None
    termination_reason: str | None = None
    trajectory: list[ExecutionStepTrace] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def has_fresh_evidence(self) -> bool:
        with self._lock:
            return bool(self.evidence_fingerprints)

    def is_query_attempted(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        sig = retrieval_query_signature(tool_name, arguments)
        with self._lock:
            return sig in self.attempted_queries

    def record_attempted_source(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        localization_stage: LocalizationStage | None = None,
    ) -> None:
        source = retrieval_source_key(tool_name, arguments)
        with self._lock:
            self.attempted_sources.add(source)
            if localization_stage is not None:
                self.localization_attempted_sources.add(
                    f"{localization_stage.value}:{source}"
                )

    def attempted_sources_for_localization_stage(
        self,
        localization_stage: LocalizationStage,
    ) -> frozenset[str]:
        prefix = f"{localization_stage.value}:"
        with self._lock:
            return frozenset(
                item[len(prefix):]
                for item in self.localization_attempted_sources
                if item.startswith(prefix)
            )

    def record_query(self, tool_name: str, arguments: Mapping[str, Any]) -> bool:
        source = retrieval_source_key(tool_name, arguments)
        sig = retrieval_query_signature(tool_name, arguments)
        with self._lock:
            if sig in self.attempted_queries:
                return False
            self.attempted_queries.add(sig)
            self.attempted_sources.add(source)
            return True

    def record_evidence(self, value: Any, *, usable: bool) -> bool:
        if not usable:
            return False
        fp = evidence_fingerprint(value)
        if fp is None:
            return False
        with self._lock:
            extracted_ctx = _extract_mutation_context_from_payload(value)
            if extracted_ctx is not None:
                if self.mutation_context is None:
                    self.mutation_context = extracted_ctx
                else:
                    self.mutation_context = self.mutation_context.merge(extracted_ctx)
            if fp in self.evidence_fingerprints:
                return False
            self.evidence_fingerprints.add(fp)
            return True

    def record_mutation(self, tool_name: str, payload: Mapping[str, Any]) -> bool:
        applied = mutation_payload_applied(tool_name, payload)
        if applied:
            with self._lock:
                self.applied_mutations.append(tool_name)
                self.workspace_changed = True
            return True
        return False

    def record_failure(self, tool_name: str, error: Any) -> None:
        """Record failure diagnostics without treating a new error string as progress."""

        reason = f"{str(tool_name).strip()}: {str(error).strip()}"
        digest = hashlib.sha256(reason.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            self.last_failure_reason = reason
            self.last_failure_digest = digest

    def clear_failure(self) -> None:
        with self._lock:
            self.last_failure_reason = None
            self.last_failure_digest = None

    def record_no_progress_result(self, value: Any) -> int:
        """Count only a repeated stable state/action/result as a fixed point.

        Two different recoverable failures are not convergence.  In particular, an
        authority rejection followed by a phase correction must leave a small model one
        more turn to obey the corrected contract.  The global tool-round limit still
        bounds alternating or otherwise non-convergent behavior.
        """

        stable = _stable_value(value, drop_volatile=True)
        canonical = json.dumps(
            stable,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            if digest == self.last_result_digest:
                self.no_progress_streak += 1
            else:
                self.last_result_digest = digest
                self.no_progress_streak = 1
            return self.no_progress_streak

    def clear_no_progress_result(self) -> None:
        with self._lock:
            self.last_result_digest = None
            self.no_progress_streak = 0

    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
        localization_stage: LocalizationStage | None = None,
    ) -> str | None:
        exposed = set(exposed_tools)
        attempted = (
            self.attempted_sources_for_localization_stage(localization_stage)
            if localization_stage is not None
            else frozenset(self.attempted_sources)
        )
        for name in preferred:
            if name in exposed and name not in attempted:
                return name
        return None


class RetrievalNoProgressError(ModelConfigurationError):
    pass


class RetrievalProgress:
    """Compatibility interface delegating to HostRunState."""

    def __init__(self, state: HostRunState | None = None, *, no_progress_limit: int | None = None) -> None:
        self._state = state or HostRunState()
        self.attempted_queries = self._state.attempted_queries
        self.attempted_sources = self._state.attempted_sources
        self.evidence_fingerprints = self._state.evidence_fingerprints
        self._lock = self._state._lock
        self._no_progress_limit = no_progress_limit
        self.no_progress_observations = 0

    def begin(self, tool_name: str, arguments: Mapping[str, Any]) -> RetrievalDecision:
        if self._state.record_query(tool_name, arguments):
            return RetrievalDecision.EXECUTE
        return RetrievalDecision.DUPLICATE_QUERY

    def observe(
        self,
        *args: Any,
        usable: bool = True,
        **kwargs: Any,
    ) -> RetrievalObservation:
        if len(args) >= 3:
            value = args[2]
        elif len(args) >= 1:
            value = args[0]
        else:
            value = kwargs.get("value")

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


_CODE_MARKERS = frozenset({
    "class ", "interface ", "enum ", "record ", "public ", "private ", "protected ",
    "package ", "import ", "void ", "return ", "final ", "static ", "new ",
    "extends ", "implements ", "override", "{", "}", ";", "(", ")",
})


def _is_code_bearing_text(text: Any) -> bool:
    """Return whether a text snippet contains actual Java/source code structure."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 15:
        return False
    lower = stripped.casefold()
    return any(marker in lower for marker in _CODE_MARKERS)


def _is_new_file_creation(payload: Any) -> bool:
    """Return whether the prompt is an explicit creation of a new target file."""
    if isinstance(payload, Mapping):
        op = str(payload.get("operation", "")).strip().casefold()
        if op in ("create_file", "new_file"):
            return bool(payload.get("path") or payload.get("target_path"))
        target = payload.get("target_file")
        if isinstance(target, Mapping) and target.get("create") is True:
            return bool(target.get("path"))
    return False


def _sequence_has_values(value: Any) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and value
    )


def _evidence_task_from_module(module: Any) -> Mapping[str, Any] | None:
    """Read the canonical task-local envelope and its legacy compatibility shape."""

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


def _fresh_target_has_reuse_evidence(payload: Any) -> bool:
    """Fail closed when a nominally fresh binding also carries reuse evidence."""

    if not isinstance(payload, Mapping):
        return False
    module = payload.get("module")
    task = _evidence_task_from_module(module)
    if task is None:
        return False
    bindings = task.get("production_bindings")
    if not isinstance(bindings, Sequence) or isinstance(
        bindings, (str, bytes, bytearray)
    ):
        return False
    production_bindings = tuple(
        item for item in bindings if isinstance(item, Mapping)
    )
    actions = {
        str(item.get("reuse_action") or "").strip().casefold()
        for item in production_bindings
        if str(item.get("reuse_action") or "").strip()
    }
    if actions != {"fresh"}:
        return False

    if any(
        _sequence_has_values(task.get(key))
        for key in ("reuse_refs", "component_refs", "source_refs")
    ):
        return True
    return any(
        _sequence_has_values(binding.get(key))
        for binding in production_bindings
        for key in ("reuse_refs", "component_refs", "source_refs")
    )


def _fresh_owned_symbol_context(payload: Any) -> TargetMutationContext | None:
    """Resolve a host-reserved fresh target without guessing a repository file."""

    if not isinstance(payload, Mapping):
        return None
    module = payload.get("module")
    task = _evidence_task_from_module(module)
    if task is None:
        return None
    bindings = task.get("production_bindings")
    if not isinstance(bindings, Sequence) or isinstance(
        bindings, (str, bytes, bytearray)
    ):
        return None

    production_bindings = tuple(
        item for item in bindings if isinstance(item, Mapping)
    )
    if not production_bindings:
        return None
    actions = {
        str(item.get("reuse_action") or "").strip().casefold()
        for item in production_bindings
        if str(item.get("reuse_action") or "").strip()
    }
    if actions != {"fresh"}:
        return None

    for binding in production_bindings:
        anchors = binding.get("owned_anchors")
        if not isinstance(anchors, Sequence) or isinstance(
            anchors, (str, bytes, bytearray)
        ):
            continue
        for anchor in anchors:
            if (
                not isinstance(anchor, Mapping)
                or str(anchor.get("kind") or "") != "symbol"
            ):
                continue
            locator = _normalized_target_path(anchor.get("locator"))
            target_path, separator, target_symbol = locator.partition("#")
            target_path = target_path.strip()
            if (
                not target_path
                or target_path.startswith("/")
                or ".." in target_path.split("/")
                or not _is_workspace_file_path(target_path)
            ):
                continue
            return TargetMutationContext(
                target_path=target_path,
                target_symbol=(
                    target_symbol.strip()
                    if separator and target_symbol.strip()
                    else None
                ),
                is_new_file=True,
                evidence_source="evidence_fresh_owned_anchor",
            )
    return None


def _initial_exact_target_context(
    payload: Any,
    *,
    prospective: TargetMutationContext,
) -> TargetMutationContext | None:
    """Reuse bounded initial source only when it proves the reserved target path."""

    if not isinstance(payload, Mapping):
        return None
    initial = payload.get("initial_exact_source_context")
    if not isinstance(initial, (Mapping, list, tuple)):
        return None
    target_path = _normalized_target_path(prospective.target_path)
    if not target_path:
        return None

    candidates: list[Any] = []
    if isinstance(initial, Mapping):
        for key in (
            "global_anchors",
            "records",
            "excerpts",
            "files",
            "sources",
            "hits",
            "results",
        ):
            value = initial.get(key)
            if isinstance(value, Mapping):
                candidates.extend(
                    {"path": str(path), "content": content}
                    for path, content in value.items()
                )
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                candidates.extend(value)
        candidates.append(initial)
    else:
        candidates.extend(initial)

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_path = _normalized_target_path(
            candidate.get("source_path")
            or candidate.get("path")
            or candidate.get("file")
            or candidate.get("uri")
        )
        candidate_path = candidate_path.removeprefix("file:///").removeprefix(
            "file://"
        )
        if candidate_path != target_path:
            continue
        existing = _extract_mutation_context_from_payload({"hits": [candidate]})
        if (
            existing is None
            or _normalized_target_path(existing.target_path) != target_path
        ):
            existing = TargetMutationContext(
                target_path=target_path,
                target_symbol=prospective.target_symbol,
                is_new_file=False,
                evidence_source="initial_exact_target_path",
            )
        elif prospective.target_symbol and not existing.target_symbol:
            existing = replace(
                existing,
                target_symbol=prospective.target_symbol,
            )
        return replace(existing, is_new_file=False)
    return None


def _extract_mutation_context_from_payload(payload: Any) -> TargetMutationContext | None:
    """Extract a target-bound MutationContext from a message payload or tool return.

    Implements hierarchical localization: File -> Symbol -> Edit location -> Bound source body.
    """
    if not isinstance(payload, Mapping):
        if isinstance(payload, (list, tuple)):
            for item in payload:
                ctx = _extract_mutation_context_from_payload(item)
                if ctx is not None and ctx.is_mutation_ready:
                    return ctx
            for item in payload:
                ctx = _extract_mutation_context_from_payload(item)
                if ctx is not None:
                    return ctx
        return None

    if _fresh_target_has_reuse_evidence(payload):
        return TargetMutationContext(
            evidence_source="reuse_evidence_requires_localization"
        )

    fresh = _fresh_owned_symbol_context(payload)
    if fresh is not None:
        existing = _initial_exact_target_context(payload, prospective=fresh)
        return existing or fresh

    for wrapper_key in ("structured_content", "result", "data", "body", "_mmm_observation", "raw_result", "structured", "observation"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, (Mapping, list, tuple)) and wrapped is not payload:
            ctx = _extract_mutation_context_from_payload(wrapped)
            if ctx is not None:
                return ctx

    if _is_new_file_creation(payload):
        target_path = str(payload.get("path") or payload.get("target_path") or "")
        if not target_path and isinstance(payload.get("target_file"), Mapping):
            target_path = str(payload["target_file"].get("path", ""))
        if target_path:
            return TargetMutationContext(
                target_path=target_path,
                is_new_file=True,
                evidence_source="new_file_spec",
            )

    hits = payload.get("hits") or payload.get("results")
    if isinstance(hits, (list, tuple)):
        for hit in hits:
            if isinstance(hit, Mapping):
                meta = hit.get("metadata") if isinstance(hit.get("metadata"), Mapping) else {}
                path = str(
                    hit.get("source_path")
                    or hit.get("path")
                    or hit.get("file")
                    or hit.get("uri")
                    or meta.get("path")
                    or meta.get("source_path")
                    or ""
                ).strip()
                if path and _is_workspace_file_path(path):
                    snippet = (
                        hit.get("text")
                        or hit.get("snippet")
                        or hit.get("code")
                        or hit.get("content")
                        or hit.get("source")
                    )
                    if isinstance(snippet, (list, tuple)):
                        snippet = "\n".join(str(s) for s in snippet)
                    if isinstance(snippet, str) and _is_code_bearing_text(snippet):
                        symbol = (
                            str(
                                hit.get("symbol")
                                or hit.get("function")
                                or hit.get("name")
                                or meta.get("symbol")
                                or ""
                            ).strip()
                            or None
                        )
                        start_line = hit.get("start_line") or hit.get("line")
                        end_line = hit.get("end_line")
                        return TargetMutationContext(
                            target_path=path,
                            target_symbol=symbol,
                            source_body=snippet,
                            start_line=int(start_line) if isinstance(start_line, int) else None,
                            end_line=int(end_line) if isinstance(end_line, int) else None,
                            evidence_source="search_code_rag",
                        )
        for hit in hits:
            if isinstance(hit, Mapping):
                meta = hit.get("metadata") if isinstance(hit.get("metadata"), Mapping) else {}
                path = str(
                    hit.get("source_path")
                    or hit.get("path")
                    or hit.get("file")
                    or hit.get("uri")
                    or meta.get("path")
                    or meta.get("source_path")
                    or ""
                ).strip()
                if path and _is_workspace_file_path(path):
                    return TargetMutationContext(
                        target_path=path,
                        source_body=None,
                        evidence_source="search_code_rag_path_only",
                    )

    sources = payload.get("sources")
    if isinstance(sources, (list, tuple)):
        for src in sources:
            if isinstance(src, Mapping):
                path = str(src.get("path") or src.get("file") or src.get("source_id") or src.get("url") or "").strip()
                content = src.get("content") or src.get("snippet") or src.get("code") or src.get("text") or src.get("summary")
                if path and _is_workspace_file_path(path) and isinstance(content, str) and _is_code_bearing_text(content):
                    return TargetMutationContext(
                        target_path=path,
                        source_body=content,
                        evidence_source="sources_code",
                    )
        for src in sources:
            if isinstance(src, Mapping):
                path = str(src.get("path") or src.get("file") or src.get("source_id") or "").strip()
                if path and _is_workspace_file_path(path):
                    return TargetMutationContext(
                        target_path=path,
                        source_body=None,
                        evidence_source="sources_path_only",
                    )

    symbols = payload.get("symbols")
    if isinstance(symbols, (list, tuple)) and len(symbols) > 0:
        for sym in symbols:
            if isinstance(sym, Mapping):
                name = str(sym.get("name", "")).strip()
                container = str(sym.get("containerName", "")).strip()
                symbol_id = f"{container}#{name}" if container and name else (name or None)
                location = sym.get("location")
                path = ""
                start_line = None
                end_line = None
                if isinstance(location, Mapping):
                    uri = str(location.get("uri", ""))
                    path = uri.replace("file:///", "").replace("file://", "")
                    range_info = location.get("range")
                    if isinstance(range_info, Mapping):
                        start = range_info.get("start")
                        end = range_info.get("end")
                        if isinstance(start, Mapping):
                            start_line = start.get("line")
                        if isinstance(end, Mapping):
                            end_line = end.get("line")
                if path or symbol_id:
                    return TargetMutationContext(
                        target_path=path,
                        target_symbol=symbol_id,
                        source_body=None,
                        start_line=int(start_line) if isinstance(start_line, int) else None,
                        end_line=int(end_line) if isinstance(end_line, int) else None,
                        evidence_source="java_workspace_symbols",
                    )

    for key in ("global_anchors", "page_observations", "records", "excerpts"):
        records = payload.get(key)
        if isinstance(records, (list, tuple)):
            for rec in records:
                if isinstance(rec, Mapping):
                    path = str(rec.get("path") or rec.get("file") or "").strip()
                    text = (
                        rec.get("text")
                        or rec.get("content")
                        or rec.get("source")
                        or rec.get("snippet")
                        or rec.get("code")
                    )
                    if path and isinstance(text, str) and _is_code_bearing_text(text):
                        return TargetMutationContext(
                            target_path=path,
                            source_body=text,
                            evidence_source=f"observation_page_{key}",
                        )

    files = payload.get("files")
    if isinstance(files, Mapping):
        for path, content in files.items():
            if _is_code_bearing_text(str(content)):
                return TargetMutationContext(
                    target_path=str(path),
                    source_body=str(content),
                    evidence_source="files_map",
                )
    elif isinstance(files, (list, tuple)):
        for item in files:
            if isinstance(item, Mapping):
                path = str(item.get("path") or item.get("file") or "").strip()
                content = item.get("content") or item.get("source") or item.get("code")
                if path and isinstance(content, str) and _is_code_bearing_text(content):
                    return TargetMutationContext(
                        target_path=path,
                        source_body=content,
                        evidence_source="files_list",
                    )
        for item in files:
            if isinstance(item, str) and item.strip():
                return TargetMutationContext(
                    target_path=item.strip(),
                    source_body=None,
                    evidence_source="files_names_only",
                )

    exact = payload.get("initial_exact_source_context")
    if exact is not None and exact is not payload:
        ctx = _extract_mutation_context_from_payload(exact)
        if ctx is not None:
            return ctx

    target_file = payload.get("target_file")
    if isinstance(target_file, str) and target_file.strip():
        source_val = payload.get("source") or payload.get("content") or payload.get("code")
        source_body = str(source_val) if isinstance(source_val, str) and _is_code_bearing_text(source_val) else None
        return TargetMutationContext(
            target_path=target_file.strip(),
            source_body=source_body,
            evidence_source="target_file_field",
        )

    return None


def is_mutation_ready(
    messages: Sequence[Mapping[str, Any]],
    state: HostRunState,
) -> bool:
    """Return true only when the target is localized and bound to concrete source context."""
    with state._lock:
        if state.mutation_context is not None and state.mutation_context.is_mutation_ready:
            return True

    for message in messages:
        content = message.get("content")
        payload: Any = None
        if isinstance(content, Mapping):
            payload = content
        elif isinstance(content, str) and content.lstrip().startswith("{"):
            try:
                payload = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                continue
        if payload is not None:
            ctx = _extract_mutation_context_from_payload(payload)
            if ctx is not None and ctx.is_mutation_ready:
                with state._lock:
                    if state.mutation_context is None or not state.mutation_context.is_mutation_ready:
                        state.mutation_context = ctx
                return True

    return False


def _tool_name(schema: Mapping[str, Any]) -> str:
    fn = schema.get("function")
    return str(fn.get("name", "")).strip() if isinstance(fn, Mapping) else ""


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
    if not by_name:
        return ()

    if phase == LoopPhase.OBSERVE:
        if localization_active is None:
            localization_active = mutation_context is not None
        if not localization_active:
            selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
            if (
                mutation_context is None
                and "search_code_rag" in selected_names
                and "java_workspace_symbols" in selected_names
            ):
                selected_names.remove("java_workspace_symbols")
        else:
            stage = (
                mutation_context.localization_stage
                if mutation_context is not None
                else LocalizationStage.NEED_FILE
            )
            if stage == LocalizationStage.NEED_FILE:
                preferred = ("search_code_rag", "search_project_rag", "inspect_existing_mod")
                untried = [name for name in preferred if name in by_name and name not in attempted_sources]
                if semantic_retrieval_choice:
                    semantic = [name for name in untried if name in {"search_code_rag", "search_project_rag"}]
                    selected_names = semantic or ([untried[0]] if untried else [])
                else:
                    selected_names = [untried[0]] if untried else []
            elif stage == LocalizationStage.NEED_SYMBOL:
                preferred = ("java_workspace_symbols", "search_code_rag", "search_project_rag")
                untried = [name for name in preferred if name in by_name and name not in attempted_sources]
                selected_names = [untried[0]] if untried else []
            elif stage == LocalizationStage.NEED_BODY:
                preferred = ("search_code_rag", "inspect_existing_mod")
                untried = [name for name in preferred if name in by_name and name not in attempted_sources]
                selected_names = [untried[0]] if untried else []
            else:
                selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
    elif phase == LoopPhase.ACT:
        # apply_source_edit is the canonical model-facing source mutation surface.
        # Prefer it whenever available so alternate mutators cannot bypass the
        # repository-localized target binding. Legacy fallbacks remain available
        # only when the canonical tool is genuinely absent.
        if "apply_source_edit" in by_name:
            selected_names = ["apply_source_edit"]
        else:
            selected_names = [name for name in by_name if name in _MUTATION_ACT_TOOLS]
    elif phase in (LoopPhase.VERIFY, LoopPhase.RECOVER):
        selected_names = [
            name for name in by_name
            if name in _VERIFY_TOOLS or name in _READ_OBSERVE_TOOLS
        ]
    else:
        selected_names = []

    return tuple(by_name[name] for name in selected_names if name in by_name)


def _replace_live_messages(
    messages: list[dict[str, Any]],
    fitted: tuple[Mapping[str, Any], ...],
) -> bool:
    replacement = [dict(message) for message in fitted]
    if replacement == messages:
        return False
    messages[:] = replacement
    return True


def _atomic_output_recovery_instruction(request: GenerationRequest) -> str:
    names = frozenset(_tool_name(schema) for schema in request.tools if _tool_name(schema))
    if names & _MUTATION_ACT_TOOLS:
        return (
            "The preceding assistant action exceeded the bounded output allowance and is discarded. "
            "Do not continue, reproduce, or complete that oversized payload. Call exactly one visible "
            "source-mutation tool now with one small semantic edit and no prose. For a new Java file, "
            "the first action must be create_java_type with only package_name and an empty type "
            "declaration; never create a complete Java file with create_file. After each tool "
            "observation, add at most one import with add_java_import or one field/constructor/method/"
            "nested declaration with insert_java_member. For an existing file, use one bounded "
            "replace_exact/insert action. The host will preserve the same mutation target and workspace "
            "state between actions."
        )
    return (
        "The preceding assistant action exceeded the bounded output allowance and is discarded. "
        "Do not continue that oversized payload. Produce exactly one concise visible tool call or one "
        "concise final answer using the already-grounded state; do not emit a long reconstruction."
    )


def _retry_atomic_after_output_exhaustion(
    router: Any,
    *,
    config: Any,
    adapter: Any,
    request: GenerationRequest,
    messages: list[dict[str, Any]],
    media_paths: tuple[Any, ...],
) -> Any:
    """Regenerate one bounded action without leaving the current HostRunState."""

    messages.append({"role": "system", "content": _atomic_output_recovery_instruction(request)})
    retry_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
        parallel_tool_calls=False if request.tools else request.parallel_tool_calls,
    )
    print(
        "agent output: atomic-action recovery",
        f"tools={sorted(_tool_name(schema) for schema in request.tools if _tool_name(schema))}",
        flush=True,
    )
    try:
        with router._generation_scope(config):
            return adapter.generate_turn(retry_request)
    except BaseException as retry_exc:
        if completion_boundary_kind(retry_exc) == OUTPUT_EXHAUSTED:
            raise ModelConfigurationError(
                "ATOMIC_ACTION_OUTPUT_STALLED: the model exceeded the output allowance twice "
                "without completing one bounded semantic action; refusing to reset the agent state "
                "or restart an oversized action."
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
    """Select the largest deterministic retry that leaves useful live output space."""

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
    """Fit one live agent turn and recover bounded completion failures in-place."""

    turn_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
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
    if callable(exact_accounting):
        accounting = exact_accounting(turn_request)
        if accounting.input_tokens < accounting.context_tokens:
            fitted = tuple(messages)
        else:
            fitted = fit_messages_to_context(messages, config=config, tools=request.tools)
    else:
        fitted = fit_messages_to_context(messages, config=config, tools=request.tools)
    _replace_live_messages(messages, fitted)
    turn_request = replace(turn_request, messages=tuple(messages))
    try:
        with router._generation_scope(config):
            return adapter.generate_turn(turn_request)
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
            emergency = emergency_fit_messages(
                messages,
                budget_bytes=emergency_budget,
            )
            recovery_receipt = {"budget_bytes": emergency_budget}
        if not _replace_live_messages(messages, emergency):
            mark_context_recovery_exhausted(exc)
            raise
        retry_request = replace(
            turn_request,
            messages=tuple(messages),
            media_paths=media_paths,
        )
        print(
            "agent context: deterministic overflow recovery",
            f"messages={len(messages)}",
            *(f"{key}={value}" for key, value in recovery_receipt.items()),
            flush=True,
        )
        try:
            with router._generation_scope(config):
                return adapter.generate_turn(retry_request)
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
    """Unified single-owner retrieve/act/observe/verify execution loop."""

    from .agent_capability_context import (
        reviewed_mcp_servers_for_model_role,
        skills_for_tool,
    )
    from .grounding_policy import host_baseline_evidence_ready
    from .model_router import (
        _RAG_EVIDENCE_TOOLS,
        _agent_tool_round_limit,
        _execute_tool_waves,
        _external_rag_capability,
        _tool_schema_names,
        _usable_external_rag_result,
        _usable_rag_result,
    )

    messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
    all_exposed_tools = tuple(request.tools)
    all_exposed_names = frozenset(_tool_schema_names(all_exposed_tools))
    state = HostRunState()
    forced_rag_tool: str | None = None
    forced_rag_attempts = 0
    required_rag_choice = False
    round_limit = _agent_tool_round_limit()
    host_grounded = host_baseline_evidence_ready(request.messages)
    require_rag = bool(
        router._agent_require_fresh_evidence
        and not host_grounded
        and role in {"coder", "coder_safe"}
        and all_exposed_names & _RAG_EVIDENCE_TOOLS
    )
    implementation_requires_mutation = bool(
        role in {"coder", "coder_safe"}
        and stage == "generation"
        and implementation_requested(request.messages)
    )
    reviewed_external_servers = reviewed_mcp_servers_for_model_role(stage, role)

    mutation_ready = is_mutation_ready(request.messages, state)

    if require_rag and not state.has_fresh_evidence:
        state.phase = LoopPhase.OBSERVE
    elif implementation_requires_mutation and not mutation_history_applied(messages) and mutation_ready:
        state.phase = LoopPhase.ACT
    else:
        state.phase = LoopPhase.OBSERVE

    while True:
        state.step_index += 1

        if state.step_index > round_limit:
            if require_rag and not state.has_fresh_evidence:
                raise ModelConfigurationError(
                    "Agent reached the host tool-round limit before required "
                    "evidence became available."
                )
            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder reached the host tool-round limit before a "
                    "reviewed source mutation was applied; refusing a prose-only implementation."
                )
            return _finalize_without_tools(
                router,
                config,
                adapter,
                request,
                messages,
                instruction=(
                    f"The host tool-round limit was reached after {round_limit} rounds. "
                    "Do not call more tools. Return the final answer using only observations already present."
                ),
                empty_error="Agent returned an empty final response at the explicit tool-round limit.",
            )

        if forced_rag_tool is None and state.no_progress_streak >= 2:
            traj_summary = format_trajectory_summary(state.trajectory)
            reason_suffix = f": {state.last_failure_reason}" if state.last_failure_reason else ""
            print(
                f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"[HOST NO-PROGRESS BOUNDARY HIT] Step={state.step_index} Streak={state.no_progress_streak}\n"
                f"Reason: {state.last_failure_reason}\n"
                f"Trajectory:\n{traj_summary}\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
                flush=True,
            )
            if require_rag and not state.has_fresh_evidence:
                raise ModelConfigurationError(
                    "Required production evidence is unavailable: every host-forceable "
                    "RAG source was already attempted without novel usable evidence, and "
                    f"the model selected no other reviewed retrieval route.{reason_suffix}\n"
                    f"Execution trajectory:\n{traj_summary}"
                )
            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder reached a no-progress boundary before a reviewed source "
                    f"mutation was applied; refusing a prose-only implementation.{reason_suffix}\n"
                    f"Execution trajectory:\n{traj_summary}"
                )
            return _finalize_without_tools(
                router,
                config,
                adapter,
                request,
                messages,
                instruction=(
                    "Execution converged to a no-progress fixed point. "
                    "Do not call more tools. Return the final answer from existing observations."
                ),
                empty_error="Agent returned an empty final response after no-progress convergence.",
            )

        phase_before = state.phase.value
        loc_stage_before = (
            state.mutation_context.localization_stage.value
            if state.mutation_context is not None
            else LocalizationStage.NEED_FILE.value
        )
        ctx_before = _mutation_context_dict(state.mutation_context)

        current_localization_stage = (
            state.mutation_context.localization_stage
            if state.mutation_context is not None
            else LocalizationStage.NEED_FILE
        )
        mutation_localization_active = bool(
            implementation_requires_mutation and state.phase == LoopPhase.OBSERVE
        )
        phase_attempted_sources = (
            state.attempted_sources_for_localization_stage(current_localization_stage)
            if mutation_localization_active
            else frozenset()
        )
        phase_tools = _filter_tools_for_phase(
            all_exposed_tools,
            state.phase,
            role,
            mutation_context=(state.mutation_context if mutation_localization_active else None),
            attempted_sources=phase_attempted_sources,
            localization_active=mutation_localization_active,
            semantic_retrieval_choice=bool(require_rag and not state.has_fresh_evidence),
        )
        if required_rag_choice:
            phase_tools = tuple(
                schema for schema in phase_tools
                if _tool_name(schema) in _RAG_EVIDENCE_TOOLS
            )
            if not phase_tools:
                raise ModelConfigurationError(
                    "Required production evidence is unavailable: no reviewed RAG tool remains "
                    "eligible for semantic selection."
                )
        phase_tool_names = frozenset(_tool_name(s) for s in phase_tools if _tool_name(s))
        if implementation_requires_mutation and state.phase == LoopPhase.OBSERVE and not phase_tools:
            state.termination_reason = "MUTATION_LOCALIZATION_STALLED"
            raise ModelConfigurationError(
                "MUTATION_LOCALIZATION_STALLED: no untried reviewed localization source remains "
                f"for stage {current_localization_stage.value}; refusing a repeated retrieval/model round."
            )

        tool_choice = request.tool_choice
        parallel_tool_calls = request.parallel_tool_calls

        if required_rag_choice:
            tool_choice = "required"
            parallel_tool_calls = False
        elif forced_rag_tool is not None:
            tool_choice = {"type": "function", "function": {"name": forced_rag_tool}}
            parallel_tool_calls = False
        elif state.phase == LoopPhase.ACT:
            mutation_names = [n for n in phase_tool_names if n in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel_tool_calls = False

        target_desc = (
            f"path={state.mutation_context.target_path} symbol={state.mutation_context.target_symbol} has_body={bool(state.mutation_context.source_body)}"
            if state.mutation_context
            else "target=None"
        )
        print(
            f"\n[HOST STEP {state.step_index}] role={role} phase={phase_before} stage={loc_stage_before} {target_desc}\n"
            f"  exposed_tools={sorted(phase_tool_names)} tool_choice={tool_choice}",
            flush=True,
        )

        turn_request = replace(
            request,
            tools=phase_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        turn = _generate_turn_with_context_recovery(
            router,
            config=config,
            adapter=adapter,
            request=turn_request,
            messages=messages,
            media_paths=request.media_paths if state.step_index == 1 else (),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        if not turn.tool_calls:
            content = turn.content.strip()
            print(f"  model emitted prose -> {content[:80]}...", flush=True)
            if not content:
                raise ModelConfigurationError("Tool-capable model returned an empty final response.")
            if required_rag_choice:
                raise ModelConfigurationError(
                    "Production coder did not honor the host-required evidence invariant: "
                    "one reviewed RAG tool call was required, but the model returned prose."
                )
            if forced_rag_tool is not None:
                forced_rag_attempts += 1
                if forced_rag_attempts >= 1:
                    raise ModelConfigurationError(
                        f"Production coder did not honor host-forced RAG tool choice {forced_rag_tool!r} "
                        "after one bounded forced attempt."
                    )
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": f"Call the required function {forced_rag_tool} exactly once now. Do not answer in prose.",
                    },
                ])
                trace_entry = ExecutionStepTrace(
                    step_index=state.step_index,
                    phase_before=phase_before,
                    localization_stage_before=loc_stage_before,
                    mutation_context_before=ctx_before,
                    exposed_tools=sorted(phase_tool_names),
                    tool_choice=tool_choice,
                    input_messages_count=len(messages),
                    model_response_content=content,
                    model_tool_calls=[],
                    query_signatures=[],
                    tool_results=[],
                    mutation_context_after=ctx_before,
                    localization_stage_after=loc_stage_before,
                    phase_after=state.phase.value,
                    turn_made_progress=False,
                    no_progress_streak_after=state.no_progress_streak,
                    action_decision=f"host_guided_to_{forced_rag_tool}",
                )
                state.trajectory.append(trace_entry)
                continue
            if require_rag and not state.has_fresh_evidence:
                eligible_rag_names = tuple(
                    sorted(
                        name
                        for name in phase_tool_names
                        if name in _RAG_EVIDENCE_TOOLS and name not in state.attempted_sources
                    )
                )
                if not eligible_rag_names:
                    raise ModelConfigurationError(
                        "Required production evidence is unavailable: every reviewed RAG route "
                        "was exhausted without novel usable evidence."
                    )
                if len(eligible_rag_names) == 1:
                    forced_rag_tool = eligible_rag_names[0]
                    forced_rag_attempts = 0
                    guidance = (
                        f"Baseline production evidence is still required. Call {forced_rag_tool} "
                        "exactly once with a concrete query for the current implementation need."
                    )
                    action_decision = f"host_required_rag_{forced_rag_tool}"
                else:
                    required_rag_choice = True
                    guidance = (
                        "Baseline production evidence is still required. Select exactly one currently "
                        "exposed reviewed RAG function that best matches the information need and call "
                        "it with a concrete query. The host requires evidence but does not choose the "
                        "semantic retrieval route for you."
                    )
                    action_decision = "host_required_rag_semantic_choice"
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "system", "content": guidance},
                ])
                trace_entry = ExecutionStepTrace(
                    step_index=state.step_index,
                    phase_before=phase_before,
                    localization_stage_before=loc_stage_before,
                    mutation_context_before=ctx_before,
                    exposed_tools=sorted(phase_tool_names),
                    tool_choice=tool_choice,
                    input_messages_count=len(messages),
                    model_response_content=content,
                    model_tool_calls=[],
                    query_signatures=[],
                    tool_results=[],
                    mutation_context_after=ctx_before,
                    localization_stage_after=loc_stage_before,
                    phase_after=state.phase.value,
                    turn_made_progress=False,
                    no_progress_streak_after=state.no_progress_streak,
                    action_decision=action_decision,
                )
                state.trajectory.append(trace_entry)
                continue
            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):
                if state.phase == LoopPhase.OBSERVE and is_mutation_ready(messages, state):
                    state.phase = LoopPhase.ACT
                    messages.extend([
                        {"role": "assistant", "content": content},
                        {
                            "role": "system",
                            "content": (
                                "Target source context is grounded. Do not finalize in prose: "
                                "proceed to the source-mutation action for the implementation now."
                            ),
                        },
                    ])
                    state.no_progress_streak = 0
                    trace_entry = ExecutionStepTrace(
                        step_index=state.step_index,
                        phase_before=phase_before,
                        localization_stage_before=loc_stage_before,
                        mutation_context_before=ctx_before,
                        exposed_tools=sorted(phase_tool_names),
                        tool_choice=tool_choice,
                        input_messages_count=len(messages),
                        model_response_content=content,
                        model_tool_calls=[],
                        query_signatures=[],
                        tool_results=[],
                        mutation_context_after=ctx_before,
                        localization_stage_after=loc_stage_before,
                        phase_after=LoopPhase.ACT.value,
                        turn_made_progress=True,
                        no_progress_streak_after=0,
                        action_decision="transition_to_act",
                    )
                    state.trajectory.append(trace_entry)
                    continue
                elif state.phase == LoopPhase.OBSERVE:
                    loc_stage = (
                        state.mutation_context.localization_stage
                        if state.mutation_context is not None
                        else LocalizationStage.NEED_FILE
                    )
                    if loc_stage == LocalizationStage.NEED_FILE:
                        preferred = ("search_code_rag", "search_project_rag")
                        prompt_msg = "Locate the target file path using {tool} before modifying source."
                    elif loc_stage == LocalizationStage.NEED_SYMBOL:
                        preferred = ("java_workspace_symbols",)
                        target_p = state.mutation_context.target_path or "the target file"
                        prompt_msg = f"Target file '{target_p}' identified. Call {{tool}} to inspect symbol declarations."
                    elif loc_stage == LocalizationStage.NEED_BODY:
                        preferred = ("search_code_rag", "inspect_existing_mod")
                        target_s = (
                            state.mutation_context.target_symbol
                            or state.mutation_context.target_path
                            or "the target symbol"
                        )
                        prompt_msg = f"Target symbol '{target_s}' located. Call {{tool}} to retrieve the concrete function/method source body."
                    else:
                        preferred = ("search_code_rag", "java_workspace_symbols", "search_project_rag")
                        prompt_msg = "Call {tool} to inspect the target file or symbol before modifying source."

                    forced_tool = state.next_untried_internal_tool(
                        all_exposed_names,
                        preferred=preferred,
                        localization_stage=loc_stage,
                    )
                    if forced_tool is not None:
                        forced_rag_tool = forced_tool
                        forced_rag_attempts = 0
                        messages.extend([
                            {"role": "assistant", "content": content},
                            {
                                "role": "system",
                                "content": prompt_msg.format(tool=forced_rag_tool),
                            },
                        ])
                        trace_entry = ExecutionStepTrace(
                            step_index=state.step_index,
                            phase_before=phase_before,
                            localization_stage_before=loc_stage_before,
                            mutation_context_before=ctx_before,
                            exposed_tools=sorted(phase_tool_names),
                            tool_choice=tool_choice,
                            input_messages_count=len(messages),
                            model_response_content=content,
                            model_tool_calls=[],
                            query_signatures=[],
                            tool_results=[],
                            mutation_context_after=ctx_before,
                            localization_stage_after=loc_stage_before,
                            phase_after=state.phase.value,
                            turn_made_progress=False,
                            no_progress_streak_after=state.no_progress_streak,
                            action_decision=f"host_guided_to_{forced_rag_tool}",
                        )
                        state.trajectory.append(trace_entry)
                        continue
                raise ModelConfigurationError(
                    "Writable coder returned a final prose answer before a reviewed source mutation "
                    "was applied; implementation completion requires a real source diff."
                )
            return content

        calls_desc = [f"{c.name}({json.dumps(dict(c.arguments), ensure_ascii=False)[:80]})" for c in turn.tool_calls]
        print(f"  model emitted calls -> {calls_desc}", flush=True)

        if required_rag_choice:
            if (
                len(turn.tool_calls) != 1
                or turn.tool_calls[0].name not in _RAG_EVIDENCE_TOOLS
                or turn.tool_calls[0].name not in phase_tool_names
            ):
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    "Production coder violated the host-required evidence invariant; expected exactly "
                    f"one reviewed RAG tool from {sorted(phase_tool_names)}, received {called}."
                )
            required_rag_choice = False

        if forced_rag_tool is not None:
            if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_rag_tool:
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    f"Production coder violated host-forced RAG tool choice {forced_rag_tool!r}; received {called}."
                )
            forced_rag_tool = None
            forced_rag_attempts = 0

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
                or (call.name == "external_mcp_call" and bool(_external_rag_capability(call.arguments)))
            )

        def execute(
            call: Any,
            allowed_phase_tools: frozenset[str] = phase_tool_names,
            localization_stage: LocalizationStage = current_localization_stage,
        ) -> tuple[Any, Mapping[str, Any]]:
            route_metadata: dict[str, Any] = {
                "skills": list(skills_for_tool(stage, call.name, model_role=role))
            }
            if call.name == "external_mcp_call":
                capability = str(call.arguments.get("capability", "")).strip()
                if capability:
                    route_metadata["external_mcp_capability"] = capability

            if call.name not in allowed_phase_tools:
                err_msg = (
                    f"Agent attempted tool {call.name!r} outside its allowed phase {state.phase.value!r} "
                    f"(allowed tools: {sorted(allowed_phase_tools)})."
                )
                state.record_failure(call.name, err_msg)
                if (
                    call.name in _MUTATION_ACT_TOOLS
                    and implementation_requires_mutation
                    and state.mutation_context is not None
                    and state.mutation_context.is_mutation_ready
                ):
                    # The target is already localized. A stale mutation emitted during
                    # OBSERVE needs a phase correction, not another retrieval round.
                    state.phase = LoopPhase.ACT
                print(f"  [!] PHASE VIOLATION: {call.name} -> {err_msg}", flush=True)
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **route_metadata,
                    "failure_code": "PHASE_PROTOCOL_VIOLATION",
                    "error": err_msg,
                }

            if is_evidence_tool(call):
                if state.is_query_attempted(call.name, call.arguments):
                    err_msg = (
                        f"RetrievalNoProgress: equivalent query already attempted for {call.name} with "
                        f"args={json.dumps(dict(call.arguments), ensure_ascii=False)}"
                    )
                    state.record_failure(call.name, err_msg)
                    print(f"  [!] DUP QUERY REJECTED: {call.name} -> {err_msg}", flush=True)
                    return call, {
                        "ok": False,
                        "tool": call.name,
                        **route_metadata,
                        "error": err_msg,
                    }
                localization_attempt_stage = (
                    localization_stage
                    if implementation_requires_mutation and state.phase == LoopPhase.OBSERVE
                    else None
                )
                state.record_attempted_source(
                    call.name,
                    call.arguments,
                    localization_stage=localization_attempt_stage,
                )

            target_error = _mutation_target_error(
                call.name,
                call.arguments,
                state.mutation_context,
            )
            if target_error is not None:
                state.record_failure(call.name, target_error)
                print(
                    f"  [!] MUTATION TARGET REJECTED: {call.name} -> {target_error}",
                    flush=True,
                )
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **route_metadata,
                    "failure_code": target_error.partition(":")[0],
                    "error": target_error,
                }

            try:
                if call.name.startswith("external_mcp_"):
                    scoped_call = getattr(runtime, "call_scoped", None)
                    if callable(scoped_call):
                        result = scoped_call(
                            stage,
                            call.name,
                            call.arguments,
                            external_server_ids=reviewed_external_servers,
                        )
                    else:
                        raise ModelConfigurationError(
                            "External MCP execution requires a role-scoped agent runtime."
                        )
                else:
                    result = runtime.call(stage, call.name, call.arguments)

                if is_evidence_tool(call):
                    state.record_query(call.name, call.arguments)

                payload: Mapping[str, Any] = {
                    "ok": True,
                    "tool": call.name,
                    **route_metadata,
                    "result": result,
                }
            except Exception as exc:  # noqa: BLE001 - tool failures become observations
                err_msg = f"{type(exc).__name__}: {exc}"
                payload = {
                    "ok": False,
                    "tool": call.name,
                    **route_metadata,
                    "error": err_msg,
                }
                state.record_failure(call.name, err_msg)
                print(f"  [!] TOOL EXCEPTION: {call.name} -> {err_msg}", flush=True)
            return call, payload

        executed = _execute_tool_waves(tuple(turn.tool_calls), execute)
        turn_made_progress = False

        for call, payload in executed:
            ok = payload.get("ok")
            res_preview = str(payload.get("result", payload.get("error", "")))[:100].replace("\n", " ")
            print(f"  tool result -> {call.name} ok={ok} payload={res_preview}", flush=True)

            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            }
            messages.append(
                dict(
                    bounded_tool_message(
                        tool_message,
                        config=config,
                        tools=request.tools,
                    )
                )
            )

            if call.name in _MUTATION_ACT_TOOLS:
                if state.record_mutation(call.name, payload):
                    turn_made_progress = True
                    if all_exposed_names & _VERIFY_TOOLS:
                        state.phase = LoopPhase.VERIFY
                elif bool(payload.get("ok")):
                    # A successful transport with an UNCHANGED/no-diff receipt is not
                    # semantic progress. Counting it reset the fixed-point guard and
                    # let small coders loop indefinitely on the same replacement.
                    state.record_failure(
                        call.name,
                        "MUTATION_UNCHANGED: host receipt proved no source-byte change",
                    )
                else:
                    failure_code = str(payload.get("failure_code") or "")
                    authority_retry = bool(
                        failure_code
                        in {
                            "MUTATION_TARGET_DRIFT",
                            "MUTATION_TARGET_UNBOUND",
                            "MUTATION_TARGET_CREATION_CONFLICT",
                            "PHASE_PROTOCOL_VIOLATION",
                        }
                        and state.mutation_context is not None
                        and state.mutation_context.is_mutation_ready
                    )
                    if authority_retry:
                        # Retrieval cannot expand write authority. Reissue the same
                        # single mutation action against the host-pinned target.
                        state.phase = LoopPhase.ACT
                    elif all_exposed_names & _READ_OBSERVE_TOOLS:
                        state.phase = LoopPhase.OBSERVE
                    state.record_failure(call.name, payload.get("error", "mutation failed"))
                continue

            if call.name in _VERIFY_TOOLS:
                status = "PASS" if bool(payload.get("ok")) else "FAIL"
                if status != state.validation_status:
                    state.validation_status = status
                    turn_made_progress = True
                if status == "FAIL" and implementation_requires_mutation:
                    state.record_failure(call.name, payload.get("error", "verification FAIL"))
                    state.phase = LoopPhase.ACT
                continue

            if is_evidence_tool(call):
                if not bool(payload.get("ok")):
                    state.record_failure(call.name, payload.get("error", "evidence tool error"))
                    continue
                usable = (
                    _usable_rag_result(payload.get("result"))
                    if call.name in _RAG_EVIDENCE_TOOLS
                    else (
                        _usable_external_rag_result(call.arguments, payload.get("result"))
                        if call.name == "external_mcp_call"
                        else bool(payload.get("result"))
                    )
                )
                before_ctx = state.mutation_context
                recorded = state.record_evidence(payload.get("result"), usable=usable)
                after_ctx = state.mutation_context

                ctx_progress = bool(
                    after_ctx is not None
                    and (
                        before_ctx is None
                        or after_ctx.localization_stage != before_ctx.localization_stage
                        or after_ctx.target_path != before_ctx.target_path
                        or after_ctx.target_symbol != before_ctx.target_symbol
                        or after_ctx.source_body != before_ctx.source_body
                    )
                )

                # For implementation, novelty alone is not progress: evidence must
                # advance file/symbol/body localization. This prevents an unrelated
                # first RAG hit from resetting a blocked mutation retry.
                if ctx_progress or (not implementation_requires_mutation and recorded):
                    turn_made_progress = True
                    if (
                        state.phase == LoopPhase.OBSERVE
                        and implementation_requires_mutation
                        and is_mutation_ready(messages, state)
                    ):
                        state.phase = LoopPhase.ACT

        phase_after = state.phase.value
        loc_stage_after = (
            state.mutation_context.localization_stage.value
            if state.mutation_context is not None
            else LocalizationStage.NEED_FILE.value
        )
        ctx_after = _mutation_context_dict(state.mutation_context)
        model_calls_info = [{"name": c.name, "arguments": dict(c.arguments)} for c in turn.tool_calls]
        query_sigs = [retrieval_query_signature(c.name, c.arguments) for c in turn.tool_calls]
        results_info = [
            {
                "name": call.name,
                "ok": payload.get("ok"),
                "failure_code": payload.get("failure_code"),
                "error": payload.get("error"),
            }
            for call, payload in executed
        ]

        if turn_made_progress:
            state.clear_failure()
            state.clear_no_progress_result()
        else:
            state.record_no_progress_result(
                {
                    "phase_before": phase_before,
                    "phase_after": phase_after,
                    "localization_stage_before": loc_stage_before,
                    "localization_stage_after": loc_stage_after,
                    "mutation_context_after": ctx_after,
                    "model_tool_calls": model_calls_info,
                    "tool_results": results_info,
                }
            )

        trace_entry = ExecutionStepTrace(
            step_index=state.step_index,
            phase_before=phase_before,
            localization_stage_before=loc_stage_before,
            mutation_context_before=ctx_before,
            exposed_tools=sorted(phase_tool_names),
            tool_choice=tool_choice,
            input_messages_count=len(messages),
            model_response_content=turn.content or None,
            model_tool_calls=model_calls_info,
            query_signatures=query_sigs,
            tool_results=results_info,
            mutation_context_after=ctx_after,
            localization_stage_after=loc_stage_after,
            phase_after=phase_after,
            turn_made_progress=turn_made_progress,
            no_progress_streak_after=state.no_progress_streak,
            action_decision="tool_wave_executed",
        )
        state.trajectory.append(trace_entry)
        print(
            f"  step {state.step_index} finished: stage={loc_stage_before}->{loc_stage_after} "
            f"phase={phase_before}->{phase_after} progress={turn_made_progress} streak={state.no_progress_streak}\n",
            flush=True,
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
    final_messages_mutable = [dict(message) for message in final_messages]
    final_turn = _generate_turn_with_context_recovery(
        router,
        config=config,
        adapter=adapter,
        request=final_request,
        messages=final_messages_mutable,
        media_paths=(),
        tool_choice=None,
        parallel_tool_calls=False,
    )
    if final_turn.tool_calls:
        raise ModelConfigurationError("Agent emitted tool calls after the host disabled tools.")
    content = final_turn.content.strip()
    if not content:
        raise ModelConfigurationError(empty_error)
    return content


__all__ = [
    "_LOCALIZATION_EVIDENCE_TOOLS",
    "_MUTATION_ACT_TOOLS",
    "_READ_OBSERVE_TOOLS",
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
    "evidence_fingerprint",
    "format_trajectory_summary",
    "generate_with_tools",
    "is_mutation_ready",
    "normalize_retrieval_query",
    "retrieval_query_signature",
    "retrieval_source_key",
]
