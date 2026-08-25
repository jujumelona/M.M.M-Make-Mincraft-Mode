from __future__ import annotations

"""Single host-owned execution loop for retrieve/act/observe/verify."""

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .agent_intent import implementation_requested
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
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
    SOURCE_MUTATION_NAMES,
    mutation_history_applied,
    mutation_payload_applied,
)


class LoopPhase(str, Enum):
    OBSERVE = "OBSERVE"
    ACT = "ACT"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"


class RetrievalDecision(str, Enum):
    EXECUTE = "execute"
    DUPLICATE_QUERY = "duplicate_query"


class RetrievalObservation(str, Enum):
    FRESH = "fresh"
    WEAK = "weak"
    DUPLICATE_EVIDENCE = "duplicate_evidence"


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


def retrieval_query_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    source = retrieval_source_key(tool_name, arguments)
    data = dict(arguments)
    query = normalize_retrieval_query(data.pop("query", ""))
    if tool_name == "external_mcp_call":
        nested = data.get("arguments")
        if isinstance(nested, Mapping):
            nested_data = dict(nested)
            nested_query = normalize_retrieval_query(nested_data.pop("query", ""))
            if nested_query:
                query = nested_query
            data["arguments"] = nested_data
    canonical = json.dumps(
        {"source": source, "query": query, "scope": _stable_value(data, drop_volatile=False)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_fingerprint(value: Any) -> str | None:
    stable = _stable_value(value, drop_volatile=True)
    if stable in (None, "", [], {}):
        return None
    canonical = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LocalizationStage(str, Enum):
    NEED_FILE = "NEED_FILE"
    NEED_SYMBOL = "NEED_SYMBOL"
    NEED_BODY = "NEED_BODY"
    READY = "READY"


@dataclass(frozen=True)
class TargetMutationContext:
    """Target-bound mutation context enforcing hierarchical localization before ACT phase entry.

    Adheres to Agentless (file -> symbol -> edit location -> concrete source span)
    and 2026 repository-scale function-level repair findings.
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

    def merge(self, other: "TargetMutationContext") -> "TargetMutationContext":
        """Cumulatively accumulate localization discoveries across multiple retrieval turns."""
        return TargetMutationContext(
            target_path=other.target_path or self.target_path,
            target_symbol=other.target_symbol or self.target_symbol,
            source_body=other.source_body or self.source_body,
            start_line=other.start_line if other.start_line is not None else self.start_line,
            end_line=other.end_line if other.end_line is not None else self.end_line,
            is_new_file=other.is_new_file or self.is_new_file,
            evidence_source=other.evidence_source or self.evidence_source,
            base_revision_sha=other.base_revision_sha or self.base_revision_sha,
        )


@dataclass
class HostRunState:
    """Unified single-owner state tracking execution, progress deltas, and bounds."""

    phase: LoopPhase = LoopPhase.OBSERVE
    step_index: int = 0
    no_progress_streak: int = 0
    attempted_queries: set[str] = field(default_factory=set)
    attempted_sources: set[str] = field(default_factory=set)
    evidence_fingerprints: set[str] = field(default_factory=set)
    mutation_context: TargetMutationContext | None = None
    applied_mutations: list[str] = field(default_factory=list)
    workspace_changed: bool = False
    validation_status: str = "PENDING"
    last_failure_digest: str | None = None
    last_failure_reason: str | None = None
    last_result_digest: str | None = None
    termination_reason: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def has_fresh_evidence(self) -> bool:
        with self._lock:
            return bool(self.evidence_fingerprints)

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

    def next_untried_internal_tool(
        self,
        exposed_tools: Sequence[str] | set[str] | frozenset[str],
        *,
        preferred: Sequence[str],
    ) -> str | None:
        exposed = set(exposed_tools)
        with self._lock:
            for name in preferred:
                if name in exposed and name not in self.attempted_sources:
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

    # 1. Explicit new file creation
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

    # 2. Hits with snippet/code/lines from search_code_rag
    hits = payload.get("hits") or payload.get("results")
    if isinstance(hits, (list, tuple)):
        # Look for a hit with concrete code snippet
        for hit in hits:
            if isinstance(hit, Mapping):
                path = str(hit.get("path") or hit.get("file") or hit.get("uri") or "").strip()
                if path:
                    snippet = hit.get("snippet") or hit.get("code") or hit.get("content") or hit.get("source")
                    if isinstance(snippet, (list, tuple)):
                        snippet = "\n".join(str(s) for s in snippet)
                    if isinstance(snippet, str) and _is_code_bearing_text(snippet):
                        symbol = str(hit.get("symbol") or hit.get("function") or hit.get("name") or "").strip() or None
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
        # If no snippet, record bare path hit (hierarchical step 1: file candidate only)
        for hit in hits:
            if isinstance(hit, Mapping):
                path = str(hit.get("path") or hit.get("file") or hit.get("uri") or "").strip()
                if path:
                    return TargetMutationContext(
                        target_path=path,
                        source_body=None,
                        evidence_source="search_code_rag_path_only",
                    )

    # 3. Symbols from java_workspace_symbols (hierarchical step 2: symbol in file)
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

    # 4. Files mapping (path -> code or list of {path, content})
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

    # 5. Nested initial_exact_source_context
    exact = payload.get("initial_exact_source_context")
    if exact is not None and exact is not payload:
        ctx = _extract_mutation_context_from_payload(exact)
        if ctx is not None:
            return ctx

    # 6. Direct target_file / target_source
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
) -> tuple[Mapping[str, Any], ...]:
    del role
    by_name = {_tool_name(schema): schema for schema in exposed_tools if _tool_name(schema)}
    if not by_name:
        return ()

    if phase == LoopPhase.OBSERVE:
        stage = (
            mutation_context.localization_stage
            if mutation_context is not None
            else LocalizationStage.NEED_FILE
        )
        if stage == LocalizationStage.NEED_FILE:
            preferred = ("search_code_rag", "search_project_rag", "inspect_existing_mod")
            selected_names = [name for name in preferred if name in by_name]
            if not selected_names:
                selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
        elif stage == LocalizationStage.NEED_SYMBOL:
            preferred = ("java_workspace_symbols", "java_diagnostics")
            selected_names = [name for name in preferred if name in by_name]
            if not selected_names and "search_code_rag" in by_name:
                selected_names = ["search_code_rag"]
            elif not selected_names:
                selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
        elif stage == LocalizationStage.NEED_BODY:
            preferred = ("search_code_rag", "inspect_existing_mod")
            selected_names = [name for name in preferred if name in by_name]
            if not selected_names:
                selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
        else:
            selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
    elif phase == LoopPhase.ACT:
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
    """Fit one live agent turn and retry context pressure exactly once."""

    fitted = fit_messages_to_context(messages, config=config, tools=request.tools)
    _replace_live_messages(messages, fitted)
    turn_request = replace(
        request,
        messages=tuple(messages),
        media_paths=media_paths,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    try:
        with router._generation_scope(config):
            return adapter.generate_turn(turn_request)
    except BaseException as exc:
        if completion_boundary_kind(exc) != CONTEXT_PRESSURE:
            raise

        emergency_budget = min(
            40 * 1024,
            request_message_budget(config, request.tools),
        )
        emergency = emergency_fit_messages(
            messages,
            budget_bytes=emergency_budget,
        )
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
            f"budget_bytes={emergency_budget}",
            flush=True,
        )
        try:
            with router._generation_scope(config):
                return adapter.generate_turn(retry_request)
        except BaseException as retry_exc:
            if completion_boundary_kind(retry_exc) == CONTEXT_PRESSURE:
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

    from .agent_capability_context import reviewed_mcp_servers_for_model_role, skills_for_tool
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

        if round_limit is not None and state.step_index > round_limit:
            if require_rag and not state.has_fresh_evidence:
                raise ModelConfigurationError(
                    "Agent reached the explicit tool-round limit before required "
                    "evidence became available."
                )
            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder reached the explicit tool-round limit before a "
                    "reviewed source mutation was applied; refusing a prose-only implementation."
                )
            return _finalize_without_tools(
                router,
                config,
                adapter,
                request,
                messages,
                instruction=(
                    f"The explicitly configured host tool limit was reached after {round_limit} rounds. "
                    "Do not call more tools. Return the final answer using only observations already present."
                ),
                empty_error="Agent returned an empty final response at the explicit tool-round limit.",
            )

        if forced_rag_tool is None and state.no_progress_streak >= 2:
            reason_suffix = f": {state.last_failure_reason}" if state.last_failure_reason else ""
            if require_rag and not state.has_fresh_evidence:
                raise ModelConfigurationError(
                    "Required production evidence is unavailable: every host-forceable "
                    "RAG source was already attempted without novel usable evidence, and "
                    f"the model selected no other reviewed retrieval route.{reason_suffix}"
                )
            if implementation_requires_mutation and not state.workspace_changed and not mutation_history_applied(messages):
                raise ModelConfigurationError(
                    "Writable coder reached a no-progress boundary before a reviewed source "
                    f"mutation was applied; refusing a prose-only implementation.{reason_suffix}"
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

        phase_tools = _filter_tools_for_phase(
            all_exposed_tools,
            state.phase,
            role,
            mutation_context=state.mutation_context,
        )
        phase_tool_names = frozenset(_tool_name(s) for s in phase_tools if _tool_name(s))

        tool_choice = request.tool_choice
        parallel_tool_calls = request.parallel_tool_calls

        if forced_rag_tool is not None:
            tool_choice = {"type": "function", "function": {"name": forced_rag_tool}}
            parallel_tool_calls = False
        elif state.phase == LoopPhase.ACT:
            mutation_names = [n for n in phase_tool_names if n in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel_tool_calls = False

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
            if not content:
                raise ModelConfigurationError("Tool-capable model returned an empty final response.")
            if forced_rag_tool is not None:
                forced_rag_attempts += 1
                if forced_rag_attempts >= 2:
                    raise ModelConfigurationError(
                        f"Production coder did not honor host-forced RAG tool choice {forced_rag_tool!r} "
                        "after two bounded attempts."
                    )
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": f"Call the required function {forced_rag_tool} exactly once now. Do not answer in prose.",
                    },
                ])
                continue
            if require_rag and not state.has_fresh_evidence:
                forced_rag_tool = state.next_untried_internal_tool(
                    all_exposed_names,
                    preferred=("search_code_rag", "search_project_rag"),
                )
                if forced_rag_tool is None:
                    raise ModelConfigurationError(
                        "Required production evidence is unavailable: every host-forceable "
                        "RAG source was already attempted without novel usable evidence, and "
                        "the model selected no other reviewed retrieval route."
                    )
                forced_rag_attempts = 0
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "system",
                        "content": (
                            f"Baseline production evidence is still required. Call {forced_rag_tool} "
                            "exactly once with a concrete query for the current implementation need. "
                            "After its observation, choose any further retrieval only if it can add "
                            "materially new evidence; do not repeat exhausted queries."
                        ),
                    },
                ])
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
                        preferred = ("java_workspace_symbols", "java_diagnostics", "search_code_rag")
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
                        continue
                raise ModelConfigurationError(
                    "Writable coder returned a final prose answer before a reviewed source mutation "
                    "was applied; implementation completion requires a real source diff."
                )
            return content

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

        def is_retrieval(call: Any) -> bool:
            return call.name in _RAG_EVIDENCE_TOOLS or (
                call.name == "external_mcp_call" and bool(_external_rag_capability(call.arguments))
            )

        def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:
            route_metadata: dict[str, Any] = {
                "skills": list(skills_for_tool(stage, call.name, model_role=role))
            }
            if call.name == "external_mcp_call":
                capability = str(call.arguments.get("capability", "")).strip()
                if capability:
                    route_metadata["external_mcp_capability"] = capability

            # Check phase legality (fail-closed to phase_tool_names)
            if call.name not in phase_tool_names:
                return call, {
                    "ok": False,
                    "tool": call.name,
                    **route_metadata,
                    "error": (
                        f"Agent attempted tool {call.name!r} outside its allowed phase {state.phase.value!r} "
                        f"(allowed tools: {sorted(phase_tool_names)})."
                    ),
                }

            if is_retrieval(call):
                is_new_query = state.record_query(call.name, call.arguments)
                if not is_new_query:
                    return call, {
                        "ok": False,
                        "tool": call.name,
                        **route_metadata,
                        "error": "RetrievalNoProgress: equivalent query already attempted for this evidence need",
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

                payload: Mapping[str, Any] = {
                    "ok": True,
                    "tool": call.name,
                    **route_metadata,
                    "result": result,
                }
                if call.name in SOURCE_MUTATION_NAMES:
                    payload = {
                        **payload,
                        "_mmm_source_mutation": {
                            "tool": call.name,
                            "status": "APPLIED_BY_HOST_RUNTIME",
                        },
                    }
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                payload = {
                    "ok": False,
                    "tool": call.name,
                    **route_metadata,
                    "error": err_msg,
                }
                state.last_failure_reason = f"{call.name}: {err_msg}"
            return call, payload

        executed = _execute_tool_waves(tuple(turn.tool_calls), execute)
        turn_made_progress = False

        for call, payload in executed:
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
                elif not bool(payload.get("ok")):
                    if all_exposed_names & _READ_OBSERVE_TOOLS:
                        state.phase = LoopPhase.OBSERVE
                    err_text = str(payload.get("error", "mutation failed"))
                    state.last_failure_reason = f"{call.name}: {err_text}"
                    error_digest = hashlib.sha256(err_text.encode("utf-8")).hexdigest()[:16]
                    if error_digest != state.last_failure_digest:
                        state.last_failure_digest = error_digest
                        turn_made_progress = True
                continue

            if call.name in _VERIFY_TOOLS:
                status = "PASS" if bool(payload.get("ok")) else "FAIL"
                if status != state.validation_status:
                    state.validation_status = status
                    turn_made_progress = True
                if status == "FAIL" and implementation_requires_mutation:
                    state.last_failure_reason = f"{call.name}: {payload.get('error', 'verification FAIL')}"
                    state.phase = LoopPhase.ACT
                continue

            if is_retrieval(call):
                if not bool(payload.get("ok")):
                    state.last_failure_reason = f"{call.name}: {payload.get('error', 'retrieval error')}"
                    continue
                usable = _usable_rag_result(payload.get("result")) if call.name in _RAG_EVIDENCE_TOOLS else _usable_external_rag_result(call.arguments, payload.get("result"))
                recorded = state.record_evidence(payload.get("result"), usable=usable)
                if recorded:
                    turn_made_progress = True
                    if state.phase == LoopPhase.OBSERVE and implementation_requires_mutation:
                        if is_mutation_ready(messages, state):
                            state.phase = LoopPhase.ACT

        if turn_made_progress:
            state.no_progress_streak = 0
        else:
            state.no_progress_streak += 1


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
    "HostRunState",
    "LocalizationStage",
    "LoopPhase",
    "RetrievalDecision",
    "RetrievalNoProgressError",
    "RetrievalObservation",
    "RetrievalProgress",
    "TargetMutationContext",
    "_MUTATION_ACT_TOOLS",
    "_READ_OBSERVE_TOOLS",
    "_VERIFY_TOOLS",
    "evidence_fingerprint",
    "generate_with_tools",
    "is_mutation_ready",
    "normalize_retrieval_query",
    "retrieval_query_signature",
    "retrieval_source_key",
]
