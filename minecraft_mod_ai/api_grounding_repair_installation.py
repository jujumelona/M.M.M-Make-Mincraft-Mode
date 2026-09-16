from __future__ import annotations

"""Install fail-closed Java API grounding and verifier-repair convergence guards.

The core tool loop intentionally stays generic.  This installation tightens the two
boundaries that must be Java-aware:

* retrieval evidence may only unlock a fresh Java mutation when it contains concrete,
  target-relevant source/symbol evidence; metadata-only RAG receipts are never evidence;
* verifier repair of a host-tracked existing file may reuse the scalar ``create_file``
  payload as a transactional full-file replacement, matching the lower source-edit
  materializer's SHA-bound semantics.

It also connects the already-defined atomic-output recovery instruction to completion
boundary failures.  Recovery is idempotent by a transcript marker rather than a retry
count.
"""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import wraps
from types import ModuleType
from typing import Any

_INSTALL_MARKER = "_mmm_api_grounding_repair_v1"
_OUTPUT_RECOVERY_MARKER = "MMM_ATOMIC_OUTPUT_RECOVERY_V1"
_PROJECT_RAG_SCHEMA = "mmm/rag-result-v2"
_JAVA_SYMBOL_SCHEMA = "mmm/java-symbols-v1"
_CODE_RAG_SCHEMA = "mmm/code-rag-result-v1"

_CONTENT_KEYS = (
    "parsed_text",
    "text",
    "content",
    "snippet",
    "code",
    "source",
    "source_text",
    "body",
)
_COLLECTION_KEYS = (
    "hits",
    "results",
    "records",
    "documents",
    "chunks",
    "resources",
)
_JAVA_API_RE = re.compile(
    r"(?:\b(?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)\.[A-Za-z0-9_.$]+"
    r"|\b(?:package|import)\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+"
    r"|\b(?:class|interface|record|enum)\s+[A-Za-z_$][\w$]*)"
)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _nonempty_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_nonempty_text(item) for item in value)
    return False


def _contentful_mapping(value: Any) -> bool:
    """Return True only for payloads carrying substantive retrievable content."""
    if not isinstance(value, Mapping):
        return False
    for key in _CONTENT_KEYS:
        if _nonempty_text(value.get(key)):
            return True
    for key in _COLLECTION_KEYS:
        for item in _sequence(value.get(key)):
            if isinstance(item, str) and item.strip():
                return True
            if isinstance(item, Mapping) and _contentful_mapping(item):
                return True
    structured = value.get("structured_content")
    if isinstance(structured, Mapping) and _contentful_mapping(structured):
        return True
    for key in ("result", "data"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and _contentful_mapping(nested):
            return True
    return False


def strict_usable_rag_result(value: Any) -> bool:
    """Reject metadata-only RAG envelopes even when their mapping is non-empty."""
    if not isinstance(value, Mapping) or not value:
        return False

    hits = value.get("hits")
    if isinstance(hits, list):
        return any(isinstance(hit, Mapping) and _contentful_mapping(hit) for hit in hits)

    for key in _CONTENT_KEYS:
        if _nonempty_text(value.get(key)):
            return True

    resources = value.get("resources")
    if isinstance(resources, list) and any(
        isinstance(item, Mapping) and _contentful_mapping(item) for item in resources
    ):
        return True

    structured = value.get("structured_content")
    if isinstance(structured, Mapping):
        return strict_usable_rag_result(structured)

    for key in ("result", "data"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and strict_usable_rag_result(nested):
            return True

    # A receipt/hash/query/target/source-list proves transport/provenance, not that
    # the model received usable evidence.  Never fall back to ``bool(value)`` here.
    return False


def _nested_project_rag_schema(value: Mapping[str, Any]) -> bool:
    if str(value.get("schema_version") or "").strip() == _PROJECT_RAG_SCHEMA:
        return True
    structured = value.get("structured_content")
    return bool(
        isinstance(structured, Mapping)
        and str(structured.get("schema_version") or "").strip() == _PROJECT_RAG_SCHEMA
    )


def _java_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(value, Mapping):
        return texts
    for key in _CONTENT_KEYS:
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            texts.append(raw)
        else:
            texts.extend(str(item) for item in _sequence(raw) if isinstance(item, str) and item.strip())
    for key in _COLLECTION_KEYS:
        for item in _sequence(value.get(key)):
            if isinstance(item, Mapping):
                texts.extend(_java_texts(item))
    for key in ("structured_content", "result", "data"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            texts.extend(_java_texts(nested))
    return texts


def authoritative_java_evidence(value: Any) -> bool:
    """Classify evidence strong enough to unlock a fresh Java ACT phase."""
    if not isinstance(value, Mapping) or not value:
        return False

    # The built-in primary catalog is useful context, but its v2 envelope is not an
    # exact classpath/symbol proof.  It may never by itself authorize fresh Java.
    if _nested_project_rag_schema(value):
        return False

    schema = str(value.get("schema_version") or "").strip()
    if schema == _JAVA_SYMBOL_SCHEMA:
        return bool(_sequence(value.get("symbols")))

    if schema == _CODE_RAG_SCHEMA:
        hits = value.get("hits")
        return bool(
            isinstance(hits, list)
            and any(isinstance(hit, Mapping) and _contentful_mapping(hit) for hit in hits)
        )

    # Reviewed external MCP responses are version-injected by the host.  They count
    # only when they carry concrete Java/Fabric/Minecraft symbols or source, never
    # merely capability/provenance metadata.
    if any(_JAVA_API_RE.search(text) for text in _java_texts(value)):
        return True

    mappings = value.get("mappings")
    if isinstance(mappings, Mapping) and bool(mappings):
        return True
    if isinstance(mappings, list) and any(isinstance(item, Mapping) and item for item in mappings):
        return True

    for key in ("structured_content", "result", "data"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and authoritative_java_evidence(nested):
            return True
    return False


def _fresh_java_context(context: Any) -> bool:
    if context is None or not bool(getattr(context, "is_new_file", False)):
        return False
    path = str(getattr(context, "target_path", "") or "").replace("\\", "/").casefold()
    return path.endswith(".java")


def _repair_existing_context(context: Any) -> bool:
    return bool(
        context is not None
        and not bool(getattr(context, "is_new_file", False))
        and str(getattr(context, "evidence_source", "") or "") == "mutation_receipt"
        and isinstance(getattr(context, "source_body", None), str)
        and bool(str(getattr(context, "source_body", "")).strip())
        and str(getattr(context, "target_path", "") or "").casefold().endswith(".java")
    )


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""


def _source_operation(schema: Mapping[str, Any]) -> Mapping[str, Any] | None:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return None
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return None
    operation = properties.get("operation")
    return operation if isinstance(operation, Mapping) else None


def _arguments_path(arguments: Mapping[str, Any]) -> str:
    for key in ("path", "file", "target_path", "target_file"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            clean = value.strip().replace("\\", "/")
            while clean.startswith("./"):
                clean = clean[2:]
            return clean
    return ""


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


def install(model_router_module: ModuleType, progress_module: ModuleType) -> None:
    """Install the Java-grounding/repair boundary exactly once."""
    if getattr(progress_module, _INSTALL_MARKER, False):
        return

    original_usable_rag = model_router_module._usable_rag_result
    original_record_evidence = progress_module.HostRunState.record_evidence
    original_filter = progress_module._filter_tools_for_phase
    original_schema_for_context = progress_module._source_edit_schema_for_context
    original_target_error = progress_module._mutation_target_error
    original_generate_recovery = progress_module._generate_turn_with_context_recovery

    @wraps(original_usable_rag)
    def usable_rag_result(value: Any) -> bool:
        return strict_usable_rag_result(value)

    @wraps(original_record_evidence)
    def record_evidence(self: Any, value: Any, *, usable: bool) -> bool:
        context = getattr(self, "mutation_context", None)
        if usable and _fresh_java_context(context):
            usable = authoritative_java_evidence(value)
        return original_record_evidence(self, value, usable=usable)

    @wraps(original_filter)
    def filter_tools_for_phase(
        exposed_tools: Sequence[Mapping[str, Any]],
        phase: Any,
        role: str,
        *,
        mutation_context: Any = None,
        attempted_sources: Sequence[str] | set[str] | frozenset[str] = frozenset(),
        localization_active: bool | None = None,
        semantic_retrieval_choice: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        if (
            getattr(phase, "value", phase) == "OBSERVE"
            and _fresh_java_context(mutation_context)
            and bool(getattr(mutation_context, "is_mutation_ready", False))
        ):
            by_name = {
                _tool_name(schema): schema
                for schema in exposed_tools
                if isinstance(schema, Mapping) and _tool_name(schema)
            }
            attempted = set(attempted_sources)

            def attempted_tool(name: str) -> bool:
                return name in attempted or any(item.startswith(name + ":") for item in attempted)

            # Local exact source first, then reviewed version-injected external API,
            # then JDT.  The target-neutral primary catalog is deliberately not an
            # authorization route for a fresh Java file.
            for name in ("search_code_rag", "external_mcp_call", "java_workspace_symbols"):
                if name in by_name and not attempted_tool(name):
                    return (by_name[name],)
            return ()
        return original_filter(
            exposed_tools,
            phase,
            role,
            mutation_context=mutation_context,
            attempted_sources=attempted_sources,
            localization_active=localization_active,
            semantic_retrieval_choice=semantic_retrieval_choice,
        )

    @wraps(original_schema_for_context)
    def source_edit_schema_for_context(schema: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        narrowed = original_schema_for_context(schema, context)
        if _tool_name(schema) != "apply_source_edit" or not _repair_existing_context(context):
            return narrowed

        repaired = deepcopy(narrowed)
        original_operation = _source_operation(schema)
        repaired_operation = _source_operation(repaired)
        if isinstance(original_operation, Mapping) and isinstance(repaired_operation, dict):
            original_enum = original_operation.get("enum")
            narrowed_enum = repaired_operation.get("enum")
            if (
                isinstance(original_enum, list)
                and "create_file" in original_enum
                and isinstance(narrowed_enum, list)
                and "create_file" not in narrowed_enum
            ):
                repaired_operation["enum"] = ["create_file", *narrowed_enum]

        function = repaired.get("function")
        if isinstance(function, dict):
            description = str(function.get("description") or "").strip()
            suffix = (
                "Verifier-repair target is the same host-pinned existing file. "
                "A create_file payload with complete corrected content is transactionally "
                "materialized as a SHA-bound replacement; it cannot create a second path."
            )
            function["description"] = f"{description} {suffix}".strip()
        return repaired

    @wraps(original_target_error)
    def mutation_target_error(tool_name: str, arguments: Mapping[str, Any], context: Any) -> str | None:
        error = original_target_error(tool_name, arguments, context)
        if not error or not error.startswith("MUTATION_TARGET_CREATION_CONFLICT"):
            return error
        if tool_name != "apply_source_edit" or not _repair_existing_context(context):
            return error
        if str(arguments.get("operation") or "").strip().casefold() != "create_file":
            return error
        supplied = _arguments_path(arguments)
        target = str(getattr(context, "target_path", "") or "").replace("\\", "/")
        while target.startswith("./"):
            target = target[2:]
        return None if supplied and supplied == target else error

    @wraps(original_generate_recovery)
    def generate_turn_with_context_recovery(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_generate_recovery(*args, **kwargs)
        except BaseException as exc:
            if not _completion_boundary_error(exc):
                raise
            request = kwargs.get("request")
            messages = kwargs.get("messages")
            if request is None or not isinstance(messages, list):
                raise
            if any(
                isinstance(message, Mapping)
                and _OUTPUT_RECOVERY_MARKER in str(message.get("content") or "")
                for message in messages
            ):
                raise
            instruction = progress_module._atomic_output_recovery_instruction(request)
            messages.append(
                {
                    "role": "system",
                    "content": f"{_OUTPUT_RECOVERY_MARKER}\n{instruction}",
                }
            )
            return original_generate_recovery(*args, **kwargs)

    model_router_module._usable_rag_result = usable_rag_result
    progress_module.HostRunState.record_evidence = record_evidence
    progress_module._filter_tools_for_phase = filter_tools_for_phase
    progress_module._source_edit_schema_for_context = source_edit_schema_for_context
    progress_module._mutation_target_error = mutation_target_error
    progress_module._generate_turn_with_context_recovery = generate_turn_with_context_recovery
    setattr(progress_module, _INSTALL_MARKER, True)


__all__ = [
    "authoritative_java_evidence",
    "install",
    "strict_usable_rag_result",
]
