from __future__ import annotations

"""Host-owned evidence routing for generation and verifier repair.

The coding model authors search intent and source code; it does not own retriever
selection. Fresh implementation semantics come from the active task capsule rather
than filesystem existence, and verifier repair routes are delegated to the canonical
repair-evidence classifier.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .repair_evidence_router import classify_repair_evidence_route

_REVIEWED_EVIDENCE_TOOLS = frozenset(
    {
        "search_code_rag",
        "search_project_rag",
        "java_workspace_symbols",
        "read_reuse_source",
        "discover_ecosystem_resources",
        "inspect_modrinth_project",
        "inspect_github_repository",
        "inspect_huggingface_model",
        "assess_technology_compatibility",
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    }
)
_JAVA_API_EVIDENCE_RE = re.compile(
    r"(?:\b(?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)\.[A-Za-z0-9_.$]+"
    r"|\b(?:package|import)\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+"
    r"|\b(?:class|interface|record|enum)\s+[A-Za-z_$][\w$]*)"
)
_SYMBOL_LINE_RE = re.compile(
    r"symbol:\s+(?:(?:class|variable|method)\s+)?([A-Za-z_$][A-Za-z0-9_$]*)",
    re.IGNORECASE,
)


def initial_evidence_required(
    *,
    role: str,
    host_grounded: bool,
    router_requires_fresh_evidence: bool,
    implementation_requires_mutation: bool,
    host_target_execution_authority: bool,
    compile_backed_java: bool,
    authored_workspace_refresh: bool = False,
) -> bool:
    """Decide whether mutation needs speculative pre-implementation retrieval.

    A pinned Java target with a mandatory target compiler already has a stronger
    executable feedback loop than speculative API search. In that case the host
    lets the coder perform the bounded edit first and routes only concrete compiler
    diagnostics into repair evidence. Retrieval remains mandatory when there is no
    executable target authority, when the host explicitly needs a workspace refresh,
    or when no compile-backed feedback loop exists and policy requires fresh evidence.
    """

    if authored_workspace_refresh:
        return True
    if role not in {"coder", "coder_safe"} or host_grounded:
        return False
    if (
        implementation_requires_mutation
        and host_target_execution_authority
        and compile_backed_java
    ):
        return False
    if router_requires_fresh_evidence:
        return True
    return bool(
        implementation_requires_mutation
        and not host_target_execution_authority
    )


def semantic_fresh_java(
    reuse_action: str | None,
    target_path: str | None,
    *,
    materialized_new_file: bool,
) -> bool:
    """Bind implementation freshness to task semantics, not file existence."""

    path = str(target_path or "").replace("\\", "/").strip()
    if not path.casefold().endswith(".java"):
        return False
    action = str(reuse_action or "").strip().casefold()
    if action:
        return action == "fresh"
    return bool(materialized_new_file)


def _first_unexecuted(
    available: Sequence[str] | set[str] | frozenset[str],
    attempted: Sequence[str] | set[str] | frozenset[str],
    preferred: Sequence[str],
) -> tuple[str, ...]:
    available_names = set(available)
    attempted_names = set(attempted)
    for name in preferred:
        if name in available_names and name not in attempted_names:
            return (name,)
    return ()


def initial_evidence_frontier(
    *,
    available: Sequence[str] | set[str] | frozenset[str],
    attempted: Sequence[str] | set[str] | frozenset[str],
    localization_stage: str,
    semantic_fresh_java_target: bool,
) -> tuple[str, ...]:
    """Select exactly one host-owned evidence route for the active obligation."""

    if semantic_fresh_java_target:
        preferred = (
            "search_code_rag",
            "search_project_rag",
            "external_mcp_capabilities",
            "external_mcp_schema",
            "external_mcp_call",
        )
    elif localization_stage == "NEED_FILE":
        preferred = ("search_code_rag", "search_project_rag")
    elif localization_stage == "NEED_SYMBOL":
        preferred = ("java_workspace_symbols", "search_code_rag", "search_project_rag")
    elif localization_stage == "NEED_BODY":
        preferred = ("search_code_rag", "java_workspace_symbols", "search_project_rag")
    else:
        preferred = ("search_project_rag", "search_code_rag")
    return _first_unexecuted(available, attempted, preferred)


def evidence_obligation_satisfied(
    *,
    require_evidence: bool,
    semantic_fresh_java_target: bool,
    has_fresh_evidence: bool,
    has_authoritative_java_evidence: bool,
) -> bool:
    """Admit mutation only when the host-owned evidence obligation is satisfied."""

    if not require_evidence:
        return True
    if semantic_fresh_java_target:
        return bool(has_authoritative_java_evidence)
    return bool(has_fresh_evidence)


def verifier_diagnostic_bundle(
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    messages: list[str] = []
    files: list[str] = []
    symbols: list[str] = []
    for item in errors:
        message = str(item.get("message") or "").strip()
        if message:
            messages.append(message)
            for match in _SYMBOL_LINE_RE.finditer(message):
                symbol = match.group(1)
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
        path = str(item.get("path") or item.get("file") or "").strip()
        if path and path not in files:
            files.append(path)
    return {"messages": messages, "files": files, "symbols": symbols}


def repair_evidence_route_for_errors(
    errors: Sequence[Mapping[str, Any]],
    *,
    local_source: str | None = None,
    target_path: str | None = None,
) -> dict[str, Any]:
    base_context: dict[str, Any] = {}
    if isinstance(local_source, str) and local_source.strip():
        base_context = {
            "rag": {
                "hits": [
                    {
                        "path": str(target_path or ""),
                        "text": local_source,
                    }
                ]
            }
        }
    return classify_repair_evidence_route(
        verifier_diagnostic_bundle(errors),
        base_context,
    )


def repair_route_requires_retrieval(route: str | None) -> bool:
    return str(route or "").strip() in {"official_api", "compatibility"}


def recovery_evidence_frontier(
    *,
    available: Sequence[str] | set[str] | frozenset[str],
    attempted: Sequence[str] | set[str] | frozenset[str],
    route: str | None,
) -> tuple[str, ...]:
    kind = str(route or "").strip()
    if kind == "compatibility":
        preferred = (
            "inspect_modrinth_project",
            "search_project_rag",
            "external_mcp_capabilities",
            "external_mcp_schema",
            "external_mcp_call",
            "search_code_rag",
            "java_workspace_symbols",
        )
    elif kind == "official_api":
        preferred = (
            "search_project_rag",
            "search_code_rag",
            "external_mcp_capabilities",
            "external_mcp_schema",
            "external_mcp_call",
        )
    else:
        preferred = ("search_code_rag", "java_workspace_symbols", "search_project_rag")
    return _first_unexecuted(available, attempted, preferred)


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


def _has_quality_warning(value: Any) -> bool:
    if isinstance(value, Mapping):
        warning = value.get("retrieval_quality_warning")
        if warning not in (None, "", False, [], {}):
            return True
        return any(_has_quality_warning(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_quality_warning(child) for child in value)
    return False


def _text_fields(value: Mapping[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    for key in (
        "parsed_text",
        "text",
        "content",
        "snippet",
        "code",
        "source",
        "source_text",
        "body",
    ):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            texts.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            texts.extend(
                str(item)
                for item in raw
                if isinstance(item, str) and item.strip()
            )
    return tuple(texts)


def _has_symbol_records(value: Any) -> bool:
    if isinstance(value, Mapping):
        symbols = value.get("symbols")
        if (
            isinstance(symbols, Sequence)
            and not isinstance(symbols, (str, bytes, bytearray))
            and any(isinstance(item, Mapping) and bool(item) for item in symbols)
        ):
            return True
        return any(_has_symbol_records(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_symbol_records(child) for child in value)
    return False


def _has_mapping_records(value: Any) -> bool:
    if isinstance(value, Mapping):
        mappings = value.get("mappings")
        if isinstance(mappings, Mapping) and bool(mappings):
            return True
        if (
            isinstance(mappings, Sequence)
            and not isinstance(mappings, (str, bytes, bytearray))
            and any(isinstance(item, Mapping) and bool(item) for item in mappings)
        ):
            return True
        return any(_has_mapping_records(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_mapping_records(child) for child in value)
    return False


def _api_hit_exists(value: Any, *, target_path: str | None) -> bool:
    target = str(target_path or "").replace("\\", "/").strip()
    if isinstance(value, Mapping):
        path = str(
            value.get("source_path")
            or value.get("path")
            or value.get("file")
            or ""
        ).replace("\\", "/").strip()
        if path and target and path == target:
            return False
        if any(_JAVA_API_EVIDENCE_RE.search(text) for text in _text_fields(value)):
            return True
        return any(
            _api_hit_exists(child, target_path=target)
            for child in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_api_hit_exists(child, target_path=target) for child in value)
    return False


def authoritative_java_evidence(
    value: Any,
    *,
    target_path: str | None = None,
) -> bool:
    """Require concrete, non-degraded target/API evidence for fresh Java."""

    if not isinstance(value, Mapping) or not value or _has_quality_warning(value):
        return False
    if _mapping_schema(value, "mmm/rag-result-v2"):
        return False
    if _mapping_schema(value, "mmm/java-symbols-v1"):
        return _has_symbol_records(value)
    if _mapping_schema(value, "mmm/code-rag-result-v1"):
        return _api_hit_exists(value, target_path=target_path)
    if _api_hit_exists(value, target_path=target_path):
        return True
    return _has_mapping_records(value)


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return (
        str(function.get("name") or "").strip()
        if isinstance(function, Mapping)
        else ""
    )


def _translate_rejected_arguments(
    phase_tools: Sequence[Mapping[str, Any]],
    forced_evidence_tool: str,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    schema = next(
        (
            item
            for item in phase_tools
            if isinstance(item, Mapping) and _tool_name(item) == forced_evidence_tool
        ),
        None,
    )
    if not isinstance(schema, Mapping):
        return None
    original = str(payload.get("original_tool") or "").strip()
    if (
        not original
        or original == forced_evidence_tool
        or original not in _REVIEWED_EVIDENCE_TOOLS
    ):
        return None
    raw = payload.get("raw_arguments")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    function = schema.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    required = parameters.get("required") if isinstance(parameters, Mapping) else ()
    if not isinstance(properties, Mapping):
        return None
    candidate = {
        str(key): value
        for key, value in parsed.items()
        if str(key) in properties
    }
    required_names = {
        str(name)
        for name in required
        if isinstance(name, str) and name
    }
    if any(
        name not in candidate or candidate[name] in (None, "", [], {}, ())
        for name in required_names
    ):
        return None
    return candidate


def normalize_forced_evidence_rejection_calls(
    calls: Sequence[Any],
    *,
    phase_tools: Sequence[Mapping[str, Any]],
    forced_evidence_tool: str | None,
) -> tuple[Any, ...] | None:
    """Rebind an alternate reviewed retriever call to the host-owned route."""

    forced = str(forced_evidence_tool or "").strip()
    if not forced or len(calls) != 1:
        return None
    call = calls[0]
    if str(getattr(call, "name", "") or "").strip() != "__mmm_rejected_tool_call__":
        return None
    payload = getattr(call, "arguments", None)
    if not isinstance(payload, Mapping):
        return None
    arguments = _translate_rejected_arguments(phase_tools, forced, payload)
    if arguments is None:
        return None
    return (
        replace(
            call,
            name=forced,
            arguments=arguments,
            raw_arguments=json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )


__all__ = [
    "authoritative_java_evidence",
    "evidence_obligation_satisfied",
    "initial_evidence_frontier",
    "initial_evidence_required",
    "normalize_forced_evidence_rejection_calls",
    "recovery_evidence_frontier",
    "repair_evidence_route_for_errors",
    "repair_route_requires_retrieval",
    "semantic_fresh_java",
    "verifier_diagnostic_bundle",
]
