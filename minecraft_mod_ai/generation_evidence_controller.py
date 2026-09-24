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
_IMPORT_API_RE = re.compile(
    r"\bimport\s+((?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)"
    r"(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+)\s*;"
)
_QUALIFIED_API_RE = re.compile(
    r"\b((?:net\.minecraft|net\.fabricmc|com\.mojang|org\.quiltmc)"
    r"(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+)"
)
_MISSING_PACKAGE_RE = re.compile(
    r"\bpackage\s+([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+)"
    r"\s+does\s+not\s+exist",
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
    semantic_fresh_java_target: bool = False,
) -> bool:
    """Decide whether mutation needs speculative pre-implementation retrieval.

    A compiler can verify a candidate, but cannot supply missing implementation
    facts for a fresh task. Honor explicit fresh-evidence policy before its first
    mutation, even when the host has already materialized the target scaffold.
    Existing-source edits retain compile-first feedback when appropriate.
    """

    if authored_workspace_refresh:
        return True
    if role not in {"coder", "coder_safe"} or host_grounded:
        return False
    if semantic_fresh_java_target and router_requires_fresh_evidence:
        return True
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


def verifier_recovery_query(
    errors: Sequence[Mapping[str, Any]],
    *,
    target_path: str | None = None,
) -> str:
    """Build a deterministic API-recovery query from verifier diagnostics."""

    target = str(target_path or "").replace("\\", "/").strip()
    target_symbol = target.rsplit("/", 1)[-1].rsplit(".", 1)[0] if target else ""
    terms: list[str] = []

    def add(value: str) -> None:
        token = str(value or "").strip()
        if not token or token == target_symbol or token in terms:
            return
        terms.append(token)

    for item in errors:
        if not isinstance(item, Mapping):
            continue
        message = str(item.get("message") or "").strip()
        if not message:
            continue
        for match in _IMPORT_API_RE.finditer(message):
            add(match.group(1))
        for match in _QUALIFIED_API_RE.finditer(message):
            add(match.group(1))
        for match in _MISSING_PACKAGE_RE.finditer(message):
            add(match.group(1))
        for match in _SYMBOL_LINE_RE.finditer(message):
            add(match.group(1))

    if not terms:
        for item in errors:
            if not isinstance(item, Mapping):
                continue
            message = str(item.get("message") or "").strip()
            for raw_line in message.splitlines():
                line = raw_line.strip()
                if not line or line.casefold() in {
                    "cannot find symbol",
                    "incompatible types",
                }:
                    continue
                add(line[:160])
                break
            if terms:
                break

    return " ".join(terms[:8])[:512]


def normalize_recovery_evidence_calls(
    calls: Sequence[Any],
    *,
    errors: Sequence[Mapping[str, Any]],
    target_path: str | None,
    repair_route: str | None,
) -> tuple[Any, ...] | None:
    """Bind RECOVER retriever queries to the active verifier diagnostics."""

    if not repair_route_requires_retrieval(repair_route) or not calls:
        return None
    query = verifier_recovery_query(errors, target_path=target_path)
    if not query:
        return None

    changed = False
    normalized: list[Any] = []
    for call in calls:
        name = str(getattr(call, "name", "") or "").strip()
        raw_arguments = getattr(call, "arguments", None)
        if not isinstance(raw_arguments, Mapping):
            normalized.append(call)
            continue
        arguments = dict(raw_arguments)

        if name in {"search_project_rag", "search_code_rag", "java_workspace_symbols"}:
            if str(arguments.get("query") or "").strip() != query:
                arguments["query"] = query
                changed = True
        elif (
            name == "external_mcp_call"
            and str(arguments.get("capability") or "").strip() == "source_search"
        ):
            nested = arguments.get("arguments")
            nested_arguments = (
                {
                    str(key): value
                    for key, value in nested.items()
                    if str(key) in {"limit", "mapping", "query", "searchType", "version"}
                }
                if isinstance(nested, Mapping)
                else {}
            )
            primary = query.split(" ", 1)[0]
            if str(nested_arguments.get("query") or "").strip() != primary:
                nested_arguments["query"] = primary
            if not str(nested_arguments.get("searchType") or "").strip():
                nested_arguments["searchType"] = "class"
            if nested_arguments != (dict(nested) if isinstance(nested, Mapping) else {}):
                arguments["arguments"] = nested_arguments
                changed = True

        if arguments != dict(raw_arguments):
            normalized.append(
                replace(
                    call,
                    arguments=arguments,
                    raw_arguments=json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
        else:
            normalized.append(call)

    return tuple(normalized) if changed else None


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
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes, bytearray))
    ):
        required = ()
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
    "normalize_recovery_evidence_calls",
    "recovery_evidence_frontier",
    "repair_evidence_route_for_errors",
    "repair_route_requires_retrieval",
    "semantic_fresh_java",
    "verifier_diagnostic_bundle",
    "verifier_recovery_query",
]
