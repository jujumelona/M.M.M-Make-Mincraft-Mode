from __future__ import annotations

"""Deterministic repair-evidence routing for small coding models.

The base RepairEngine already retrieves task-local project RAG.  This module decides
whether a diagnostic also needs fresh target documentation.  The decision is host-owned
and conservative: syntax/local-symbol failures stay local when the project evidence
covers them, while loader/API/mapping/dependency/Mixin failures request official target
evidence.  The small model receives the route receipt instead of having to decide what
to research.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

_SCHEMA = "mmm/repair-evidence-route-v1"
_EXTERNAL_MARKERS = (
    "net.minecraft",
    "net.fabricmc",
    "fabric api",
    "fabric-api",
    "fabric loader",
    "fabric-loader",
    "neoforge",
    "net.neoforged",
    "forge",
    "yarn",
    "mapping",
    "mappings",
    "loom",
    "org.spongepowered.asm.mixin",
    "mixin",
    "access widener",
    "accesswidener",
    "registry",
    "registries",
    "identifier",
    "resource location",
)
_DEPENDENCY_MARKERS = (
    "could not resolve",
    "could not find",
    "failed to resolve",
    "dependency",
    "dependencies",
    "mod resolution",
    "incompatible mod set",
    "requires version",
    "requires any version",
    "no matching variant",
    "classnotfoundexception",
    "noclassdeffounderror",
    "nosuchmethoderror",
    "nosuchfielderror",
)
_LOCAL_ONLY_MARKERS = (
    "';' expected",
    "expected ';'",
    "reached end of file while parsing",
    "illegal start of expression",
    "illegal start of type",
    "not a statement",
    "duplicate class",
    "cyclic inheritance",
    "has private access",
    "has protected access",
    "is already defined",
    "variable might not have been initialized",
)
_API_SIGNATURE_MARKERS = (
    "cannot be applied to given types",
    "no suitable method found",
    "method does not override or implement",
    "incompatible types",
    "package does not exist",
    "cannot access",
)
_SYMBOL_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,}")


def _texts(diagnostic: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("messages", "tasks", "exceptions", "symbols", "files"):
        raw = diagnostic.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.extend(str(value) for value in raw if str(value).strip())
    return tuple(values)


def _local_rag_text(base_context: Mapping[str, Any]) -> str:
    rag = base_context.get("rag")
    hits = rag.get("hits") if isinstance(rag, Mapping) else None
    parts: list[str] = []
    for hit in hits if isinstance(hits, list) else ():
        if not isinstance(hit, Mapping):
            continue
        parts.append(str(hit.get("path") or ""))
        parts.append(str(hit.get("text") or ""))
    return "\n".join(parts).casefold()


def _uncovered_symbols(
    diagnostic: Mapping[str, Any],
    base_context: Mapping[str, Any],
) -> tuple[str, ...]:
    symbols = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in diagnostic.get("symbols", ())
            if str(value).strip()
        )
    )
    if not symbols:
        return ()
    local = _local_rag_text(base_context)
    if not local:
        return symbols
    return tuple(symbol for symbol in symbols if symbol.casefold() not in local)


def classify_repair_evidence_route(
    diagnostic: Mapping[str, Any],
    base_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the smallest evidence route that can explain the current failure."""

    text = "\n".join(_texts(diagnostic)).casefold()
    uncovered = _uncovered_symbols(diagnostic, base_context)
    compatibility = any(marker in text for marker in _DEPENDENCY_MARKERS) or "mixin" in text
    external_contract = any(marker in text for marker in _EXTERNAL_MARKERS)
    api_signature = any(marker in text for marker in _API_SIGNATURE_MARKERS)
    local_only = any(marker in text for marker in _LOCAL_ONLY_MARKERS)
    unresolved_symbol = "cannot find symbol" in text and bool(uncovered)

    fresh_official = bool(
        compatibility
        or external_contract
        or (api_signature and not local_only)
        or unresolved_symbol
    )
    if compatibility:
        route = "compatibility"
    elif fresh_official:
        route = "official_api"
    else:
        route = "project_local"

    reasons: list[str] = []
    if compatibility:
        reasons.append("dependency_or_runtime_contract")
    if external_contract:
        reasons.append("minecraft_loader_api_marker")
    if api_signature:
        reasons.append("api_signature_shape")
    if unresolved_symbol:
        reasons.append("symbol_missing_from_local_rag")
    if local_only and not fresh_official:
        reasons.append("local_language_or_access_error")
    if not reasons:
        reasons.append("local_project_evidence_is_sufficient_first")

    return {
        "schema_version": _SCHEMA,
        "route": route,
        "project_rag_required": True,
        "fresh_official_required": fresh_official,
        "uncovered_symbols": list(uncovered[:12]),
        "reasons": reasons,
        "small_model_policy": {
            "retrieval_owner": "host",
            "model_must_not_choose_retriever": True,
            "prefer_exact_local_evidence_before_external": True,
            "do_not_guess_external_api_when_official_required": True,
        },
    }


def official_query_prefix(route: Mapping[str, Any]) -> str:
    kind = str(route.get("route") or "")
    if kind == "compatibility":
        return "exact Minecraft loader dependency compatibility contract"
    return "exact Minecraft loader API signature compile repair"


__all__ = ["classify_repair_evidence_route", "official_query_prefix"]
