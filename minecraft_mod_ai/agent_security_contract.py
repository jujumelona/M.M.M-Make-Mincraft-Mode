from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


_INSTALL_MARKER = "_mmm_agent_security_contract_v1"
_SLICE_MARKER = "_mmm_scoped_forced_rag_receipt_v1"
_TERMINAL_RAG_WARNINGS = frozenset({"required_metadata_mismatch"})
_LEGACY_EVIDENCE_KEYS = frozenset(
    {"hits", "results", "matches", "documents", "chunks", "sources"}
)


def install(
    *,
    pre_design_rag_module: Any,
    agentic_research_module: Any,
    model_router_module: Any,
) -> None:
    """Harden already-composed agent RAG boundaries without adding another runtime.

    The package bootstrap still owns retrieval and tool execution. This contract only
    restores the bounded domain receipt that host callers require and replaces the
    permissive RAG-result truthiness fallback with deterministic receipt validation.
    """

    if getattr(model_router_module, _INSTALL_MARKER, False):
        return

    original_harden = pre_design_rag_module.harden_pre_design_research

    def harden(agentic_module: Any) -> None:
        original_harden(agentic_module)
        _install_scoped_domain_receipt(pre_design_rag_module, agentic_module)

    harden.__wrapped__ = original_harden  # type: ignore[attr-defined]
    pre_design_rag_module.harden_pre_design_research = harden

    # The production research module was composed before this post-bootstrap contract
    # is installed, so harden its current outermost evidence slice once as well.
    _install_scoped_domain_receipt(pre_design_rag_module, agentic_research_module)
    model_router_module._usable_rag_result = usable_rag_result
    setattr(model_router_module, _INSTALL_MARKER, True)


def _install_scoped_domain_receipt(pre_design_rag_module: Any, agentic_module: Any) -> None:
    current = agentic_module._domain_evidence_slice
    if getattr(current, _SLICE_MARKER, False):
        return

    def scoped_slice(domain_id: str, deterministic: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(current(domain_id, deterministic))
        if isinstance(result.get("forced_project_rag"), Mapping):
            return result

        context = getattr(pre_design_rag_module, "_FORCED_RAG_CONTEXT", None)
        forced = context.get() if context is not None else None
        if not isinstance(forced, Mapping):
            forced = deterministic.get("forced_project_rag")
        receipt = _scoped_forced_receipt(domain_id, forced)
        if receipt is not None:
            result["forced_project_rag"] = receipt
        return result

    scoped_slice.__wrapped__ = current  # type: ignore[attr-defined]
    setattr(scoped_slice, _SLICE_MARKER, True)
    agentic_module._domain_evidence_slice = scoped_slice


def _scoped_forced_receipt(
    domain_id: str,
    forced: Any,
) -> dict[str, Any] | None:
    if not isinstance(forced, Mapping):
        return None
    receipt = {str(key): value for key, value in forced.items() if key != "domains"}
    domains = forced.get("domains")
    if isinstance(domains, list):
        selected = next(
            (
                item
                for item in domains
                if isinstance(item, Mapping)
                and str(item.get("domain_id", "")).strip() == domain_id
            ),
            None,
        )
        if isinstance(selected, Mapping):
            receipt.update({str(key): value for key, value in selected.items()})
    return receipt


def usable_rag_result(value: Any) -> bool:
    """Accept RAG evidence only when its host receipt or known evidence pack is usable.

    A receipt is authoritative when present. Observation metadata, truncation previews,
    error text, or arbitrary non-empty dictionaries are never promoted to evidence.
    This prevents a small model from finalizing merely because a tool returned metadata.
    """

    found_receipt = False
    usable_receipt = False
    legacy_evidence = False

    def visit(item: Any) -> None:
        nonlocal found_receipt, usable_receipt, legacy_evidence
        if isinstance(item, Mapping):
            receipt = item.get("receipt")
            if isinstance(receipt, Mapping):
                found_receipt = True
                if _usable_receipt(receipt):
                    usable_receipt = True
            for key, child in item.items():
                if (
                    str(key).strip().lower() in _LEGACY_EVIDENCE_KEYS
                    and _nonempty_sequence(child)
                ):
                    legacy_evidence = True
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    if found_receipt:
        return usable_receipt
    return legacy_evidence


def _usable_receipt(receipt: Mapping[str, Any]) -> bool:
    warnings = receipt.get("warnings", ())
    if isinstance(warnings, str):
        warning_set = {warnings.strip()}
    elif isinstance(warnings, Sequence):
        warning_set = {str(item).strip() for item in warnings if str(item).strip()}
    else:
        warning_set = set()
    if warning_set & _TERMINAL_RAG_WARNINGS:
        return False

    try:
        result_count = int(receipt.get("result_count", 0) or 0)
        coverage = float(receipt.get("coverage_score", 0.0) or 0.0)
        relevance = float(receipt.get("relevance_score", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        result_count > 0
        and math.isfinite(coverage)
        and math.isfinite(relevance)
        and coverage > 0.0
        and relevance > 0.0
    )


def _nonempty_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
    )


__all__ = ["install", "usable_rag_result"]
