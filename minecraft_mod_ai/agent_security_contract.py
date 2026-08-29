from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALL_MARKER = "_mmm_agent_security_contract_v5"
_SLICE_MARKER = "_mmm_scoped_forced_rag_receipt_v1"
_MEMORY_MARKER = "_mmm_scoped_sanitized_repair_memory_v3"
_TERMINAL_RAG_WARNINGS = frozenset({"required_metadata_mismatch"})
_LEGACY_EVIDENCE_KEYS = frozenset(
    {"hits", "results", "matches", "documents", "chunks", "sources"}
)


def install(
    *,
    pre_design_rag_module: Any,
    agentic_research_module: Any,
    model_router_module: Any,
    agentic_optimization_module: Any | None = None,
    agent_tool_runtime_module: Any | None = None,
) -> None:
    """Harden active agent boundaries without reviving retired research wrappers."""

    if getattr(model_router_module, _INSTALL_MARKER, False):
        return

    if agentic_optimization_module is None:
        from . import agentic_optimization_contract as agentic_optimization_module
    if agent_tool_runtime_module is None:
        from . import agent_tool_runtime as agent_tool_runtime_module

    # Pre-design collection is now owned directly by pre_design_research_pipeline.
    # Security scopes the active research slice in place; it must not depend on or
    # recreate the retired harden_pre_design_research monkeypatch.
    _install_scoped_domain_receipt(pre_design_rag_module, agentic_research_module)
    model_router_module._usable_rag_result = usable_rag_result
    _install_repair_memory_boundary(
        agentic_optimization_module,
        agent_tool_runtime_module,
    )

    setattr(model_router_module, _INSTALL_MARKER, True)


def _install_scoped_domain_receipt(pre_design_rag_module: Any, agentic_module: Any) -> None:
    current = agentic_module._domain_evidence_slice
    if getattr(current, _SLICE_MARKER, False):
        return

    def scoped_slice(domain_id: str, deterministic: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(current(domain_id, deterministic))
        if isinstance(result.get("evidence_document"), Mapping):
            return result
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
    """Accept host-receipted evidence without discarding real unscored hits.

    A receipt remains authoritative when present. A positive host result count may pair
    with concrete evidence hits when optional reranker scores are unavailable; a zero
    result receipt can never be rescued by stale hits. Observation metadata, truncation
    previews, error text, or arbitrary non-empty dictionaries are not evidence.
    """

    found_receipt = False
    positive_receipt = False
    usable_receipt = False
    legacy_evidence = False

    def visit(item: Any) -> None:
        nonlocal found_receipt, positive_receipt, usable_receipt, legacy_evidence
        if isinstance(item, Mapping):
            receipt = item.get("receipt")
            if isinstance(receipt, Mapping):
                found_receipt = True
                if _positive_receipt(receipt):
                    positive_receipt = True
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
        return usable_receipt or (positive_receipt and legacy_evidence)
    return legacy_evidence


def _receipt_warning_set(receipt: Mapping[str, Any]) -> set[str]:
    warnings = receipt.get("warnings", ())
    if isinstance(warnings, str):
        return {warnings.strip()}
    if isinstance(warnings, Sequence):
        return {str(item).strip() for item in warnings if str(item).strip()}
    return set()


def _positive_receipt(receipt: Mapping[str, Any]) -> bool:
    if _receipt_warning_set(receipt) & _TERMINAL_RAG_WARNINGS:
        return False
    try:
        return int(receipt.get("result_count", 0) or 0) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def _usable_receipt(receipt: Mapping[str, Any]) -> bool:
    if _receipt_warning_set(receipt) & _TERMINAL_RAG_WARNINGS:
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


def _install_repair_memory_boundary(agentic_module: Any, runtime_module: Any) -> None:
    current_write = agentic_module._write_memory
    current_read = agentic_module._read_memory
    if (
        getattr(current_write, _MEMORY_MARKER, False)
        and getattr(current_read, _MEMORY_MARKER, False)
    ):
        return
    sanitizer = getattr(runtime_module, "_sanitize_observation", None)
    small_metadata = getattr(runtime_module, "_small_metadata", None)

    @wraps(current_read)
    def read_scoped_memory(
        root: Any,
        signature: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        _guard_memory_path(agentic_module, root)
        try:
            requested_limit = int(limit)
        except (TypeError, ValueError, OverflowError):
            requested_limit = 4
        if requested_limit <= 0:
            return []
        safe_limit = min(requested_limit, 4)
        rows = current_read(root, signature, limit=safe_limit)
        result: list[dict[str, Any]] = []
        for row in rows[:safe_limit]:
            sanitized = sanitizer(row) if callable(sanitizer) else row
            if not isinstance(sanitized, Mapping):
                continue
            evidence = sanitized.get("evidence", {})
            if callable(small_metadata):
                evidence = small_metadata(evidence)
            elif isinstance(evidence, Mapping):
                evidence = dict(evidence)
            else:
                evidence = {}
            try:
                similarity = float(sanitized.get("similarity", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(similarity) or similarity <= 0.0:
                continue
            result.append(
                {
                    "similarity": round(min(similarity, 1.0), 6),
                    "signature_sha256": str(
                        sanitized.get("signature_sha256", "")
                    )[:96],
                    "evidence": evidence,
                    "repair_pattern": _bounded_repair_pattern(
                        sanitized.get("repair_pattern", ())
                    ),
                    "memory_scope": {
                        "workflow": "repair",
                        "subtask": "diagnostic_repair",
                        "trust": "untrusted_prior_verified_evidence",
                        "can_authorize_tools": False,
                    },
                }
            )
        return result

    @wraps(current_write)
    def write_scoped_memory(root: Any, trace: Mapping[str, Any]) -> None:
        _guard_memory_path(agentic_module, root)
        sanitized = sanitizer(trace) if callable(sanitizer) else dict(trace)
        if not isinstance(sanitized, Mapping):
            return

        signature = str(sanitized.get("signature", ""))[:2048]
        evidence = sanitized.get("evidence", {})
        evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        evidence["memory_scope"] = {
            "workflow": "repair",
            "subtask": "diagnostic_repair",
            "function_error_sha256": _sha_text(signature),
            "promotion_gate": "host_verified_repair_result",
        }

        verifier = sanitized.get("winner_verifier", {})
        verifier = dict(verifier) if isinstance(verifier, Mapping) else {}
        current_write(
            root,
            {
                "signature": signature,
                "evidence": evidence,
                "repair_pattern": _bounded_repair_pattern(
                    sanitized.get("repair_pattern", ())
                ),
                "winner_verifier": verifier,
            },
        )

    setattr(read_scoped_memory, _MEMORY_MARKER, True)
    setattr(write_scoped_memory, _MEMORY_MARKER, True)
    agentic_module._read_memory = read_scoped_memory
    agentic_module._write_memory = write_scoped_memory


def _guard_memory_path(agentic_module: Any, root: Any) -> Path:
    root_path = Path(root).expanduser().resolve()
    path = Path(agentic_module._memory_path(root_path))
    parent = path.parent
    if parent.is_symlink() or path.is_symlink():
        raise RuntimeError("Refusing repair memory access through a symlink.")
    if parent.exists() and not parent.is_dir():
        raise RuntimeError("Repair memory parent is not a directory.")
    resolved_parent = parent.resolve(strict=False)
    try:
        resolved_parent.relative_to(root_path)
    except ValueError as exc:
        raise RuntimeError("Repair memory path escaped the project root.") from exc
    return path


def _bounded_repair_pattern(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    pattern: list[dict[str, Any]] = []
    for item in value[:16]:
        if not isinstance(item, Mapping):
            continue
        pattern.append(
            {
                "operation": str(item.get("operation", ""))[:64],
                "path": str(item.get("path", ""))[:1024],
                "repair_excerpt": str(item.get("repair_excerpt", ""))[:1024],
                "trust": "untrusted_prior_patch_data",
            }
        )
    return pattern


def _nonempty_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
    )


def _sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["install", "usable_rag_result"]
