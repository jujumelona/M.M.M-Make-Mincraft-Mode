from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALL_MARKER = "_mmm_agent_security_contract_v5"
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

    # These arguments remain in the bootstrap-facing signature while callers migrate,
    # but research evidence scoping is natively owned by agentic_research_game_design.
    # Security must not wrap that owner a second time.
    del pre_design_rag_module, agentic_research_module

    if agentic_optimization_module is None:
        from . import agentic_optimization_contract as agentic_optimization_module
    if agent_tool_runtime_module is None:
        from . import agent_tool_runtime as agent_tool_runtime_module

    model_router_module._usable_rag_result = usable_rag_result
    _install_repair_memory_boundary(
        agentic_optimization_module,
        agent_tool_runtime_module,
    )

    setattr(model_router_module, _INSTALL_MARKER, True)


def usable_rag_result(value: Any) -> bool:
    """Accept receipted evidence only when receipt and evidence share a result scope.

    A valid scored receipt is authoritative by itself. A positive receipt whose optional
    reranker scores are absent may fall back to concrete legacy evidence, but only from
    the same mapping subtree as that receipt. Evidence from a sibling result must never
    rescue another result's receipt. If any receipt is present, unreceipted legacy hits
    elsewhere cannot bypass it. Receipt-free legacy result packs remain compatible.
    """

    found_receipt = False

    def visit(item: Any) -> bool:
        nonlocal found_receipt
        if isinstance(item, Mapping):
            receipt = item.get("receipt")
            if isinstance(receipt, Mapping):
                found_receipt = True
                if _usable_receipt(receipt):
                    return True
                if _positive_receipt(receipt) and _mapping_has_legacy_evidence(item):
                    return True

            for child in item.values():
                if visit(child):
                    return True
            return False

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return any(visit(child) for child in item)
        return False

    if visit(value):
        return True
    if found_receipt:
        return False
    return _contains_legacy_evidence(value)


def _mapping_has_legacy_evidence(value: Mapping[str, Any]) -> bool:
    """Return whether one receipted result subtree contains concrete evidence."""

    for key, child in value.items():
        if str(key).strip().lower() == "receipt":
            continue
        if (
            str(key).strip().lower() in _LEGACY_EVIDENCE_KEYS
            and _nonempty_sequence(child)
        ):
            return True
        if isinstance(child, Mapping):
            # A nested receipt starts a new result scope and must be evaluated on its own.
            if isinstance(child.get("receipt"), Mapping):
                continue
            if _mapping_has_legacy_evidence(child):
                return True
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            for nested in child:
                if isinstance(nested, Mapping):
                    if isinstance(nested.get("receipt"), Mapping):
                        continue
                    if _mapping_has_legacy_evidence(nested):
                        return True
    return False


def _contains_legacy_evidence(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                str(key).strip().lower() in _LEGACY_EVIDENCE_KEYS
                and _nonempty_sequence(child)
            ):
                return True
            if _contains_legacy_evidence(child):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_legacy_evidence(child) for child in value)
    return False


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
