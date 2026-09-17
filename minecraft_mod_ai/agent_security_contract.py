from __future__ import annotations

"""Agent security boundaries that do not redefine execution contracts at runtime.

RAG usability is owned canonically by ``model_router._usable_rag_result``.  This module
only installs the scoped repair-memory boundary; it must never replace model-router
callables or reinterpret retrieval evidence.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALLED = False
_MEMORY_MARKER = "_mmm_scoped_sanitized_repair_memory_v3"


def install(
    *,
    pre_design_rag_module: Any,
    agentic_research_module: Any,
    model_router_module: Any,
    agentic_optimization_module: Any | None = None,
    agent_tool_runtime_module: Any | None = None,
) -> None:
    """Install repair-memory hardening without mutating model execution policy."""

    global _INSTALLED
    if _INSTALLED:
        return

    # Retain the bootstrap-facing signature while ownership is consolidated.  These
    # modules are deliberately not mutated here.
    del pre_design_rag_module, agentic_research_module, model_router_module

    if agentic_optimization_module is None:
        from . import agentic_optimization_contract as agentic_optimization_module
    if agent_tool_runtime_module is None:
        from . import agent_tool_runtime as agent_tool_runtime_module

    _install_repair_memory_boundary(
        agentic_optimization_module,
        agent_tool_runtime_module,
    )
    _INSTALLED = True


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


def _sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["install"]
