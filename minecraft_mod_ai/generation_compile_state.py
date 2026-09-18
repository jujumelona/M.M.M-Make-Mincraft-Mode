from __future__ import annotations

"""State transition helper for compile-backed Java generation."""

from collections.abc import Mapping
from typing import Any

from .root_cause_trace import emit_root_cause


def verify_compile_backed_java(
    runtime: Any,
    state: Any,
    *,
    stage: str,
) -> tuple[str, Mapping[str, Any]]:
    context = state.mutation_context
    target_path = str(getattr(context, "target_path", "") or "").strip()
    if not target_path:
        raise RuntimeError("compile-backed generation has no pinned target path")

    receipt = runtime.call(
        stage,
        "target_compile",
        {"target_path": target_path},
    )
    if not isinstance(receipt, Mapping):
        raise RuntimeError("target_compile returned a non-mapping receipt")

    raw = str(receipt.get("status") or "").strip().upper()
    if raw == "PASS":
        status = "PASS"
    elif raw == "FAIL":
        status = "FAIL"
    else:
        status = "UNAVAILABLE"

    changed = state.record_verification(
        "target_compile",
        {"result": dict(receipt)},
        status,
    )
    emit_root_cause(
        "compile_backed_generation_adjudicated",
        stage=stage,
        operation="target_compile",
        gate="generation_verifier",
        result="PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "SKIP",
        reason=str(receipt.get("reason") or raw or status),
        details={
            "status": status,
            "target_path": target_path,
            "changed": changed,
            "diagnostic_count": len(receipt.get("diagnostics") or ()),
        },
    )
    return status, receipt


__all__ = ["verify_compile_backed_java"]
