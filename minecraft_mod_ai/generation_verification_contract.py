from __future__ import annotations

"""Pure validation contract for generation-time verifier receipts."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

GENERATION_VERIFICATION_SCHEMA = "mmm/generation-verification-v1"
GENERATION_VERIFICATION_AUTHORITY = "generation_tool_loop"


def _normalized_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    while "//" in text:
        text = text.replace("//", "/")
    if not text:
        return ""
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        return text
    return path.as_posix()


def _normalized_gate(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").strip().casefold(),
    ).strip("_")


def _touched_paths_sha256(paths: Sequence[str]) -> str:
    normalized = tuple(sorted(set(paths)))
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def classify_generation_verification(
    *,
    source_status: Any,
    receipt: Any,
    touched_paths: Sequence[Any],
    required_gates: Sequence[Any],
) -> dict[str, Any]:
    """Classify exact host verifier evidence without starting another verifier."""

    normalized_touched = tuple(
        path
        for raw in touched_paths
        if (path := _normalized_path(raw))
    )
    touched_set = frozenset(normalized_touched)
    normalized_gates = tuple(
        gate
        for raw in required_gates
        if (gate := str(raw or "").strip())
    )
    normalized_gate_keys = frozenset(
        key
        for gate in normalized_gates
        if (key := _normalized_gate(gate))
    )
    source = str(source_status or "").strip().upper()
    candidate_receipt = receipt if isinstance(receipt, Mapping) else {}

    receipt_valid = bool(
        candidate_receipt.get("schema_version") == GENERATION_VERIFICATION_SCHEMA
        and candidate_receipt.get("authority") == GENERATION_VERIFICATION_AUTHORITY
    )
    terminal_status = (
        str(candidate_receipt.get("status") or "").strip().upper()
        if receipt_valid
        else "MISSING"
    )
    validation_status = str(
        candidate_receipt.get("validation_status") or ""
    ).strip().upper()
    termination_reason = str(
        candidate_receipt.get("termination_reason") or ""
    ).strip()
    verification_tool = str(
        candidate_receipt.get("verifier_tool") or ""
    ).strip()
    receipt_target_path = _normalized_path(
        candidate_receipt.get("target_path")
    )
    receipt_target_matches = bool(
        receipt_target_path and receipt_target_path in touched_set
    )
    compile_backed_java = candidate_receipt.get("compile_backed_java") is True
    target_compile_required = "target_compile" in normalized_gate_keys
    project_build_required = "project_build" in normalized_gate_keys
    java_target = receipt_target_path.lower().endswith(".java")
    compile_required = target_compile_required or compile_backed_java
    downstream_required_gate = str(
        candidate_receipt.get("downstream_required_gate") or ""
    ).strip()
    verification_scope = str(
        candidate_receipt.get("verification_scope") or ""
    ).strip().casefold()
    receipt_touched_paths_sha256 = str(
        candidate_receipt.get("touched_paths_sha256") or ""
    ).strip()
    try:
        receipt_touched_path_count = int(
            candidate_receipt.get("touched_path_count")
        )
    except (TypeError, ValueError):
        receipt_touched_path_count = -1
    project_scope_matches = bool(
        verification_scope == "project"
        and normalized_touched
        and receipt_touched_path_count == len(touched_set)
        and receipt_touched_paths_sha256 == _touched_paths_sha256(normalized_touched)
    )

    pass_semantics = bool(
        terminal_status == "PASS"
        and validation_status == "PASS"
        and termination_reason == "VERIFICATION_PASSED"
        and not downstream_required_gate
        and (
            (
                not compile_required
                and not compile_backed_java
            )
            or (
                target_compile_required
                and compile_backed_java
                and java_target
                and verification_tool == "target_compile"
            )
        )
    )
    deferred_semantics = bool(
        terminal_status == "DEFERRED_TO_TARGET_COMPILE"
        and validation_status == "DEFERRED"
        and termination_reason == "VERIFICATION_DEFERRED_TO_TARGET_COMPILE"
        and compile_backed_java
        and target_compile_required
        and java_target
        and verification_tool == "target_compile"
        and downstream_required_gate == "target_compile"
    )
    project_deferred_semantics = bool(
        terminal_status == "DEFERRED_TO_PROJECT_BUILD"
        and validation_status == "DEFERRED"
        and termination_reason == "VERIFICATION_DEFERRED_TO_PROJECT_BUILD"
        and not compile_backed_java
        and project_build_required
        and verification_tool == "project_build"
        and downstream_required_gate == "project_build"
        and project_scope_matches
    )
    receipt_semantics_valid = bool(
        receipt_valid
        and (
            (
                receipt_target_matches
                and (pass_semantics or deferred_semantics)
            )
            or project_deferred_semantics
        )
    )

    if source != "SOURCE_GENERATED" or not receipt_semantics_valid:
        generation_status = "FAIL"
    elif pass_semantics:
        generation_status = "PASS"
    elif deferred_semantics:
        generation_status = "DEFERRED_TO_TARGET_COMPILE"
    elif project_deferred_semantics:
        generation_status = "DEFERRED_TO_PROJECT_BUILD"
    else:
        generation_status = "FAIL"

    if generation_status == "PASS":
        tier = 2
    elif generation_status in {
        "DEFERRED_TO_TARGET_COMPILE",
        "DEFERRED_TO_PROJECT_BUILD",
    }:
        tier = 1
    else:
        tier = 0

    return {
        "generation_status": generation_status,
        "verifier_tier": tier,
        "verification_authority": (
            str(candidate_receipt.get("authority") or "unknown")
            if receipt_valid
            else "unknown"
        ),
        "terminal_verification_status": terminal_status,
        "validation_status": validation_status,
        "termination_reason": termination_reason,
        "verification_tool": verification_tool or None,
        "source_status": source or "MISSING",
        "required_gates": list(normalized_gates),
        "target_compile_required": target_compile_required,
        "project_build_required": project_build_required,
        "downstream_required_gate": downstream_required_gate or None,
        "compile_backed_java": compile_backed_java,
        "java_target": java_target,
        "compile_required": compile_required,
        "verification_scope": verification_scope or None,
        "project_scope_matches": project_scope_matches,
        "receipt_target_path": receipt_target_path or None,
        "receipt_target_matches": receipt_target_matches,
        "receipt_semantics_valid": receipt_semantics_valid,
    }


def generation_verifier_tier(verifier: Mapping[str, Any]) -> int:
    """Read only a fully classified verifier object."""

    if (
        str(verifier.get("verification_authority") or "").strip()
        != GENERATION_VERIFICATION_AUTHORITY
        or str(verifier.get("source_status") or "").strip().upper()
        != "SOURCE_GENERATED"
        or verifier.get("receipt_semantics_valid") is not True
    ):
        return 0
    status = str(verifier.get("generation_status") or "").strip().upper()
    if status == "PASS":
        return 2 if verifier.get("receipt_target_matches") is True else 0
    if (
        status == "DEFERRED_TO_TARGET_COMPILE"
        and verifier.get("receipt_target_matches") is True
        and verifier.get("target_compile_required") is True
        and str(verifier.get("downstream_required_gate") or "").strip()
        == "target_compile"
    ):
        return 1
    if (
        status == "DEFERRED_TO_PROJECT_BUILD"
        and verifier.get("project_scope_matches") is True
        and verifier.get("project_build_required") is True
        and str(verifier.get("downstream_required_gate") or "").strip()
        == "project_build"
    ):
        return 1
    return 0


def candidate_rank_key(
    *,
    score: float,
    candidate_index: int,
    verifier: Mapping[str, Any],
    patch_size: int,
) -> tuple[int, float, int, int]:
    """Verifier authority is lexicographically stronger than any heuristic score."""

    return (
        -generation_verifier_tier(verifier),
        -float(score),
        int(patch_size),
        int(candidate_index),
    )


__all__ = [
    "GENERATION_VERIFICATION_AUTHORITY",
    "GENERATION_VERIFICATION_SCHEMA",
    "candidate_rank_key",
    "classify_generation_verification",
    "generation_verifier_tier",
]
