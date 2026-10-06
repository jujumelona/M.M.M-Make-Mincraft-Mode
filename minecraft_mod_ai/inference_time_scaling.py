from __future__ import annotations

"""Inference-time scaling policies for the surviving host-owned repair search."""

import os
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .procedure_trace import sequence_actions
from .trajectory_memory import relevant_trajectories, synthesize_temporary_skill
from .trajectory_record_integrity import derive_levels, record_strong_skill_eligible

_VERIFIER_MARKER = "__mmm_verifier_first_tournament_v1__"
_WIDTH_MARKER = "__mmm_repair_test_time_width_v1__"
_MEMORY_MARKER = "__mmm_source_free_repair_memory_v1__"


def harden_runtime() -> None:
    from . import agentic_optimization_contract as repair_search

    _install_search_width(repair_search)
    _install_verifier_first_ranking(repair_search)
    _install_source_free_repair_memory(repair_search)


def _install_search_width(repair_search: Any) -> None:
    current_repair_count = repair_search._repair_candidate_count
    if getattr(current_repair_count, _WIDTH_MARKER, False):
        return

    @wraps(current_repair_count)
    def repair_candidate_count(
        self: Any,
        evidence: Mapping[str, Any],
        memory: Sequence[Mapping[str, Any]],
    ) -> int:
        if getattr(self, "_mmm_agentic_root", None) is None:
            return 1
        base = int(current_repair_count(self, evidence, memory))
        mode = _scaling_mode()
        if mode == "off":
            return base
        desired = _env_width("MMM_REPAIR_SEARCH_WIDTH", 2)
        diagnostics = evidence.get("diagnostics", {})
        values = (
            diagnostics.get("diagnostics", [])
            if isinstance(diagnostics, Mapping)
            else []
        )
        build = evidence.get("build", {})
        build_failed = (
            isinstance(build, Mapping)
            and str(build.get("status", "")).upper() == "FAIL"
        )
        if mode == "on" or build_failed or bool(values):
            return max(base, desired)
        return base

    setattr(repair_candidate_count, _WIDTH_MARKER, True)
    repair_search._repair_candidate_count = repair_candidate_count
    repair_search._STRATEGIES = (
        "minimal_local_fix_fresh",
        "api_contract_verified_trajectory_replay",
        "dependency_version_verified_failure_counterfactual",
    )


def _install_verifier_first_ranking(repair_search: Any) -> None:
    current_repair_verify = repair_search._verify_repair_candidate
    if getattr(current_repair_verify, _VERIFIER_MARKER, False):
        return

    @wraps(current_repair_verify)
    def verify_repair_candidate(
        self: Any,
        root: Any,
        operations: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Any],
    ):
        score, verifier = current_repair_verify(
            self, root, operations, evidence
        )
        tier = _verifier_tier(verifier)
        bounded = max(-100_000.0, min(100_000.0, float(score)))
        verifier = {
            **dict(verifier),
            "selection_policy": "verifier_first_then_patch_locality",
            "verifier_tier": tier,
        }
        return tier * 1_000_000.0 + bounded, verifier

    setattr(verify_repair_candidate, _VERIFIER_MARKER, True)
    repair_search._verify_repair_candidate = verify_repair_candidate


def _install_source_free_repair_memory(repair_search: Any) -> None:
    current_read = repair_search._read_memory
    if getattr(current_read, _MEMORY_MARKER, False):
        return

    def read_memory(
        root: Any,
        signature: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        if root is None:
            return []
        try:
            rows = relevant_trajectories(
                root,
                signature,
                task_class="repair",
                router=None,
                limit=max(1, min(6, limit)),
                current_context=None,
            )
            skill = synthesize_temporary_skill(
                signature, rows, task_class="repair"
            )
        except Exception:
            rows = []
            skill = None
        verified: list[dict[str, Any]] = []
        for rank, row in enumerate(rows):
            derived = derive_levels(row)
            verification = row.get("verification")
            verification = (
                verification
                if isinstance(verification, Mapping)
                else {}
            )
            verified.append(
                {
                    "similarity": round(max(0.50, 0.78 - 0.04 * rank), 6),
                    "memory_type": "verifier_qualified_trajectory",
                    "trajectory_id": str(row.get("trajectory_id", "")),
                    "verification_level": str(
                        verification.get("level", "L0")
                    ),
                    "verified_success": record_strong_skill_eligible(row),
                    "verified_failure": bool(
                        derived and derived.get("verified_failure") is True
                    ),
                    "procedure_actions": list(
                        sequence_actions(
                            row.get("procedure")
                            if isinstance(row.get("procedure"), Mapping)
                            else None
                        )
                    ),
                    "failure_signature": (
                        " ".join(
                            str(row.get("error_signature", "")).split()
                        )[:360]
                        if bool(
                            derived
                            and derived.get("verified_failure") is True
                        )
                        else ""
                    ),
                    "temporary_skill": skill if rank == 0 else None,
                    "rule": (
                        "Source-free procedure memory only; current hashes, "
                        "diagnostics and exact source remain authoritative."
                    ),
                }
            )
        verified.sort(
            key=lambda item: (
                -float(item.get("similarity", 0.0) or 0.0),
                str(
                    item.get("trajectory_id")
                    or item.get("signature_sha256")
                    or ""
                ),
            )
        )
        return verified[:limit]

    setattr(read_memory, _MEMORY_MARKER, True)
    repair_search._read_memory = read_memory


def _verifier_tier(verifier: Mapping[str, Any]) -> int:
    status = str(verifier.get("jdt_status", "NOT_RUN")).upper()
    raw_errors = verifier.get("jdt_error_count")
    try:
        errors = int(raw_errors) if raw_errors is not None else None
    except (TypeError, ValueError):
        errors = None
    if (
        errors == 0
        and status
        not in {"NOT_RUN", "UNAVAILABLE", "VERIFIER_ERROR", "FAIL"}
    ):
        return 4
    if status == "NOT_RUN":
        return 2
    if status == "UNAVAILABLE":
        return 1
    if status == "VERIFIER_ERROR":
        return 0
    if errors is not None and errors > 0:
        return -1
    return 1


def _scaling_mode() -> str:
    value = os.environ.get("MMM_TEST_TIME_SCALING", "auto").strip().lower()
    mode = value if value in {"auto", "on", "off"} else "auto"
    if mode != "auto":
        return mode
    try:
        from .llama_parallel_runtime_contract import _active_parallelism

        slots = int(_active_parallelism())
    except Exception:
        slots = 1
    return "auto" if slots > 1 else "off"


def _env_width(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(3, value))


__all__ = ["harden_runtime"]
