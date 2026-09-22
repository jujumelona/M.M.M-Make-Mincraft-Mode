from __future__ import annotations

"""Trusted-baseline recovery for fresh compile-backed Java generation."""

import hashlib
from dataclasses import replace
from typing import Any

from .root_cause_trace import emit_root_cause
from .source_mutation_contract import mutation_payload_applied


def _path(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./").strip()


def capture_baseline(state: Any, target: str, source: str) -> None:
    """Freeze the host materialized source before the first model mutation."""
    if (
        getattr(state, "trusted_materialized_baseline", None) is None
        and not bool(getattr(state, "workspace_changed", False))
        and target
        and isinstance(source, str)
        and source
    ):
        state.trusted_materialized_baseline = (_path(target), source, False)


def trusted_baseline(state: Any, context: Any) -> str | None:
    value = getattr(state, "trusted_materialized_baseline", None)
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or context is None
        or _path(value[0]) != _path(getattr(context, "target_path", None))
        or not isinstance(value[1], str)
    ):
        return None
    return value[1]


def rebase_invalid_api_candidate(
    state: Any,
    runtime: Any,
    *,
    stage: str,
    fresh_java_target: bool,
) -> bool:
    """Discard one API-contaminated first attempt and regenerate from trusted scaffold."""
    value = getattr(state, "trusted_materialized_baseline", None)
    context = getattr(state, "mutation_context", None)
    if (
        not fresh_java_target
        or str(getattr(state, "repair_evidence_route", "") or "") != "official_api"
        or str(getattr(state, "validation_status", "") or "") != "FAIL"
        or not isinstance(value, tuple)
        or len(value) != 3
        or bool(value[2])
        or context is None
    ):
        return False

    path, baseline, _rebased = value
    current = getattr(context, "source_body", None)
    if (
        not isinstance(baseline, str)
        or not baseline
        or not isinstance(current, str)
        or current == baseline
        or _path(path) != _path(getattr(context, "target_path", None))
    ):
        return False

    arguments = {
        "operation": "replace_exact",
        "path": _path(path),
        "old": current,
        "new": baseline,
        "count": 1,
    }
    result = runtime.call(stage, "apply_source_edit", arguments)
    if not mutation_payload_applied(
        "apply_source_edit",
        {"ok": True, "result": result},
    ):
        raise RuntimeError(
            "COMPILE_BASELINE_REBASE_FAILED: host could not restore the trusted scaffold"
        )

    with state._lock:
        state.trusted_materialized_baseline = (_path(path), baseline, True)
        state.mutation_context = replace(
            context,
            source_body=baseline,
            is_new_file=False,
            evidence_source="workspace_existing_target",
            base_revision_sha=hashlib.sha256(baseline.encode("utf-8")).hexdigest(),
        )
        state.validation_status = "PENDING"
        state.repair_guidance_fingerprint = None
        state.repair_baseline_error_count = None
        state.repair_baseline_errors = ()
        state.repair_baseline_target_diagnostics = ()
        state.repair_baseline_fingerprint = None
        state.repair_previous_source = None
        state.repair_previous_path = None
        state.last_verifier_quality = None
        state.workspace_changed = False
        state.preserved_existing_source = False
        state.semantic_fixed_point = False
        state.unapplied_mutation_fixed_point = False
        state.seen_no_progress_digests.clear()
        state.no_progress_streak = 0
        state.last_failure_reason = None

    emit_root_cause(
        "compile_candidate_rebased_to_trusted_scaffold",
        stage=stage,
        operation="apply_source_edit",
        gate="compile_recovery",
        result="PASS",
        reason=(
            "fresh Java candidate used incompatible platform APIs; restored the "
            "pre-mutation host scaffold before evidence-guided regeneration"
        ),
        details={
            "target_path": _path(path),
            "diagnostic_count": len(getattr(state, "latest_verifier_errors", ()) or ()),
            "repair_evidence_route": getattr(state, "repair_evidence_route", None),
        },
    )
    return True


__all__ = ["capture_baseline", "rebase_invalid_api_candidate", "trusted_baseline"]
