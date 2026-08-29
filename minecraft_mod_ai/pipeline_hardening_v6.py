from __future__ import annotations

"""Phase-aware pre-design research hardening.

Pre-design intentionally runs before the Minecraft/Fabric target is frozen.  A model
may therefore report ``sufficient=false`` solely because exact target-bound evidence
has been deferred until target selection.  That condition must not abort the whole
planner when the current phase already has grounded evidence.

This patch keeps the existing fail-closed behaviour for ungrounded or genuinely empty
research.  It only recovers the narrow circular-dependency case: target resolution is
explicitly deferred, the bounded synthesis rejects the group as insufficient, and the
group already contains at least one host-grounded claim.
"""

from collections.abc import Mapping, Sequence
from typing import Any

_INSTALLED = False


def _base_ref(value: Any) -> str:
    text = str(value or "").strip()
    marker = "#synthesis-"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text


def _context_text(kwargs: Mapping[str, Any], group: Sequence[Mapping[str, Any]]) -> str:
    parts = [
        str(kwargs.get("prompt") or ""),
        repr(kwargs.get("domain") or {}),
        repr(group),
    ]
    return "\n".join(parts).lower().replace("_", " ")


def _target_resolution_is_deferred(
    kwargs: Mapping[str, Any],
    group: Sequence[Mapping[str, Any]],
) -> bool:
    domain = kwargs.get("domain")
    if isinstance(domain, Mapping):
        if domain.get("target_frozen") is False:
            return True
        state = str(domain.get("target_state") or "").strip().lower()
        if state in {"unfrozen", "deferred", "pending", "target-neutral", "target_neutral"}:
            return True

    text = _context_text(kwargs, group)
    explicit_markers = (
        "target frozen=false",
        'target frozen": false',
        "target frozen': false",
        "target not frozen",
        "target is not frozen",
        "target-neutral",
        "target neutral",
        "defer until target freeze",
        "deferred until target freeze",
        "after target freeze",
        "once target is frozen",
    )
    return any(marker in text for marker in explicit_markers)


def _page_ref(note: Mapping[str, Any]) -> str:
    direct = _base_ref(note.get("page_ref"))
    if direct:
        return direct
    fragment = note.get("evidence_fragment")
    if isinstance(fragment, Mapping):
        return _base_ref(fragment.get("page_ref"))
    return ""


def _grounded_merge(group: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not group:
        return None

    claims: list[dict[str, Any]] = []
    procedures: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, tuple[str, ...]]] = set()
    seen_procedures: set[tuple[str, tuple[str, ...]]] = set()

    for note in group:
        if not isinstance(note, Mapping):
            continue
        parent_ref = _page_ref(note)

        for raw_claim in note.get("claims", ()):
            if not isinstance(raw_claim, Mapping):
                continue
            claim = dict(raw_claim)
            claim_text = str(
                claim.get("claim_text")
                or claim.get("text")
                or claim.get("claim")
                or ""
            ).strip()
            if not claim_text:
                continue
            refs = tuple(
                dict.fromkeys(
                    ref
                    for ref in (_base_ref(value) for value in claim.get("evidence_refs", ()))
                    if ref
                )
            )
            if not refs and parent_ref:
                refs = (parent_ref,)
            if not refs:
                continue
            claim["evidence_refs"] = list(refs)
            key = (claim_text, refs)
            if key not in seen_claims:
                seen_claims.add(key)
                claims.append(claim)

        for raw_procedure in note.get("procedures", ()):
            if not isinstance(raw_procedure, Mapping):
                continue
            procedure = dict(raw_procedure)
            procedure_text = str(
                procedure.get("procedure_text")
                or procedure.get("text")
                or procedure.get("procedure")
                or procedure.get("name")
                or ""
            ).strip()
            refs = tuple(
                dict.fromkeys(
                    ref
                    for ref in (_base_ref(value) for value in procedure.get("evidence_refs", ()))
                    if ref
                )
            )
            if not refs and parent_ref:
                refs = (parent_ref,)
            if refs:
                procedure["evidence_refs"] = list(refs)
            key = (procedure_text, refs)
            if key not in seen_procedures:
                seen_procedures.add(key)
                procedures.append(procedure)

    # Do not convert a genuinely empty/ungrounded research result into success.
    if not claims:
        return None

    base = dict(next(note for note in group if isinstance(note, Mapping)))
    base["claims"] = claims
    if procedures or "procedures" in base:
        base["procedures"] = procedures
    base["sufficient"] = True

    # Keep the model's unresolved items as diagnostics.  They are non-fatal only for
    # this phase; target-bound research still has to resolve them after target freeze.
    gaps = list(base.get("gaps", ())) if isinstance(base.get("gaps", ()), (list, tuple)) else []
    marker = "deferred_until_target_freeze"
    if marker not in gaps:
        gaps.append(marker)
    base["gaps"] = gaps
    return base


def install_pipeline_hardening_v6() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import agentic_pre_design_rag as rag

    original = rag._synthesize_group_with_recovery
    if getattr(original, "_mmm_phase_sufficient", False):
        _INSTALLED = True
        return

    bounded_error = getattr(rag, "_BoundedResearchOutputError", None)

    def phase_aware(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        group = kwargs.get("group")
        if group is None and len(args) >= 6:
            group = args[5]
        normalized_group = tuple(
            note for note in (group or ()) if isinstance(note, Mapping)
        )

        try:
            return original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve the original failure otherwise
            is_bounded_failure = (
                bounded_error is not None and isinstance(exc, bounded_error)
            ) or type(exc).__name__ == "_BoundedResearchOutputError"
            if not is_bounded_failure:
                raise
            if "sufficient=false" not in str(exc).lower():
                raise
            if not _target_resolution_is_deferred(kwargs, normalized_group):
                raise

            merged = _grounded_merge(normalized_group)
            if merged is None:
                # No grounded claim means this is not merely a phase-boundary issue.
                raise
            return [merged]

    phase_aware._mmm_phase_sufficient = True  # type: ignore[attr-defined]
    rag._synthesize_group_with_recovery = phase_aware
    _INSTALLED = True
