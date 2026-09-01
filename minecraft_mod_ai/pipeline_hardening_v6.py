from __future__ import annotations

"""Phase-aware pre-design research hardening.

Pre-design intentionally runs before the Minecraft/Fabric target is frozen. The
canonical research path keeps the raw evidence ledger lossless, but a small local
model can incorrectly treat the absence of target-specific APIs as proof that no
target-neutral design claim is possible. This module recovers only that circular
phase-boundary case.

Normal research remains unchanged. A retry is attempted only when:
* the domain is explicitly target-neutral/deferred,
* canonical synthesis failed specifically with zero grounded claims, and
* page-level extraction can recover claims with valid page provenance.

If page grounding still yields no claim, the original fail-closed error is preserved.
"""

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from functools import wraps
from typing import Any

_INSTALLED = False
_PAGE_GROUNDING_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "mmm_pre_design_page_grounding_context",
    default=None,
)


def _base_ref(value: Any) -> str:
    text = str(value or "").strip()
    marker = "#synthesis-"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text


def _context_text(
    kwargs: Mapping[str, Any],
    group: Sequence[Mapping[str, Any]],
) -> str:
    parts = [
        str(kwargs.get("prompt") or ""),
        repr(kwargs.get("domain") or {}),
        repr(kwargs.get("document") or {}),
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
        if state in {
            "unfrozen",
            "deferred",
            "pending",
            "target-neutral",
            "target_neutral",
        }:
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
        'versioned rag required before design": false',
        "versioned rag required before design': false",
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
    gaps: list[str] = []
    next_queries: list[str] = []

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
                    for ref in (
                        _base_ref(value)
                        for value in claim.get("evidence_refs", ())
                    )
                    if ref
                )
            )
            if not refs and parent_ref:
                refs = (parent_ref,)
            if not refs:
                continue
            # The canonical claim catalog reads the "claim" field.
            claim["claim"] = claim_text
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
                    for ref in (
                        _base_ref(value)
                        for value in procedure.get("evidence_refs", ())
                    )
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

        for raw_gap in note.get("gaps", ()):
            gap = str(raw_gap or "").strip()
            if gap and gap not in gaps:
                gaps.append(gap)
        for raw_query in note.get("next_queries", ()):
            query = str(raw_query or "").strip()
            if query and query not in next_queries:
                next_queries.append(query)

    # Never convert genuinely empty or ungrounded evidence into success.
    if not claims:
        return None

    base = dict(next(note for note in group if isinstance(note, Mapping)))
    base["claims"] = claims
    base["procedures"] = procedures
    base["gaps"] = gaps
    base["next_queries"] = next_queries
    base["sufficient"] = True
    marker = "deferred_until_target_freeze"
    if marker not in base["gaps"]:
        base["gaps"].append(marker)
    return base


def _normalized_group(
    kwargs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    group = kwargs.get("group")
    if not isinstance(group, Sequence) or isinstance(
        group, (str, bytes, bytearray)
    ):
        return ()
    return tuple(item for item in group if isinstance(item, Mapping))


def _is_zero_grounded_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "zero grounded claims" in text
        or "zero grounded claim" in text
        or "no evidence-backed design-relevant claim" in text
    )


def _is_bounded_failure(rag: Any, exc: BaseException) -> bool:
    bounded_error = getattr(rag, "_BoundedResearchOutputError", None)
    return (
        bounded_error is not None
        and isinstance(exc, bounded_error)
    ) or type(exc).__name__ == "_BoundedResearchOutputError"


def _install_synthesis_recovery(rag: Any) -> None:
    original = rag._synthesize_group_with_recovery
    if getattr(original, "_mmm_phase_sufficient_v2", False):
        return

    @wraps(original)
    def phase_aware(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        group = _normalized_group(kwargs)
        deferred = _target_resolution_is_deferred(kwargs, group)

        try:
            result = original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve non-target failures
            if not _is_bounded_failure(rag, exc) or not deferred:
                raise
            merged = _grounded_merge(group)
            if merged is None:
                raise
            return [merged]

        if not deferred:
            return result

        result_notes = tuple(
            item for item in (result or ()) if isinstance(item, Mapping)
        )
        # The canonical recovery routine commonly catches its own bounded error and
        # returns claims=[]/sufficient=false. Inspect that return value instead of
        # waiting for an exception that never reaches this wrapper.
        if any(note.get("sufficient") is not True for note in result_notes):
            merged_result = _grounded_merge(result_notes)
            if merged_result is not None:
                return [merged_result]
            merged_children = _grounded_merge(group)
            if merged_children is not None:
                return [merged_children]
        return result

    phase_aware._mmm_phase_sufficient_v2 = True  # type: ignore[attr-defined]
    rag._synthesize_group_with_recovery = phase_aware


def _install_page_grounding_recovery(rag: Any) -> None:
    original_host = rag._host_page_note
    if getattr(original_host, "_mmm_phase_page_grounding_v2", False):
        return

    @wraps(original_host)
    def phase_host_page_note(
        domain_id: str,
        page: Mapping[str, Any],
    ) -> dict[str, Any]:
        base = original_host(domain_id, page)
        context = _PAGE_GROUNDING_CONTEXT.get()
        if context is None or context.get("domain_id") != domain_id:
            return base

        failures = context["failures"]
        page_ref = str(page.get("page_ref", "")).strip()
        extracted = rag._read_page_losslessly(
            context["agentic_module"],
            context["router"],
            prompt=context["prompt"],
            domain=context["domain"],
            document=context["document"],
            page=page,
            domain_key=context["domain_key"],
            progress_label=f"domain {domain_id} phase-recovery {page_ref}",
            failures=failures,
        )
        merged = _grounded_merge(
            tuple(item for item in extracted if isinstance(item, Mapping))
        )
        if merged is None:
            return base

        merged["domain_id"] = domain_id
        if "evidence_fragment" in base:
            merged["evidence_fragment"] = base["evidence_fragment"]
        if page_ref:
            merged["page_ref"] = page_ref
        return merged

    phase_host_page_note._mmm_phase_page_grounding_v2 = True  # type: ignore[attr-defined]
    rag._host_page_note = phase_host_page_note


def _install_domain_retry(rag: Any) -> None:
    original = rag._research_document_domain
    if getattr(original, "_mmm_phase_domain_retry_v2", False):
        return

    @wraps(original)
    def phase_domain_retry(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if _PAGE_GROUNDING_CONTEXT.get() is not None:
            return original(*args, **kwargs)

        try:
            return original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retry only the exact bounded case
            if not _is_bounded_failure(rag, exc) or not _is_zero_grounded_failure(exc):
                raise

            domain = kwargs.get("domain")
            document = kwargs.get("document")
            if not isinstance(domain, Mapping) or not isinstance(document, Mapping):
                raise
            if not _target_resolution_is_deferred(kwargs, ()):
                raise

            agentic_module = args[0] if len(args) >= 1 else kwargs.get("agentic_module")
            router = args[1] if len(args) >= 2 else kwargs.get("router")
            prompt = str(kwargs.get("prompt") or "")
            domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
            domain_key = rag._domain_checkpoint_key(
                router,
                prompt=prompt,
                domain=domain,
                document=document,
            )
            recovery_failures: list[dict[str, str]] = []
            token = _PAGE_GROUNDING_CONTEXT.set(
                {
                    "agentic_module": agentic_module,
                    "router": router,
                    "prompt": prompt,
                    "domain": domain,
                    "document": document,
                    "domain_id": domain_id,
                    "domain_key": domain_key,
                    "failures": recovery_failures,
                }
            )
            try:
                return original(*args, **kwargs)
            finally:
                _PAGE_GROUNDING_CONTEXT.reset(token)

    phase_domain_retry._mmm_phase_domain_retry_v2 = True  # type: ignore[attr-defined]
    rag._research_document_domain = phase_domain_retry


def install_pipeline_hardening_v6() -> None:
    """Compatibility no-op; canonical pre-design research has no runtime hardening patch."""
    global _INSTALLED
    _INSTALLED = True


install_pipeline_hardening_v6()


__all__ = [
    "_grounded_merge",
    "_target_resolution_is_deferred",
    "install_pipeline_hardening_v6",
]
