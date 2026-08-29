from __future__ import annotations

"""Final pipeline hardening for semantic seed search and lossless page research."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline_hardening import _base_evidence_ref, _replace_bound_references

_INSTALLED = False
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9.+]+|[가-힣]{2,}")
_SEED_NOISE = frozenset(
    {
        "minecraft",
        "mod",
        "mods",
        "make",
        "create",
        "system",
        "semantic",
        "implementation",
        "implement",
        "task",
        "feature",
        "mechanic",
        "module",
        "generated",
        "generator",
        "interaction",
        "logic",
        "code",
        "design",
        "plan",
        "planning",
        "research",
        "fabric",
        "forge",
        "neoforge",
        "version",
        "java",
        "with",
        "that",
        "this",
        "the",
        "and",
        "for",
        "please",
    }
)


def _query_tokens(value: Any) -> list[str]:
    text = _CAMEL_BOUNDARY.sub(" ", str(value or ""))
    text = re.sub(r"[_/:\-]+", " ", text)
    return [
        token
        for token in _QUERY_TOKEN.findall(text)
        if len(token.strip()) >= 2
        and token.casefold() not in _SEED_NOISE
    ]


def bounded_seed_query(prompt: str, game_design: Mapping[str, Any]) -> str:
    """Build a short, high-signal ecosystem query instead of serializing the plan."""

    sources: list[Any] = []
    sources.append(game_design.get("title", ""))

    modules = game_design.get("modules")
    if isinstance(modules, Sequence) and not isinstance(modules, (str, bytes, bytearray)):
        for item in modules:
            if not isinstance(item, Mapping):
                continue
            sources.extend(
                (
                    item.get("name", ""),
                    item.get("kind", ""),
                    item.get("plugin_id", ""),
                    item.get("reason", ""),
                )
            )

    capabilities = game_design.get("capabilities")
    if isinstance(capabilities, Sequence) and not isinstance(
        capabilities, (str, bytes, bytearray)
    ):
        sources.extend(capabilities)

    sources.extend(
        (
            game_design.get("pitch", ""),
            prompt,
        )
    )

    tokens: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for token in _query_tokens(source):
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
            if len(tokens) >= 16:
                break
        if len(tokens) >= 16:
            break

    if not tokens:
        fallback = " ".join(str(prompt or "").split()).strip()
        return fallback[:240] or "gameplay"

    query = " ".join(tokens)
    return query[:320].rstrip()


def _strict_provenance_repair(
    note: Mapping[str, Any],
    *,
    allowed_refs: Sequence[str],
) -> dict[str, Any]:
    """Filter provenance without fabricating evidence references.

    A claim whose cited references do not survive the host allow-list is omitted.
    Assigning every child page to an uncited claim would satisfy the shape validator
    while destroying the evidence/claim relation, so this fails closed instead.
    """

    result = dict(note)
    allowed = tuple(
        dict.fromkeys(
            _base_evidence_ref(ref)
            for ref in allowed_refs
            if _base_evidence_ref(ref)
        )
    )
    allowed_set = set(allowed)

    claims: list[Any] = []
    dropped = 0
    for claim in result.get("claims", ()):
        if not isinstance(claim, Mapping):
            dropped += 1
            continue
        item = dict(claim)
        refs = [
            _base_evidence_ref(ref)
            for ref in item.get("evidence_refs", ())
            if _base_evidence_ref(ref) in allowed_set
        ]
        refs = list(dict.fromkeys(refs))
        if not refs:
            dropped += 1
            continue
        item["evidence_refs"] = refs
        claims.append(item)
    if "claims" in result:
        result["claims"] = claims

    procedures: list[Any] = []
    for procedure in result.get("procedures", ()):
        if not isinstance(procedure, Mapping):
            continue
        item = dict(procedure)
        if "evidence_refs" in item:
            refs = [
                _base_evidence_ref(ref)
                for ref in item.get("evidence_refs", ())
                if _base_evidence_ref(ref) in allowed_set
            ]
            refs = list(dict.fromkeys(refs))
            if not refs:
                continue
            item["evidence_refs"] = refs
        procedures.append(item)
    if "procedures" in result:
        result["procedures"] = procedures

    if dropped:
        gaps = [str(value) for value in result.get("gaps", ()) if str(value).strip()]
        gaps.append(
            f"{dropped} synthesized claim(s) were omitted because no host-issued "
            "evidence reference survived provenance validation."
        )
        result["gaps"] = gaps
        if not claims:
            result["sufficient"] = False
    return result


def _install_semantic_seed_search() -> None:
    from . import ecosystem_discovery as ecosystem

    original = ecosystem._seed_query
    if getattr(original, "_mmm_bounded_semantic_seed_query", False):
        return

    def seed_query(prompt: str, game_design: dict[str, Any]) -> str:
        return bounded_seed_query(prompt, game_design)

    seed_query._mmm_bounded_semantic_seed_query = True  # type: ignore[attr-defined]
    ecosystem._seed_query = seed_query
    _replace_bound_references(original, seed_query)


def _install_strict_provenance_filter() -> None:
    from . import pipeline_hardening as v1

    v1._repair_note_provenance = _strict_provenance_repair


def _install_lossless_page_research() -> None:
    from . import agentic_pre_design_rag as rag

    original = rag._research_document_domain
    if getattr(original, "_mmm_lossless_page_research", False):
        return

    def research_document_domain(
        agentic_module: Any,
        router: Any,
        *,
        prompt: str,
        domain: Mapping[str, Any],
        document: Mapping[str, Any],
        trace_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        del trace_metadata
        domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
        domain_key = rag._domain_checkpoint_key(
            router,
            prompt=prompt,
            domain=domain,
            document=document,
        )
        with rag._domain_lock(domain_key):
            cached = rag._read_complete_manifest(agentic_module, domain_key, domain_id)
            if cached is not None:
                rag._emit_research_progress(
                    "domain_checkpoint_complete",
                    domain_id=domain_id,
                    manifest_path=str(rag._manifest_path(domain_key)),
                    checkpoint_dir=str(rag._checkpoint_dir(domain_key)),
                    note=cached,
                )
                return cached

            pages = rag._read_evidence_pages(document)
            failures: list[dict[str, str]] = []
            page_notes: list[dict[str, Any]] = []
            rag._emit_research_progress(
                "domain_start",
                domain_id=domain_id,
                page_count=len(pages),
                evidence_document=rag._prompt_document_receipt(document),
                evidence_pages_path=document.get("pages_path"),
                evidence_raw_path=document.get("raw_path"),
                checkpoint_dir=str(rag._checkpoint_dir(domain_key)),
            )

            for page_index, page in enumerate(pages):
                page_ref = str(page.get("page_ref", ""))
                rag._emit_research_progress(
                    "page_start",
                    domain_id=domain_id,
                    page_index=page_index + 1,
                    page_count=len(pages),
                    page_ref=page_ref,
                )
                extracted = rag._read_page_losslessly(
                    agentic_module,
                    router,
                    prompt=prompt,
                    domain=domain,
                    document=document,
                    page=page,
                    domain_key=domain_key,
                    progress_label=f"{domain_id} page {page_index + 1}/{len(pages)}",
                    failures=failures,
                )
                page_notes.extend(extracted)
                rag._emit_research_progress(
                    "page_grounded",
                    domain_id=domain_id,
                    page_index=page_index + 1,
                    page_count=len(pages),
                    page_ref=page_ref,
                    note_count=len(extracted),
                    claim_count=sum(
                        len(note.get("claims", ()))
                        for note in extracted
                        if isinstance(note, Mapping)
                    ),
                )

            summary = rag._hierarchical_synthesis(
                agentic_module,
                router,
                prompt=prompt,
                domain=domain,
                page_notes=page_notes,
                domain_key=domain_key,
                failures=failures,
            )
            claims = rag._stable_unique_claims([*page_notes, summary])
            catalog = rag._materialize_claim_catalog(
                domain_key,
                domain_id,
                claims,
            )
            evidence_ledger = rag._materialize_evidence_ledger(
                domain_key,
                domain_id,
                pages,
            )

            failure_reasons: list[str] = []
            if failures:
                failure_reasons.append("bounded page read or synthesis failure")
            if summary.get("sufficient") is not True:
                failure_reasons.append("synthesis returned sufficient=false")
            if not claims:
                failure_reasons.append("synthesis produced zero grounded claims")

            status = "failed" if failure_reasons else "complete"
            note: dict[str, Any] = {
                **rag._core_note(summary),
                "evidence_document": rag._prompt_document_receipt(document),
                "claim_catalog": catalog,
                "evidence_ledger": evidence_ledger,
                "checkpoint": {
                    "schema_version": rag._DOMAIN_CHECKPOINT_SCHEMA,
                    "request_sha256": "sha256:" + domain_key,
                    "status": status,
                    "manifest_path": str(rag._manifest_path(domain_key)),
                    "checkpoint_dir": str(rag._checkpoint_dir(domain_key)),
                },
            }
            if failures:
                existing_gaps = [str(item) for item in note.get("gaps", ())]
                note["gaps"] = existing_gaps + [
                    f"{item['unit']}: {item['error']}" for item in failures
                ]
                note["research_failures"] = list(failures)
            if failure_reasons:
                note["sufficient"] = False
                note["fixed_point"] = True
                note["failure_reasons"] = failure_reasons

            rag._write_manifest(
                domain_key,
                status=status,
                note=note,
                failures=failures,
            )

            if status != "complete":
                rag._emit_research_progress(
                    "domain_failure",
                    domain_id=domain_id,
                    status=status,
                    failure_reasons=failure_reasons,
                    failures=failures,
                    summary=summary,
                    claim_catalog=catalog,
                    evidence_ledger=evidence_ledger,
                    evidence_document=rag._prompt_document_receipt(document),
                    manifest_path=str(rag._manifest_path(domain_key)),
                    checkpoint_dir=str(rag._checkpoint_dir(domain_key)),
                    note=note,
                )
                raise rag._BoundedResearchOutputError(
                    "pre-design research failed closed for domain "
                    f"{domain_id!r}: {'; '.join(failure_reasons)}; "
                    f"manifest={rag._manifest_path(domain_key)}"
                )

            rag._emit_research_progress(
                "domain_complete",
                domain_id=domain_id,
                status=status,
                claim_count=catalog["claim_count"],
                procedure_count=len(note.get("procedures", ())),
                page_count=len(pages),
                failure_count=0,
                claim_catalog=catalog,
                evidence_ledger=evidence_ledger,
                manifest_path=str(rag._manifest_path(domain_key)),
                checkpoint_dir=str(rag._checkpoint_dir(domain_key)),
            )
            return note

    research_document_domain._mmm_lossless_page_research = True  # type: ignore[attr-defined]
    rag._research_document_domain = research_document_domain
    _replace_bound_references(original, research_document_domain)


def install_pipeline_hardening_v4() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_semantic_seed_search()
    _install_strict_provenance_filter()
    _install_lossless_page_research()
    _INSTALLED = True


__all__ = [
    "_strict_provenance_repair",
    "bounded_seed_query",
    "install_pipeline_hardening_v4",
]
