from __future__ import annotations

from pathlib import Path

ROOT = Path("minecraft_mod_ai")


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"marker not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, count), encoding="utf-8")


# 1) Text-native design contract: keep callers compatible and make the model contract explicit.
p = ROOT / "agentic_research_game_design.py"
replace(
    p,
    '    fields: Sequence[str],\n    host_properties: Mapping[str, Any],\n    research: Mapping[str, Any],',
    '    fields: Sequence[str],\n    host_properties: Mapping[str, Any] | None = None,\n    research: Mapping[str, Any],',
)
replace(
    p,
    '        + " Preserve exact approved requirement IDs. No JSON, code fences, <think>, analysis, "',
    '        + " Preserve exact approved requirement IDs. Write design content as Markdown, not JSON. "\n        + "No code fences, <think>, analysis, "',
)

# 2) External GitHub unauthenticated rate-limit safety: retain two-request headroom.
p = ROOT / "pre_design_external_source_contract.py"
replace(
    p,
    '    provider_cap = _HARD_MAX_QUERIES if _github_token() else 10\n',
    '    provider_cap = _HARD_MAX_QUERIES if _github_token() else 8\n',
)

# 3) Fusion relevance: Modrinth is ecosystem-identified; generic GitHub needs strong semantic overlap.
p = ROOT / "pre_design_rag_fusion.py"
replace(
    p,
    '''    if external and not (record_terms & ecosystem):\n        return False\n    return bool(intent & record_terms)\n''',
    '''    overlap = intent & record_terms\n    provider_ecosystem = "modrinth" in source_type or source_id.startswith("modrinth:") or "modrinth.com/" in url\n    if external and not provider_ecosystem and not (record_terms & ecosystem) and len(overlap) < 2:\n        return False\n    return bool(overlap)\n''',
)

# 4) Explicit provider requirements still fail closed; ordinary pre-design donor absence stays advisory.
p = ROOT / "pre_design_research_pipeline.py"
replace(
    p,
    '''def _validate_domain_provider_grounding(\n    domain: Mapping[str, Any],\n    grounded: Mapping[str, Any],\n) -> None:\n    # Target-neutral donor retrieval is advisory; exact target/API checks fail closed later.\n    del domain, grounded\n    return None\n''',
    '''def _validate_domain_provider_grounding(\n    domain: Mapping[str, Any],\n    grounded: Mapping[str, Any],\n) -> None:\n    # Ordinary target-neutral donor discovery is advisory.  An explicitly declared\n    # required provider is different: the host promised that evidence source and must\n    # fail closed if it is absent.\n    required = {str(value).strip().casefold() for value in domain.get("required_providers", ()) if str(value).strip()}\n    if not required:\n        return\n    available: set[str] = set()\n    for row in grounded.get("queries", ()) if isinstance(grounded, Mapping) else ():\n        if not isinstance(row, Mapping):\n            continue\n        for record in row.get("evidence_records", ()):\n            if not isinstance(record, Mapping):\n                continue\n            if _is_github_record(record):\n                available.add("github")\n            source_type = str(record.get("source_type") or "").casefold()\n            source_id = str(record.get("source_id") or "").casefold()\n            if "modrinth" in source_type or source_id.startswith("modrinth:"):\n                available.add("modrinth")\n            section = str(record.get("retrieval_section") or "").casefold()\n            if section == "project_rag" or source_id.startswith("project:"):\n                available.add("project_rag")\n    missing = sorted(required - available)\n    if missing:\n        raise PreDesignResearchFailure(\n            "pre-design required provider evidence is missing: " + ", ".join(missing)\n        )\n''',
)

# 5) Legacy corrective module remains non-canonical, but its diagnostics must not promote page-local gaps to blockers.
p = ROOT / "pre_design_rag_corrective.py"
replace(
    p,
    '''def _round_is_terminally_sufficient(\n    summary: Mapping[str, Any],\n    *,\n    page_gaps: Any = (),\n    support_rejections: Any = (),\n) -> bool:\n    """Require verified claims and zero unresolved evidence obligations."""\n    return (\n        _round_has_verified_claims(summary)\n        and not _stable_text(page_gaps)\n        and not _stable_text(support_rejections)\n    )\n''',
    '''def _round_is_terminally_sufficient(\n    summary: Mapping[str, Any],\n    *,\n    page_gaps: Any = (),\n    support_rejections: Any = (),\n) -> bool:\n    """Verified evidence is sufficient for this advisory legacy reader.\n\n    Page-local omissions and rejected sibling candidates are diagnostics, not authored\n    requirement obligations.  The canonical small-model owner no longer uses this state\n    machine, but callers that still exercise it must preserve the same semantics.\n    """\n    del page_gaps, support_rejections\n    return _round_has_verified_claims(summary)\n''',
)
replace(
    p,
    '        summary["gaps"] = list(active_page_gaps)\n',
    '        summary["gaps"] = [] if claims else list(active_page_gaps)\n',
)
replace(
    p,
    '        if active_support_rejections:\n            reasons.append("unresolved support rejections remain")\n',
    '        if active_support_rejections and not claims:\n            reasons.append("unresolved support rejections remain")\n',
)
replace(
    p,
    '                "gap_semantics": "unresolved_page_or_support_gap_blocks_domain_sufficiency",\n',
    '                "gap_semantics": "page_local_diagnostic_not_domain_blocker",\n',
)

# 6) Bounded legacy runtime compatibility: diagnostics/normalizer are optional hooks, never new failure points.
p = ROOT / "runtime_stability_contract.py"
replace(
    p,
    '''        module._emit_research_progress("model_attempt", label=progress_label, attempt=1)\n        raw = ""\n''',
    '''        module._emit_research_progress("model_attempt", label=progress_label, attempt=1)\n        emit_failure = getattr(module, "_emit_bounded_failure", None)\n        normalize_json = getattr(module, "_normalize_bounded_json_text", None)\n        hash_text = getattr(module, "_sha256_text", None)\n\n        def report_failure(event: str, *, error: Exception, raw_output: str) -> None:\n            if callable(emit_failure):\n                emit_failure(\n                    event,\n                    progress_label=progress_label,\n                    raw_output=raw_output,\n                    error=error,\n                )\n\n        raw = ""\n''',
)
replace(
    p,
    '''            module._emit_bounded_failure(\n                "bounded_model_failure",\n                progress_label=progress_label,\n                raw_output=raw,\n                error=exc,\n            )\n''',
    '''            report_failure("bounded_model_failure", error=exc, raw_output=raw)\n''',
)
replace(
    p,
    '''            module._emit_bounded_failure(\n                "bounded_parse_failure",\n                progress_label=progress_label,\n                raw_output=raw,\n                error=first_error,\n            )\n            try:\n                normalized = module._normalize_bounded_json_text(raw)\n                parsed = parser(normalized)\n            except Exception as normalized_error:\n                module._emit_bounded_failure(\n                    "bounded_host_normalization_failure",\n                    progress_label=progress_label,\n                    raw_output=raw,\n                    error=normalized_error,\n                )\n                raise module._BoundedResearchOutputError(\n                    "bounded structured output failed after deterministic host normalization: "\n                    f"{type(normalized_error).__name__}: {normalized_error}"\n                ) from normalized_error\n''',
    '''            report_failure("bounded_parse_failure", error=first_error, raw_output=raw)\n            try:\n                normalized = normalize_json(raw) if callable(normalize_json) else raw\n                parsed = parser(normalized)\n            except Exception as normalized_error:\n                report_failure(\n                    "bounded_host_normalization_failure",\n                    error=normalized_error,\n                    raw_output=raw,\n                )\n                raise module._BoundedResearchOutputError(\n                    "bounded structured output failed after host repair: "\n                    f"{type(normalized_error).__name__}: {normalized_error}"\n                ) from normalized_error\n''',
)
replace(
    p,
    '                raw_output_sha256=module._sha256_text(raw),\n',
    '                raw_output_sha256=hash_text(raw) if callable(hash_text) else "",\n',
)

# 7) New small-model owner must not require optional receipt helpers from test/minimal RAG hosts.
p = ROOT / "small_model_predesign_research.py"
replace(
    p,
    '''def research_document_domain(\n''',
    '''def _document_receipt(project_rag: Any, document: Mapping[str, Any]) -> dict[str, Any]:\n    receipt = getattr(project_rag, "_prompt_document_receipt", None)\n    if callable(receipt):\n        value = receipt(document)\n        return dict(value) if isinstance(value, Mapping) else {"value": value}\n    return {\n        "schema_version": "mmm/research-evidence-document-receipt-v1",\n        "domain_id": str(document.get("domain_id") or ""),\n        "document_sha256": str(document.get("document_sha256") or ""),\n        "page_count": int(document.get("page_count") or 0),\n        "raw_path": str(document.get("raw_path") or ""),\n        "pages_path": str(document.get("pages_path") or ""),\n    }\n\n\ndef research_document_domain(\n''',
)
replace(
    p,
    '        "evidence_document": project_rag._prompt_document_receipt(working_document),\n',
    '        "evidence_document": _document_receipt(project_rag, working_document),\n',
)

# 8) Keep old pure merge helpers as compatibility utilities without restoring the old corrective execution owner.
p = ROOT / "pre_design_domain_research.py"
p.write_text('''from __future__ import annotations\n\nfrom collections.abc import Mapping, Sequence\nfrom typing import Any\n\n# Canonical pre-design research entrypoint. The previous corrective/page-gap state\n# machine is intentionally not on the execution path.\nfrom .small_model_predesign_research import research_document_domain\n\n\ndef _root_page_claims(notes: Sequence[Mapping[str, Any]], *, page_ref: str) -> list[dict[str, Any]]:\n    result: list[dict[str, Any]] = []\n    for note in notes:\n        for raw in note.get("claims", ()) if isinstance(note, Mapping) else ():\n            if not isinstance(raw, Mapping):\n                continue\n            claim = dict(raw)\n            if not str(claim.get("claim") or "").strip():\n                continue\n            claim["evidence_refs"] = [page_ref]\n            result.append(claim)\n    return result\n\n\ndef _merge_page_notes(\n    domain_id: str,\n    page_notes: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],\n) -> dict[str, Any]:\n    claims: list[dict[str, Any]] = []\n    gaps: list[str] = []\n    next_queries: list[str] = []\n    procedures: list[Any] = []\n    for page_ref, notes in page_notes:\n        claims.extend(_root_page_claims(notes, page_ref=page_ref))\n        for note in notes:\n            if not isinstance(note, Mapping):\n                continue\n            for value in note.get("gaps", ()):\n                text = str(value).strip()\n                if text and text not in gaps:\n                    gaps.append(text)\n            for value in note.get("next_queries", ()):\n                text = str(value).strip()\n                if text and text not in next_queries:\n                    next_queries.append(text)\n            for value in note.get("procedures", ()):\n                if value not in procedures:\n                    procedures.append(value)\n    sufficient = bool(claims)\n    if not sufficient and not gaps:\n        gaps.append("No evidence-backed design claim was extracted from the host-issued pages.")\n    return {\n        "domain_id": domain_id,\n        "claims": claims,\n        "gaps": [] if sufficient else gaps,\n        "next_queries": next_queries,\n        "procedures": procedures,\n        "sufficient": sufficient,\n        "fixed_point": False,\n    }\n\n\n__all__ = ["research_document_domain", "_root_page_claims", "_merge_page_notes"]\n''', encoding="utf-8")

# 9) The old direct-owner test asserted the removed corrective implementation. Replace it with a test of the new canonical owner.
p = Path("tests/test_pre_design_rag_direct_owner.py")
p.write_text('''from __future__ import annotations\n\nfrom minecraft_mod_ai import pre_design_domain_research as owner\n\n\ndef test_direct_owner_is_small_model_host_pipeline_and_missing_receipt_helper_is_safe():\n    calls: list[dict[str, object]] = []\n\n    class Rag:\n        @staticmethod\n        def _read_evidence_pages(document):\n            del document\n            return [\n                {\n                    "page_ref": "sha256:noise#page=1/1",\n                    "content": "Unrelated finance dashboard material.",\n                }\n            ]\n\n    class Router:\n        def generate_text(self, role, messages, **kwargs):\n            calls.append({"role": role, "messages": messages, **kwargs})\n            return "NONE"\n\n    document = {\n        "domain_id": "req_colony",\n        "document_sha256": "sha256:doc",\n        "page_count": 1,\n    }\n    note = owner.research_document_domain(\n        object(),\n        Rag,\n        Router(),\n        prompt="식민지",\n        domain={\n            "domain_id": "req_colony",\n            "objective": "persistent colony mechanics",\n            "queries": ["minecraft persistent colony mechanics"],\n        },\n        document=document,\n        trace_metadata=None,\n    )\n\n    assert owner.research_document_domain.__module__.endswith("small_model_predesign_research")\n    assert len(calls) == 1\n    assert calls[0]["response_format"] == "text"\n    assert calls[0]["response_schema"] is None\n    assert note["research_mode"] == "advisory_predesign"\n    assert note["research_evidence_status"] == "no_relevant_external_evidence"\n    assert note["sufficient"] is True\n    assert note["gaps"] == []\n    assert note["quality_contract"]["model_json"] is False\n    assert note["quality_contract"]["model_corrective_queries"] is False\n    assert note["evidence_document"]["document_sha256"] == "sha256:doc"\n''', encoding="utf-8")

print("integrated CI contract repair applied")
