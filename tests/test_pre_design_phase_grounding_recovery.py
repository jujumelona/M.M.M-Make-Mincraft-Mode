from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agentic_pre_design_rag as production_rag
from minecraft_mod_ai import runtime_contracts as contracts


class _BoundedResearchOutputError(RuntimeError):
    pass


def _grounded_note(domain_id: str = "request") -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "claims": [
            {
                "claim": "Target-neutral architecture evidence is usable during pre-design.",
                "evidence_refs": ["sha256:evidence#page=1/1"],
            }
        ],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
        "page_ref": "sha256:evidence#page=1/1",
    }


def _fake_domain_rag(*, page_has_claim: bool) -> SimpleNamespace:
    rag = SimpleNamespace()
    rag._BoundedResearchOutputError = _BoundedResearchOutputError
    rag.calls = {"domain": 0, "page": 0}

    def host_page_note(domain_id, page):
        return {
            "domain_id": domain_id,
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "procedures": [],
            "sufficient": True,
            "evidence_fragment": {
                "page_ref": page["page_ref"],
                "content": page["content"],
            },
        }

    def read_page_losslessly(
        agentic_module,
        router,
        *,
        prompt,
        domain,
        document,
        page,
        domain_key,
        progress_label,
        failures,
    ):
        del agentic_module, router, prompt, document, domain_key, progress_label, failures
        rag.calls["page"] += 1
        return [_grounded_note(str(domain["domain_id"]))] if page_has_claim else []

    def domain_checkpoint_key(router, *, prompt, domain, document):
        del router, prompt, domain, document
        return "checkpoint"

    def research_document_domain(
        agentic_module,
        router,
        *,
        prompt,
        domain,
        document,
        trace_metadata,
    ):
        del agentic_module, router, prompt, trace_metadata
        rag.calls["domain"] += 1
        notes = [rag._host_page_note(domain["domain_id"], page) for page in document["pages"]]
        claims = [claim for note in notes for claim in note.get("claims", ())]
        if not claims:
            raise _BoundedResearchOutputError(
                "synthesis returned sufficient=false; synthesis produced zero grounded claims"
            )
        return {"claims": claims, "sufficient": True}

    rag._host_page_note = host_page_note
    rag._read_page_losslessly = read_page_losslessly
    rag._domain_checkpoint_key = domain_checkpoint_key
    rag._research_document_domain = research_document_domain
    return rag


def test_runtime_contracts_install_without_numbered_wrapper_markers() -> None:
    assert getattr(production_rag._research_document_domain, "_mmm_target_neutral_domain_retry", False)
    assert not hasattr(production_rag._research_document_domain, "_mmm_phase_domain_retry_v2")


def test_provenance_filter_rejects_unissued_or_missing_refs() -> None:
    repaired = contracts._repair_note_provenance(
        {
            "domain_id": "request",
            "claims": [
                {"claim": "valid", "evidence_refs": ["sha256:page"]},
                {"claim": "invented", "evidence_refs": ["sha256:not-issued"]},
                {"claim": "uncited", "evidence_refs": []},
            ],
            "gaps": [],
            "sufficient": True,
        },
        allowed_refs=["sha256:page"],
    )
    assert repaired["claims"] == [
        {"claim": "valid", "evidence_refs": ["sha256:page"]}
    ]
    assert repaired["sufficient"] is True

    empty = contracts._repair_note_provenance(
        {"claims": [{"claim": "uncited", "evidence_refs": []}], "sufficient": True},
        allowed_refs=["sha256:page"],
    )
    assert empty["claims"] == []
    assert empty["sufficient"] is False


def test_zero_claim_domain_retries_with_page_grounding() -> None:
    rag = _fake_domain_rag(page_has_claim=True)
    original_host = production_rag._host_page_note
    original_domain = production_rag._research_document_domain
    try:
        production_rag._host_page_note = rag._host_page_note
        production_rag._read_page_losslessly = rag._read_page_losslessly
        production_rag._domain_checkpoint_key = rag._domain_checkpoint_key
        production_rag._research_document_domain = rag._research_document_domain
        contracts._install_target_neutral_grounding()
        result = production_rag._research_document_domain(
            object(),
            object(),
            prompt="make a mod",
            domain={
                "domain_id": "request",
                "target_frozen": False,
                "target_state": "target-neutral",
            },
            document={
                "pages": [
                    {
                        "page_ref": "sha256:evidence#page=1/1",
                        "content": "official target-neutral architecture evidence",
                    }
                ]
            },
            trace_metadata=None,
        )
        assert result["sufficient"] is True
        assert result["claims"]
        assert rag.calls == {"domain": 2, "page": 1}
    finally:
        production_rag._host_page_note = original_host
        production_rag._research_document_domain = original_domain


def test_page_grounding_failure_remains_fail_closed() -> None:
    rag = _fake_domain_rag(page_has_claim=False)
    original_host = production_rag._host_page_note
    original_domain = production_rag._research_document_domain
    try:
        production_rag._host_page_note = rag._host_page_note
        production_rag._read_page_losslessly = rag._read_page_losslessly
        production_rag._domain_checkpoint_key = rag._domain_checkpoint_key
        production_rag._research_document_domain = rag._research_document_domain
        contracts._install_target_neutral_grounding()
        with pytest.raises(_BoundedResearchOutputError, match="zero grounded claims"):
            production_rag._research_document_domain(
                object(),
                object(),
                prompt="make a mod",
                domain={
                    "domain_id": "request",
                    "target_frozen": False,
                    "target_state": "target-neutral",
                },
                document={
                    "pages": [
                        {
                            "page_ref": "sha256:evidence#page=1/1",
                            "content": "no grounded design statement",
                        }
                    ]
                },
                trace_metadata=None,
            )
        assert rag.calls == {"domain": 2, "page": 1}
    finally:
        production_rag._host_page_note = original_host
        production_rag._research_document_domain = original_domain


def test_exact_target_domain_does_not_use_phase_retry() -> None:
    rag = _fake_domain_rag(page_has_claim=True)
    original_host = production_rag._host_page_note
    original_domain = production_rag._research_document_domain
    try:
        production_rag._host_page_note = rag._host_page_note
        production_rag._read_page_losslessly = rag._read_page_losslessly
        production_rag._domain_checkpoint_key = rag._domain_checkpoint_key
        production_rag._research_document_domain = rag._research_document_domain
        contracts._install_target_neutral_grounding()
        with pytest.raises(_BoundedResearchOutputError, match="zero grounded claims"):
            production_rag._research_document_domain(
                object(),
                object(),
                prompt="make a mod",
                domain={
                    "domain_id": "request",
                    "target_frozen": True,
                    "target_state": "frozen",
                },
                document={
                    "pages": [
                        {
                            "page_ref": "sha256:evidence#page=1/1",
                            "content": "target-specific evidence is required here",
                        }
                    ]
                },
                trace_metadata=None,
            )
        assert rag.calls == {"domain": 1, "page": 0}
    finally:
        production_rag._host_page_note = original_host
        production_rag._research_document_domain = original_domain
