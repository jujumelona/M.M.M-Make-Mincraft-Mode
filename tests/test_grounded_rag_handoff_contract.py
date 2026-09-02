from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import model_meta_output_contract as meta
from minecraft_mod_ai import pre_design_domain_research as domain_research
from minecraft_mod_ai.research_evidence_state import record_grounded_evidence


class _ProjectRag:
    @staticmethod
    def _read_evidence_pages(_document):
        return [
            {
                "page_ref": "page:alpha",
                "content": (
                    "Fabric persistent state can be synchronized from server-owned data. "
                    "A trading screen should treat the server as authoritative."
                ),
            },
            {
                "page_ref": "page:beta",
                "content": (
                    "Repository source demonstrates a registry-backed item implementation. "
                    "The source is pinned by the host before reuse."
                ),
            },
        ]


def _domain() -> dict[str, object]:
    return {
        "domain_id": "request",
        "objective": "persistent trading and reusable item implementation",
        "requirements": ["persistent trading", "item implementation"],
        "queries": ["server authoritative trading", "registry item source"],
    }


def test_source_bodies_survive_empty_small_model_extraction(monkeypatch) -> None:
    monkeypatch.setattr(
        domain_research,
        "_small_model_research_document_domain",
        lambda *_args, **_kwargs: {
            "domain_id": "request",
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "procedures": [],
            "sufficient": True,
            "fixed_point": False,
            "research_mode": "advisory_predesign",
            "research_evidence_status": "no_relevant_external_evidence",
            "source_body_count": 2,
            "model_called": True,
            "page_local_diagnostics": ["ignored_malformed_model_line"],
        },
    )

    note = domain_research.research_document_domain(
        object(),
        _ProjectRag(),
        object(),
        prompt="persistent trading with reusable item code",
        domain=_domain(),
        document={"domain_id": "request"},
        trace_metadata=None,
    )

    assert note["research_evidence_status"] == "partial"
    assert note["evidence_extraction_status"] == (
        "host_source_evidence_available_model_exact_claim_absent"
    )
    assert note["model_grounded_claim_count"] == 0
    assert note["host_grounded_evidence_card_count"] == 2
    cards = note["grounded_evidence_cards"]
    assert {card["page_ref"] for card in cards} == {"page:alpha", "page:beta"}
    for card in cards:
        source = next(
            page["content"]
            for page in _ProjectRag._read_evidence_pages({})
            if page["page_ref"] == card["page_ref"]
        )
        assert card["exact_excerpt"] in source
        assert card["semantic_claim"] is False


def test_blanket_no_evidence_output_is_rejected_after_host_grounding() -> None:
    record_grounded_evidence(
        "rag-contract-request",
        source_body_count=2,
        evidence_card_count=2,
    )
    with pytest.raises(ValueError, match="contradicts host-grounded RAG evidence"):
        meta.assert_design_field_clean(
            "core_loop",
            ["research_evidence_status: no_relevant_external_evidence found for any domain"],
        )


def test_specific_unresolved_fact_is_not_rejected_as_blanket_absence() -> None:
    record_grounded_evidence(
        "rag-contract-request-specific",
        source_body_count=2,
        evidence_card_count=2,
    )
    meta.assert_design_field_clean(
        "core_loop",
        ["No external evidence was found for the exact future mapping signature."],
    )
