from __future__ import annotations

from minecraft_mod_ai import pre_design_domain_research as research


def test_canonical_predesign_path_is_host_grounded_without_model_synthesis():
    assert research.research_document_domain.__module__ == (
        "minecraft_mod_ai.pre_design_domain_research"
    )


def test_irrelevant_page_fails_closed_without_spending_model_turn():
    calls: list[object] = []

    class Router:
        def generate_text(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("host evidence projection must not call the model")

    class Project:
        @staticmethod
        def _read_evidence_pages(document):
            del document
            return [
                {
                    "page_ref": "host#1",
                    "content": "Microsoft Build Student Zone learning path unrelated material",
                }
            ]

    note = research.research_document_domain(
        object(),
        Project(),
        Router(),
        prompt="식민지화 우주 모드",
        domain={
            "domain_id": "request",
            "objective": "space colonization",
            "queries": ["space colonization persistence"],
        },
        document={"page_count": 1},
        trace_metadata=None,
    )
    assert calls == []
    assert note["model_called"] is False
    assert note["source_body_count"] == 1
    assert note["host_grounded_evidence_card_count"] == 0
    assert note["sufficient"] is False
    assert note["fixed_point"] is False
    assert note["gaps"]
    assert note["checkpoint"]["status"] == "blocked"
    assert note["research_evidence_status"] == "no_relevant_external_evidence"


def test_relevant_materialized_body_becomes_exact_host_evidence_without_model_turn():
    class Project:
        @staticmethod
        def _read_evidence_pages(document):
            del document
            return [
                {
                    "page_ref": "host#1",
                    "content": (
                        "Space colony state persists across server restarts. "
                        "Unrelated trailing material."
                    ),
                }
            ]

    class Router:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("host evidence projection must not call the model")

    note = research.research_document_domain(
        object(),
        Project(),
        Router(),
        prompt="persistent colony",
        domain={
            "domain_id": "request",
            "objective": "persistent colony state",
            "queries": ["colony state persistence"],
        },
        document={"page_count": 1},
        trace_metadata=None,
    )

    assert note["model_called"] is False
    assert note["sufficient"] is True
    assert note["checkpoint"]["status"] == "complete"
    assert note["claims"]
    assert note["claims"][0]["evidence_refs"] == ["host#1"]
    assert note["grounded_evidence_cards"][0]["verification"] == (
        "host_exact_substring_from_materialized_source_page"
    )
