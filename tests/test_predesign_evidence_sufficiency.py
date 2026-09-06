from __future__ import annotations

import json

from minecraft_mod_ai.pre_design_domain_research import research_document_domain


class _ProjectRag:
    def __init__(self, pages):
        self.pages = list(pages)

    def _read_evidence_pages(self, document):
        return list(self.pages)


def _research(project_rag):
    return research_document_domain(
        object(),
        project_rag,
        object(),
        prompt="Build an energy storage mod",
        domain={
            "domain_id": "r_001",
            "objective": "Find energy storage implementation evidence",
            "requirements": ["persistent energy storage behavior"],
            "queries": ["Minecraft energy storage implementation"],
        },
        document={"page_count": len(project_rag.pages)},
        trace_metadata=None,
    )


def test_zero_materialized_source_bodies_are_not_sufficient() -> None:
    note = _research(_ProjectRag([]))

    assert note["sufficient"] is False
    assert note["checkpoint"]["status"] == "blocked"
    assert note["evidence_extraction_status"] == "no_claim_bearing_source_body"
    assert note["source_body_count"] == 0
    assert note["claims"] == []


def test_exact_materialized_source_excerpt_is_sufficient() -> None:
    source = {
        "source_id": "curseforge:1",
        "source_type": "curseforge_mod_body",
        "url": "https://example.invalid/mod",
        "title": "Energy Example",
        "content_sha256": "sha256:test",
        "content": "Persistent energy storage is owned by the server and survives reload.",
    }
    page = {
        "page_ref": "sha256:doc#page=1/1",
        "content": json.dumps(source),
    }

    note = _research(_ProjectRag([page]))

    assert note["sufficient"] is True
    assert note["checkpoint"]["status"] == "complete"
    assert note["source_body_count"] == 1
    assert len(note["claims"]) == 1
    assert note["claims"][0]["evidence_refs"] == ["sha256:doc#page=1/1"]
